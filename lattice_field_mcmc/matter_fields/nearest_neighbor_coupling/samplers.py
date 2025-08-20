import gc
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union

from .bases import ScalarLatticeSampler2D, ThermalLatticeSampler2D

# =============================================================================
# Phi-4 Model Implementation
# =============================================================================
class ScalarPhi4Model2D(ScalarLatticeSampler2D):
    r"""Implements the 2D scalar Phi-4 lattice field theory.

    This class simulates a system with the action:
    S = \sum_x [ -2\kappa \sum_\mu \phi_x \phi_{x+\mu} +
                 (1 - 2\lambda) \phi^2 + \lambda \phi^4 ]
    where the phase transition is primarily driven by the hopping parameter
    kappa. The model includes a feature for dynamically tuning the Metropolis
    proposal width to maintain a target acceptance rate.

    Attributes:
        lambda_ (Tensor): The self-interaction coupling constant.
        proposal_widths (Tensor): The current Metropolis proposal width for
            each kappa value. Shape: [batch_size].
        target_acceptance_rate (float): The desired acceptance rate for the
            tuning algorithm.
        tune_interval (int): How often (in sweeps) to adjust proposal widths.
        tuning_strength (float): The strength of the adjustment factor.
    """
    def __init__(self,
                 L: int,
                 kappa: Tensor,
                 lambda_: float,
                 n_chains: int = 30,
                 proposal_width: Union[float, Tensor] = 0.2,
                 target_acceptance_rate: float = 0.5,
                 tune_interval: int = 20,
                 tuning_strength: float = 0.1,
                 device: Optional[torch.device] = None,
                 use_amp: bool = False,
                 large_size_simulate: bool = False,
                 pt_enabled: bool = True) -> None:
        """Initializes the ScalarPhi4Model2D sampler.

        Args:
            L: The linear size of the lattice.
            kappa: A 1D tensor of kappa coupling constants.
            lambda_: The lambda self-interaction coupling constant.
            n_chains: The number of independent Monte Carlo chains per kappa.
            proposal_width: The initial Metropolis proposal width. Can be a
                single float (applied to all replicas) or a 1D tensor with a
                width for each kappa value.
            target_acceptance_rate: The target acceptance rate for the dynamic
                tuning mechanism.
            tune_interval: The number of sweeps between proposal width
                adjustments.
            tuning_strength: A factor controlling how aggressively the proposal
                width is adjusted.
            device: The PyTorch device to use.
            use_amp: If True, enables Automatic Mixed Precision.
            large_size_simulate: If True, moves samples to CPU to save VRAM.
            pt_enabled: If True, enables parallel tempering.
        """
        # Call the parent constructor first.
        super().__init__(
            L=L, kappa=kappa, n_chains=n_chains, device=device,
            use_amp=use_amp, large_size_simulate=large_size_simulate,
            pt_enabled=pt_enabled
        )

        # Register model-specific parameters as buffers to ensure they are
        # moved to the correct device along with the model.
        self.register_buffer(
            'lambda_',
            torch.tensor(lambda_, dtype=torch.float32, device=self.device))

        # --- Initialize proposal widths and tuning parameters ---
        if isinstance(proposal_width, float):
            # If a single float is given, create a tensor for all replicas.
            initial_widths = torch.full(
                (self.batch_size,), proposal_width,
                dtype=torch.float32, device=self.device)
        elif isinstance(proposal_width, Tensor):
            # If a tensor is given, ensure its shape matches kappa's.
            if proposal_width.shape != (self.batch_size,):
                raise ValueError(
                    f"Shape of proposal_width tensor {proposal_width.shape} "
                    f"must match kappa shape {(self.batch_size,)}")
            initial_widths = proposal_width.to(
                self.device, dtype=torch.float32)
        else:
            raise TypeError(
                "proposal_width must be a float or a Tensor, not "
                f"{type(proposal_width)}")

        self.register_buffer('proposal_widths', initial_widths)

        # Store tuning hyperparameters.
        self.target_acceptance_rate = target_acceptance_rate
        self.tune_interval = tune_interval
        self.tuning_strength = tuning_strength

        # Buffers to track acceptance rate statistics for each kappa value.
        self.register_buffer(
            'acceptance_stats',
            torch.zeros(self.batch_size, dtype=torch.float32,
                        device=self.device))
        self.register_buffer(
            'acceptance_counter',
            torch.tensor(0, dtype=torch.long, device=self.device))

        print(f"Proposal width tuning enabled: target_rate="
              f"{target_acceptance_rate}, interval={tune_interval}")


    def init_field(self) -> None:
        """Initializes the scalar field with random values.

        The field `phi` is initialized with values drawn from a standard
        normal distribution (mean 0, variance 1).
        """
        self.field = torch.randn(
            (self.batch_size, self.n_chains, self.L, self.L),
            dtype=torch.float32, device=self.device
        )

    @torch.compile
    def metropolis_update_sub_lattice(self, lattice_color: str) -> None:
        """Performs a Metropolis update for the Phi-4 model.

        Args:
            lattice_color: 'black' or 'white', specifying the sub-lattice.
        """
        # Select the appropriate pre-computed indices for the sub-lattice.
        if lattice_color == 'black':
            (batch_idx, chain_idx, i_sites, j_sites,
             neighbors_i, neighbors_j) = (
                self.black_batch_idx, self.black_chain_idx,
                self.black_i_sites, self.black_j_sites,
                self.black_neighbors_i, self.black_neighbors_j)
        else:  # 'white'
            (batch_idx, chain_idx, i_sites, j_sites,
             neighbors_i, neighbors_j) = (
                self.white_batch_idx, self.white_chain_idx,
                self.white_i_sites, self.white_j_sites,
                self.white_neighbors_i, self.white_neighbors_j)

        # Get current field values and the sum of their neighbors.
        phi_old = self.field[batch_idx, chain_idx, i_sites, j_sites]
        phi_neighbors = self.field[
            batch_idx.unsqueeze(-1), chain_idx.unsqueeze(-1),
            neighbors_i, neighbors_j]
        sum_phi_neighbors = phi_neighbors.sum(dim=-1)

        # Propose a new field value with a random uniform update.
        # The proposal width is specific to each kappa value.
        current_proposal_widths = self.proposal_widths.view(
            self.batch_size, 1, 1)
        d_phi = (torch.rand_like(phi_old) - 0.5) * current_proposal_widths
        phi_new = phi_old + d_phi

        # Calculate the change in the local action (delta S).
        phi_old_sq = phi_old.pow(2)
        phi_new_sq = phi_new.pow(2)
        potential_old = (1 - 2 * self.lambda_) * phi_old_sq + \
                        self.lambda_ * phi_old_sq.pow(2)
        potential_new = (1 - 2 * self.lambda_) * phi_new_sq + \
                        self.lambda_ * phi_new_sq.pow(2)
        kappa_view = self.kappa.view(self.batch_size, 1, 1)
        kinetic_old = -2 * kappa_view * phi_old * sum_phi_neighbors
        kinetic_new = -2 * kappa_view * phi_new * sum_phi_neighbors
        delta_S = (potential_new - potential_old) + (kinetic_new - kinetic_old)

        # Accept or reject the move based on the Metropolis criterion.
        log_u = torch.log(torch.rand_like(delta_S, dtype=torch.float32))
        accept_mask = log_u < (-delta_S)

        # --- Track acceptance rate for each kappa ---
        # Average over chains and sites to get a rate per kappa.
        acceptance_rate_per_kappa = accept_mask.float().mean(dim=(-1, -2))
        self.acceptance_stats += acceptance_rate_per_kappa
        # Increment the counter for each sub-lattice update performed.
        self.acceptance_counter += 1

        # Apply the update where the move was accepted.
        phi_updated = torch.where(accept_mask, phi_new, phi_old)
        self.field[batch_idx, chain_idx, i_sites, j_sites] = phi_updated

    def adjust_proposal_width(self, is_thermalization: bool = False) -> None:
        """Adjusts proposal width based on the measured acceptance rate.

        This method is called periodically during the simulation. It compares
        the average acceptance rate over the last interval to the target rate
        and adjusts the proposal width for each kappa value accordingly.

        Args:
            is_thermalization: A boolean flag used to control whether to print
                tuning status updates.
        """
        # Ensure we don't divide by zero if called before any updates.
        if self.acceptance_counter == 0:
            return

        # Calculate the average acceptance rate over the tuning interval.
        avg_acceptance_rate = self.acceptance_stats / self.acceptance_counter

        # Calculate the adjustment factor. If the rate is too high, the width
        # increases. If the rate is too low, the width decreases.
        error = avg_acceptance_rate - self.target_acceptance_rate
        adjustment_factor = 1.0 + self.tuning_strength * error

        # Apply the adjustment to the proposal widths.
        new_proposal_widths = self.proposal_widths * adjustment_factor

        # Clamp the values to a reasonable range to prevent them from becoming
        # zero, negative, or excessively large.
        self.proposal_widths.copy_(
            torch.clamp(new_proposal_widths, min=1e-3, max=5.0))

        # Optionally, print the tuning status to monitor convergence.
        if is_thermalization:
            print(f"  [Tuning] Avg Acc Rate: "
                  f"{avg_acceptance_rate.mean().item():.3f} -> "
                  f"Avg Proposal Width: "
                  f"{self.proposal_widths.mean().item():.3f}")

        # Reset the trackers for the next tuning interval.
        self.acceptance_stats.zero_()
        self.acceptance_counter.zero_()

    def compute_potential_term(self, field: Tensor = None) -> Tensor:
        """Helper function to compute the potential part of the action."""
        if field is None:
            field = self.field
        field_sq = field.pow(2)
        potential = (1 - 2 * self.lambda_) * field_sq + \
                    self.lambda_ * field_sq.pow(2)
        # Sum over the spatial dimensions (L, L).
        return potential.sum(dim=(-2, -1))

    def compute_kinetic_term(self, field: Tensor = None) -> Tensor:
        r"""Computes the kinetic term K = -2 \sum_{x,\mu} \phi_x \phi_{x+\mu}."""
        if field is None:
            field = self.field
        # To avoid double counting, only sum over forward neighbors (right, down)
        phi_down = torch.roll(field, shifts=1, dims=-2)  # Neighbor in +i
        phi_right = torch.roll(field, shifts=1, dims=-1) # Neighbor in +j
        interaction = field * (phi_down + phi_right)
        # Sum over spatial dimensions and multiply by the standard factor.
        return -2.0 * interaction.sum(dim=(-2, -1))

    def compute_action(self, field: Tensor = None) -> Tensor:
        r"""Computes the total action S = \sum_x [ \kappa*K_x + V_x ]."""
        if field is None:
            kinetic_part = self.compute_kinetic_term()
            potential_part = self.compute_potential_term()
        else:
            kinetic_part = self.compute_kinetic_term(field)
            potential_part = self.compute_potential_term(field)
        # Expand kappa to match the shape of the other terms for broadcasting.
        kappa_exp = self.kappa.view(self.batch_size, 1).expand_as(kinetic_part).to(kinetic_part.device)
        total_action = kappa_exp * kinetic_part + potential_part
        return total_action

    def compute_average_action(self, field: Tensor = None) -> Tensor:
        """Computes the average action per site."""
        if field is None:
            total_action = self.compute_action()
        else:
            total_action = self.compute_action(field)
        # Average over ensemble and sites.
        return total_action.mean(dim=1) / (self.L * self.L)

    def compute_magnetization(self, field: Tensor = None) -> Tensor:
        """Computes the magnetization of the field."""
        if field is None:
            field = self.field
        # Magnetization is the average field value per kappa.
        return field.mean(dim=(-2, -1))

    def compute_average_magnetization(self, field: Tensor = None) -> Tensor:
        """Computes the average magnetization per kappa."""
        if field is None:
            mag = self.compute_magnetization()
        else:
            mag = self.compute_magnetization(field)
        # Average over ensemble and sites.
        return mag.abs().mean(dim=1)  # Use absolute value for magnetization

    def compute_susceptibility(self, field: Tensor = None) -> Tensor:
        """Computes the magnetic susceptibility per kappa."""
        if field is None:
            mag = self.compute_magnetization()
        else:
            mag = self.compute_magnetization(field)
        # Susceptibility is the variance of the magnetization per kappa.
        return (self.L * self.L) * mag.abs().var(dim=1)

    def compute_binder_cumulant(self, field: Tensor = None) -> Tensor:
        """Computes the Binder cumulant for the magnetization."""
        if field is None:
            mag = self.compute_magnetization()
        else:
            mag = self.compute_magnetization(field)
        return 1 - 1/3 * (mag.pow(4).mean(dim=1) /
                          mag.pow(2).mean(dim=1).pow(2))

    def compute_time_slice_correlation(self, field: Tensor) -> Tensor:
        """Computes the connected two-point correlation function of time slices.

        This function calculates G_c(t) = <S(t)S(0)>_c as defined in Appendix B
        of the source document (1705.06231v2). The input field tensor has the shape
        [batch_size, n_samples, L, L].

        This implementation leverages the Wiener-Khinchin theorem by using FFTs
        to efficiently compute the autocorrelation. This is mathematically
        equivalent to averaging the correlation over all possible choices of
        the reference time slice t=0, which significantly improves statistics
        by using the full translational invariance of the periodic lattice.

        Args:
            field: A tensor of field configurations.

        Returns:
            A tensor of shape [batch_size, L] containing the connected
            correlation function G_c(t) for each replica.
        """
        # Ensure the field has the expected 4 dimensions
        if field.ndim != 4:
            raise ValueError(
                f"Input field must have 4 dimensions [B, N_samples, L, L], "
                f"but got {field.ndim}")

        # S(t) is the spatial average of the field at each time t.
        # The time dimension is the third one (dim=-2).
        # S has shape: [batch_size, n_samples, L]
        S = field.mean(dim=-1).abs()

        # 1. Use FFT to compute the autocorrelation.
        # This is equivalent to averaging over all possible t_0 choices.
        # S_fft has shape [batch_size, n_samples, L]
        S_fft = torch.fft.fft(S, dim=-1)

        # 2. Compute the power spectral density, |F(S)|^2, where F is the FFT.
        # psd has shape [batch_size, n_samples, L]
        psd = S_fft.abs().pow(2)

        # 3. The inverse FFT of the PSD gives the unconnected correlation.
        # We take .real as the result must be real-valued.
        # The division by L is for normalization in the convolution theorem.
        # G_unconnected_per_sample has shape [batch_size, n_samples, L]
        G_unconnected_per_sample = torch.fft.ifft(psd, dim=-1).real / self.L

        # 4. Average over the ensemble (n_samples) to get <S(t)S(0)>.
        # G_unconnected has shape [batch_size, L]
        G_unconnected = G_unconnected_per_sample.mean(dim=1)

        # 5. Compute the disconnected part, <S>^2.
        # This requires averaging S over both the time and sample dimensions.
        # avg_S_scalar has shape [batch_size]
        avg_S_scalar = S.mean(dim=(-2, -1))
        disconnected_term = avg_S_scalar.pow(2)

        # 6. Compute the connected correlator: G_c(t) = <S(t)S(0)> - <S>^2.
        # Unsqueeze the disconnected term for broadcasting over the time dim.
        G_c = G_unconnected - disconnected_term.unsqueeze(-1)

        # --- Symmetrization to reduce noise ---
        # Due to periodic boundary conditions, G_c(t) should be equal to
        # G_c(L-t). We average them to reduce statistical errors.
        G_c_flipped = torch.roll(torch.flip(G_c, dims=[-1]), shifts=1, dims=-1)
        # G_c_flipped[t=0] is still G_c[t=0], other t are G_c[L-t].

        # Average G_c(t) and G_c(L-t).
        G_c_sym = 0.5 * (G_c + G_c_flipped)
        # The t=0 point should not be averaged with itself, so restore it.
        G_c_sym[:, 0] = G_c[:, 0]

        # Clean up intermediate tensors to free memory
        del (G_unconnected, G_unconnected_per_sample, S_fft, psd, S,
             avg_S_scalar, disconnected_term, G_c_flipped, G_c)
        gc.collect()

        return G_c_sym

    def compute_renormalized_mass(self, field: Tensor) -> Tensor:
        """Computes the renormalized mass from field configurations.

        This calculation is based on the formulas provided in Appendix B of the
        paper "Cooling Stochastic Quantization with colored noise" (https://arxiv.org/abs/1705.06231)
        (arXiv:1705.06231v2). It uses the time-slice correlation function G_c(t)
        to compute the susceptibility (chi2) and the second moment (mu2).

        The formula for d=2 is: m_R^2 = 4 * chi2 / mu2.

        Args:
            field: A tensor of field configurations with shape
                   [batch_size, n_samples, L, L].

        Returns:
            A tensor of shape [batch_size] containing the renormalized mass
            for each kappa value.
        """
        if field is None:
            raise ValueError(
                "Field samples must be provided to compute renormalized mass.")

        # 1. Compute the time-slice correlation function G_c(t)
        # G_c has shape [batch_size, L]
        G_c = self.compute_time_slice_correlation(field)

        # 2. Compute the connected susceptibility chi2
        # chi2 has shape [batch_size]
        chi2 = self.L * G_c.sum(dim=-1)
        chi2_from_m_fluctuation = self.compute_susceptibility(field)

        # 3. Compute the second moment mu2 from G_c(t)
        t = torch.arange(
            self.L, device=field.device, dtype=field.dtype)
        t_sq = t.pow(2)
        # mu2 has shape [batch_size]
        mu2 = 2 * self.L * (G_c * t_sq).sum(dim=-1)
        # Because <|S(t)|> is not exactly equal to <|M|>, we need to
        # rescale mu2 to account for the fluctuation in the magnetization.
        mu2_fixed = mu2 * (chi2_from_m_fluctuation / chi2)

        # 4. Compute the renormalized mass squared
        # Based on formula (B15): m_R^2 = 4 * chi2 / mu2
        # Add a small epsilon to the denominator for numerical stability
        m_R_sq = 4 * chi2 / mu2_fixed

        # Return the square root. Use relu to prevent issues from numerical
        # noise that might make m_R_sq slightly negative.
        m_R = torch.sqrt(torch.relu(m_R_sq))
        del G_c, chi2, chi2_from_m_fluctuation, mu2, mu2_fixed, t, t_sq, m_R_sq
        gc.collect()
        return m_R

    def compute_rescaled_renormalized_mass(self, field: Tensor = None) -> Tensor:
        """Computes the rescaled renormalized mass (N * m_R).

        This quantity is the renormalized mass m_R multiplied by the linear
        lattice size N (L in this code). This scaling is used in the analysis
        presented in the source document (https://arxiv.org/abs/1705.06231) (e.g., Fig. 12 and Fig. 17).

        Args:
            field: A tensor of field configurations with shape
                   [batch_size, n_samples, L, L]. If None, an error will be
                   raised.

        Returns:
            A tensor of shape [batch_size] containing the rescaled
            renormalized mass for each kappa value.
        """
        if field is None:
            raise ValueError(
                "Field samples must be provided to compute rescaled renormalized mass.")

        # 1. Compute the standard renormalized mass
        m_R = self.compute_renormalized_mass(field)

        # 2. Rescale by multiplying with the lattice size L (N in the paper)
        rescaled_m_R = self.L * m_R

        return rescaled_m_R


# =============================================================================
# Subclass: XYModel
# =============================================================================
class XYModel2D(ThermalLatticeSampler2D):
    r"""Implementation of the 2D XY model (including the Berezinskii-Kosterlitz-Thouless transition).

    This class retains the full functionality and performance optimizations (in-place updates, vectorization)
    of the original BKTXYSampler, with added support for adaptive updates.
    """

    def __init__(self,
                 L: int,
                 T: Tensor,
                 n_chains: int = 30,
                 J: float = 1.0,
                 device: torch.device = torch.device("cpu"),
                 max_delta: float = torch.pi,
                 adaptive: bool = False,
                 target_acceptance: float = 0.6,
                 adapt_rate: float = 0.1,
                 adapt_interval: int = 5,
                 ema_alpha: float = 0.1,
                 use_amp: bool = True,
                 large_size_simulate: bool = False,
                 pt_enabled: bool = True) -> None:
        """
        Args:
            L (int): Lattice size.
            T (Tensor): Temperature tensor.
            n_chains (int, optional): Number of chains. Defaults to 30.
            J (float, optional): Coupling constant. Defaults to 1.0.
            device (torch.device, optional): Compute device. Defaults to CPU.
            max_delta (float, optional): Maximum angular perturbation. Defaults to π.
            adaptive (bool, optional): Flag to enable adaptive max_delta adjustments. Defaults to False.
            target_acceptance (float, optional): Target acceptance rate for adaptive updates. Defaults to 0.6.
            adapt_rate (float, optional): Adaptation rate. Defaults to 0.1.
            adapt_interval (int, optional): Number of sweeps between adaptations. Defaults to 5.
            ema_alpha (float, optional): Exponential moving average factor for the acceptance rate. Defaults to 0.1.
            use_amp (bool, optional): Enable automatic mixed precision. Defaults to True.
            large_size_simulate (bool, optional): Flag for large-scale simulation. Defaults to False.
            pt_enabled (bool, optional): Enable parallel tempering. Defaults to True.
        """
        super().__init__(
            L=L,
            T=T,
            n_chains=n_chains,
            J=J,
            device=device,
            use_amp=use_amp,
            large_size_simulate=large_size_simulate,
            pt_enabled=pt_enabled,
        )
        # XY-model specific parameters for adaptive updates.
        self.adaptive = adaptive
        self.target_acceptance = target_acceptance
        self.adapt_rate = adapt_rate
        self.adapt_interval = adapt_interval
        self.ema_alpha = ema_alpha

        # Register the maximum angular perturbation as a buffer.
        self.max_delta = torch.tensor(max_delta, device=device, dtype=torch.float32)
        # Initialize counters for acceptance statistics.
        self.register_buffer('accept_count', torch.tensor(0, dtype=torch.int64, device=self.device))
        self.register_buffer('total_trials', torch.tensor(0, dtype=torch.int64, device=self.device))
        self.sweep_count = 0
        # Initialize the exponential moving average of the acceptance rate.
        self.register_buffer('ema_accept_rate', torch.tensor(self.target_acceptance, device=self.device, dtype=torch.float32))

    def init_spins(self) -> None:
        r"""Initialize the spin configuration for the XY model.

        Spins are represented as angles ∈ [0, 2π). The tensor shape is [batch_size, n_chains, L, L].
        """
        theta_init = 2 * torch.pi * torch.rand(
            (self.batch_size, self.n_chains, self.L, self.L),
            dtype=torch.float32,
            device=self.device
        )
        self.spins = theta_init

    def one_sweep(self, adaptive: bool = False) -> None:
        r"""Perform one sweep and adjust max_delta if adaptive updates are enabled.

        Overrides the base class one_sweep to include adaptive adjustments.

        Args:
            adaptive (bool, optional): Whether to use adaptive update parameters. Defaults to False.
        """
        super().one_sweep(adaptive=self.adaptive)
        if self.adaptive:
            self.sweep_count += 1
            if self.sweep_count % self.adapt_interval == 0:
                self.adjust_max_delta()

    @torch.compile
    def metropolis_update_sub_lattice(self, lattice_color: str, adaptive: bool = False) -> None:
        r"""Perform a Metropolis update on the specified sub-lattice for the XY model.

        For each site in the sub-lattice, an angular perturbation is applied, the energy difference is computed,
        and the new angle is accepted or rejected according to the Metropolis criterion.

        Args:
            lattice_color (str): 'black' or 'white' indicating which sub-lattice to update.
            adaptive (bool, optional): Flag for using adaptive parameters. Defaults to False.
        """
        T = self.T

        # Select indices based on the chosen sub-lattice.
        if lattice_color == 'black':
            batch_idx = self.black_batch_idx
            chain_idx = self.black_chain_idx
            i_sites = self.black_i_sites
            j_sites = self.black_j_sites
            neighbors_i = self.black_neighbors_i
            neighbors_j = self.black_neighbors_j
            N = self.num_black
        else:
            batch_idx = self.white_batch_idx
            chain_idx = self.white_chain_idx
            i_sites = self.white_i_sites
            j_sites = self.white_j_sites
            neighbors_i = self.white_neighbors_i
            neighbors_j = self.white_neighbors_j
            N = self.num_white

        theta_old = self.spins[batch_idx, chain_idx, i_sites, j_sites]

        # Generate random angular perturbations in the interval [-max_delta/2, max_delta/2].
        dtheta = torch.rand(self.batch_size, self.n_chains, N, device=self.device)
        dtheta.mul_(self.max_delta).add_(-self.max_delta / 2)
        theta_new = torch.remainder(theta_old + dtheta, 2 * torch.pi)

        # Retrieve neighbor angles.
        theta_neighbors = self.spins[
            batch_idx.unsqueeze(-1),
            chain_idx.unsqueeze(-1),
            neighbors_i,
            neighbors_j
        ]  # Shape: [B, C, N, 4]

        # Compute the energy difference ΔE for the perturbation.
        diff_new = theta_new.unsqueeze(-1) - theta_neighbors
        diff_old = theta_old.unsqueeze(-1) - theta_neighbors
        delta_cos = torch.cos(diff_new) - torch.cos(diff_old)
        delta_E = -self.J * delta_cos.sum(dim=-1)

        # Expand temperature tensor and compute acceptance probability.
        T_exp = T.view(self.batch_size, 1).expand(self.batch_size, self.n_chains)
        T_sites = T_exp[batch_idx, chain_idx]  # Shape: [B, C, N]
        p_acc = torch.exp(-delta_E / T_sites)

        rand_vals = torch.rand(self.batch_size, self.n_chains, N, device=self.device)
        accept_mask = (rand_vals < p_acc)

        if self.adaptive:
            self.accept_count.add_(accept_mask.sum())
            self.total_trials.add_(accept_mask.numel())

        # Update spins based on the acceptance decision.
        theta_updated = torch.where(accept_mask, theta_new, theta_old)
        self.spins[batch_idx, chain_idx, i_sites, j_sites] = theta_updated

        # Clean up intermediate variables.
        del dtheta, diff_new, diff_old, theta_updated, delta_cos, T_exp, delta_E, rand_vals, p_acc, theta_new, theta_old, accept_mask

    def adjust_max_delta(self) -> None:
        r"""Adaptively adjust the maximum angular perturbation (max_delta) based on the acceptance rate.

        The method uses an exponential moving average of the acceptance rate and modifies max_delta accordingly.
        """
        if self.total_trials == self.zero:
            return

        current_rate = self.accept_count.float() / self.total_trials.float()
        self.ema_accept_rate.mul_(1 - self.ema_alpha).add_(self.ema_alpha * current_rate)

        factor = torch.exp(self.adapt_rate * (self.ema_accept_rate - self.target_acceptance))
        new_max_delta = torch.clamp(self.max_delta * factor, min=1e-4, max=2 * torch.pi)
        self.max_delta.copy_(new_max_delta)

        self.accept_count.zero_()
        self.total_trials.zero_()
        del factor, new_max_delta, current_rate

    def compute_energy(self) -> Tensor:
        r"""Compute the total energy for the XY model.

        The energy is given by:
            E = -J * Σ cos(θ(i) - θ(neighbor))
        Only the right and down neighbors are considered to avoid double counting.

        Returns:
            Tensor: Energy tensor of shape [batch_size, n_chains].
        """
        theta = self.spins.to(self.device)
        t_top = torch.roll(theta, shifts=1, dims=3)
        t_right = torch.roll(theta, shifts=-1, dims=2)
        E_local = torch.cos(theta - t_top) + torch.cos(theta - t_right)
        E_batch = -self.J * E_local.sum(dim=(2, 3))

        del theta, t_top, t_right, E_local
        return E_batch

    def compute_average_energy(self) -> Tensor:
        r"""Compute the average energy per site for the XY model.

        Returns:
            Tensor: Average energy tensor.
        """
        return self.compute_energy().mean(dim=1) / self.L**2

    def compute_specific_heat_capacity(self) -> Tensor:
        r"""Compute the heat capacity per site for the XY model.
        The heat capacity is given by:
            C = (1 / T^2) * var(E)
        where var(E) is the variance of the energy.
        The factor of 1 / L^2 is included to normalize the heat capacity per site.

        Returns:
            Tensor: Heat capacity tensor.
        """
        T = self.T.to(self.device)
        c = torch.var(self.compute_energy(), dim=1) / T**2
        return c / self.L**2

    def compute_spin_stiffness(self) -> Tensor:
        r"""Compute the spin stiffness for the XY model.
        The spin stiffness is given by:
            ρ_s = (J * <cos(θ(i) - θ(i+1))> - (J^2 / T) * <sin(θ(i) - θ(i+1))^2>) / L^2
        where <...> denotes the average over all sites.
        The factor of 1 / L^2 is included to normalize the stiffness per site.

        Reference: https://arxiv.org/pdf/1101.3281#page=50

        Returns:
            Tensor: Spin stiffness tensor.
        """
        theta = self.spins.to(self.device)
        diff_y = torch.roll(theta, shifts=1, dims=3) - theta
        avg_links_y = torch.cos(diff_y).sum(dim=(2, 3)).mean(dim=1)
        avg_currents2_y = (torch.sin(diff_y).sum(dim=(2, 3)) ** 2).mean(dim=1)
        del diff_y, theta
        return (self.J * avg_links_y - (self.J**2 / self.T) * avg_currents2_y) / self.L**2

    def compute_magnetization(self) -> Tensor:
        r"""Compute the magnetization per site for the XY model.
        The magnetization is given by:
            m = (1 / L^2) * Σ (m_x^2 + m_y^2)^(1/2)
        where the sum is over all lattice sites.

        Reference: https://iopscience.iop.org/article/10.1088/0953-8984/4/24/011

        Returns:
            Tensor: Magnetization tensor.
        """
        theta = self.spins.to(self.device)
        mx = torch.cos(theta).sum(dim=(2, 3)).unsqueeze(-1)
        my = torch.sin(theta).sum(dim=(2, 3)).unsqueeze(-1)
        m = torch.stack([mx, my], dim=2).norm(dim=2).squeeze(dim=2)
        del theta, mx, my
        return m.mean(dim=1) / self.L**2

    def compute_susceptibility(self) -> Tensor:
        r"""Compute the susceptibility per site for the XY model.
        The susceptibility is given by:
            χ = (1 / T) * var(m)
        where m is the magnetization per site.

        Returns:
            Tensor: Susceptibility tensor.
        """
        theta = self.spins.to(self.device)
        mx = torch.cos(theta).sum(dim=(2, 3)).unsqueeze(-1)
        my = torch.sin(theta).sum(dim=(2, 3)).unsqueeze(-1)
        m = torch.stack([mx, my], dim=2).norm(dim=2).squeeze(dim=2)
        del theta, mx, my
        return m.var(dim=1) * (1.0 / self.T) / self.L**2

    def _principal_value(self, delta: Tensor) -> Tensor:
        """Map angle to range [−π,π].

        Arg:
            delta: Tensor of shape [num_temp, num_samples, H, W].

        Returns:
            Tensor of shape [num_temp, num_samples, H, W]. Elements ∈ [−π,π]
        """
        return (delta + torch.pi) % (2*torch.pi) - torch.pi

    def compute_vortex_density(self) -> Tensor:
        """Compute the vortex density for the XY model.
        The vortex density is computed using the formula:
            ρ_v = (1 / L^2) * Σ |ω(i,j)| / (2π)
        where ω(i,j) is the vorticity tensor at site (i,j).
        The sum is over all lattice sites, and the factor of 1 / L^2 is included to normalize the density per site.

        Reference: https://arxiv.org/pdf/2207.13748#page=20

        Returns:
            Tensor: Vortex density tensor.
        """
        theta = self.spins.to(self.device)
        theta_ip = torch.roll(theta, shifts=-1, dims=-2)      # i+1, j
        theta_jp = torch.roll(theta, shifts=-1, dims=-1)      # i, j+1
        theta_ipp_jp = torch.roll(theta_ip, shifts=-1, dims=-1)  # i+1, j+1

        d1 = self._principal_value(delta=theta_jp     - theta)         # (i,j)->(i,j+1)
        d2 = self._principal_value(delta=theta_ipp_jp - theta_jp)      # (i,j+1)->(i+1,j+1)
        d3 = self._principal_value(delta=theta_ip     - theta_ipp_jp)  # (i+1,j+1)->(i+1,j)
        d4 = self._principal_value(delta=theta        - theta_ip)      # (i+1,j)->(i,j)

        omega = d1 + d2 + d3 + d4  # ∈ [−4π,4π]
        q = torch.round(omega / (2*torch.pi)).to(torch.int32)
        del theta, theta_ip, theta_jp, theta_ipp_jp, d1, d2, d3, d4, omega
        gc.collect()
        return q.to(torch.float).abs().mean(dim=(1, 2, 3))


# =============================================================================
# Subclass: IsingModel
# =============================================================================
class IsingModel2D(ThermalLatticeSampler2D):
    r"""Implementation of the 2D Ising model using a spin-flip Metropolis update.

    The update is performed using a checkerboard sub-lattice approach.
    """

    def __init__(self,
                 L: int,
                 T: Tensor,
                 n_chains: int = 30,
                 J: float = 1.0,
                 device: torch.device = torch.device("cpu"),
                 use_amp: bool = True,
                 large_size_simulate: bool = False,
                 pt_enabled: bool = False) -> None:
        """
        Args:
            L (int): Lattice size.
            T (Tensor): Temperature tensor.
            n_chains (int, optional): Number of chains. Defaults to 30.
            J (float, optional): Coupling constant. Defaults to 1.0.
            device (torch.device, optional): Compute device. Defaults to CPU.
            use_amp (bool, optional): Enable automatic mixed precision. Defaults to True.
            large_size_simulate (bool, optional): Flag for large-scale simulation. Defaults to False.
            pt_enabled (bool, optional): Enable parallel tempering. Defaults to False.
        """
        super().__init__(
            L=L,
            T=T,
            n_chains=n_chains,
            J=J,
            device=device,
            use_amp=use_amp,
            large_size_simulate=large_size_simulate,
            pt_enabled=pt_enabled,
        )

    def init_spins(self) -> None:
        r"""Initialize spins for the Ising model.

        Spins take values ±1. The resulting tensor has shape [batch_size, n_chains, L, L].
        """
        spins_init = torch.randint(0, 2, (self.batch_size, self.n_chains, self.L, self.L),
                                    device=self.device, dtype=torch.float32)
        # Map {0,1} to {-1,+1}
        spins_init = 2 * spins_init - 1
        self.spins = spins_init

    @torch.compile
    def metropolis_update_sub_lattice(self, lattice_color: str, adaptive: bool = False) -> None:
        r"""Perform the Metropolis update on the specified sub-lattice for the Ising model.

        The update consists of attempting to flip the spin at each site and accepting the flip
        based on the computed energy difference.

        Args:
            lattice_color (str): 'black' or 'white' indicating which sub-lattice to update.
            adaptive (bool, optional): Not used for the Ising update. Defaults to False.
        """
        T = self.T

        if lattice_color == 'black':
            batch_idx = self.black_batch_idx
            chain_idx = self.black_chain_idx
            i_sites = self.black_i_sites
            j_sites = self.black_j_sites
            neighbors_i = self.black_neighbors_i
            neighbors_j = self.black_neighbors_j
            N = self.num_black
        else:
            batch_idx = self.white_batch_idx
            chain_idx = self.white_chain_idx
            i_sites = self.white_i_sites
            j_sites = self.white_j_sites
            neighbors_i = self.white_neighbors_i
            neighbors_j = self.white_neighbors_j
            N = self.num_white

        s_old = self.spins[batch_idx, chain_idx, i_sites, j_sites]
        s_new = -s_old

        # Retrieve neighboring spins.
        s_neighbors = self.spins[
            batch_idx.unsqueeze(-1),
            chain_idx.unsqueeze(-1),
            neighbors_i,
            neighbors_j
        ]  # Shape: [B, C, N, 4]

        # Compute energy difference ΔE = -J * (s_new * Σ(neighbors) - s_old * Σ(neighbors))
        sum_neighbors = s_neighbors.sum(dim=-1)
        delta_E = -self.J * (s_new * sum_neighbors - s_old * sum_neighbors)

        T_exp = T.view(self.batch_size, 1).expand(self.batch_size, self.n_chains)
        T_sites = T_exp[batch_idx, chain_idx]  # Shape: [B, C, N]
        p_acc = torch.exp(-delta_E / T_sites)

        rand_vals = torch.rand(self.batch_size, self.n_chains, N, device=self.device)
        accept_mask = (rand_vals < p_acc)

        s_updated = torch.where(accept_mask, s_new, s_old)
        self.spins[batch_idx, chain_idx, i_sites, j_sites] = s_updated

        del s_new, s_old, s_neighbors, sum_neighbors, T_exp, delta_E, rand_vals, p_acc, accept_mask, s_updated

    def compute_energy(self) -> Tensor:
        r"""Compute the energy of the Ising model.

        The energy is computed as:
            E = -J * Σ( s(i) * s(up) + s(i) * s(right) )
        where only the top and right neighbors are considered to avoid double counting.

        Returns:
            Tensor: Energy tensor with shape [batch_size, n_chains].
        """
        s = self.spins.to(self.device)
        s_up = torch.roll(s, shifts=1, dims=2)
        s_right = torch.roll(s, shifts=-1, dims=3)
        E_local = s * s_up + s * s_right
        E_batch = -self.J * E_local.sum(dim=(2, 3))

        del s, s_up, s_right, E_local
        return E_batch

    def compute_average_energy(self) -> Tensor:
        r"""Compute the average energy per site for the Ising model.

        Returns:
            Tensor: Average energy tensor.
        """
        return self.compute_energy().mean(dim=1) / self.L**2

    def compute_specific_heat_capacity(self) -> Tensor:
        r"""Compute the heat capacity per site for the Ising model.
        The heat capacity is given by:
            C = (1 / T^2) * var(E)
        where var(E) is the variance of the energy.
        The factor of 1 / L^2 is included to normalize the heat capacity per site.

        Returns:
            Tensor: Heat capacity tensor.
        """
        c = torch.var(self.compute_energy(), dim=1) / self.T.to(self.device)**2
        return c / self.L**2

    def compute_magnetization(self) -> Tensor:
        r"""Compute the magnetization per site for the Ising model.
        The magnetization is given by:
            m = (1 / L^2) * Σ s(i)
        where the sum is over all lattice sites.

        Returns:
            Tensor: Magnetization tensor.
        """
        return self.spins.to(self.device).mean(dim=(2, 3)).abs().mean(dim=1)

    def compute_susceptibility(self) -> Tensor:
        r"""Compute the susceptibility per site for the Ising model.
        The susceptibility is given by:
            χ = (1 / T) * var(m)
        where m is the magnetization per site.

        Returns:
            Tensor: Susceptibility tensor.
        """
        return self.spins.to(self.device).mean(dim=(2, 3)).abs().var(dim=1) / self.T.to(self.device)

    def compute_binder_cumulant(self) -> Tensor:
        r"""Compute the Binder cumulant for the Ising model.

        The Binder cumulant is given by:
            U_4 = 1 - (1 / 3) * (《m^4》 / 《m^2》^2)
        where m^2 is the square of the magnetization and m^4 is the fourth moment of the magnetization.

        Returns:
            Tensor: Binder cumulant tensor.
        """
        m2 = self.spins.to(self.device).mean(dim=(2, 3)).pow(2).mean(dim=1)
        m4 = self.spins.to(self.device).mean(dim=(2, 3)).pow(4).mean(dim=1)
        return 1.0 - m4.div(3.0 * m2 * m2)

    def compute_domain_wall_density(self) -> Tensor:
        r"""Compute the domain wall density for the Ising model.

        The domain wall density is given by:
            ρ_dw = ⟨(1−s_i s_j)/2⟩
        where the average is taken over all pairs of neighboring spins.

        Returns:
            Tensor: Domain wall density tensor.
        """
        spins = self.spins.to(self.device)
        shift_x = spins.roll(shifts=-1, dims=3)
        shift_y = spins.roll(shifts=1, dims=2)
        dw_x = (1 - spins * shift_x) / 2
        dw_y = (1 - spins * shift_y) / 2
        dw_density = (dw_x + dw_y).mean(dim=(2, 3))
        return dw_density.mean(dim=1)

    def compute_exact_magnetization(self) -> Tensor:
        r"""Compute the exact spontaneous magnetization for the Ising model.

        The spontaneous magnetization is given by:
            m = (1 - sinh(2 * J / T)^(-4))^(1/8)
        This formula is valid for T < 2J / log(1 + sqrt(2)).
        The function returns 0 for T >= 2J / log(1 + sqrt(2)).

        Reference: https://journals.aps.org/pr/abstract/10.1103/PhysRev.85.808

        Returns:
            Tensor: Exact spontaneous magnetization tensor.
        """
        k = self.J / self.T
        kc = 0.5 * torch.log(1 + torch.sqrt(2 * torch.ones_like(self.T)))
        m = (1 - (1/torch.sinh(2 * k)).pow(4)).pow(0.125)
        return m.masked_fill_(k<=kc, 0.0)


# =============================================================================
# Subclass: PottsModel
# =============================================================================
class PottsModel2D(ThermalLatticeSampler2D):
    r"""Implementation of the 2D q-state Potts model.

    In the Potts model, spins are integer states in {0, 1, ..., q-1} and the energy is given by:
      E = -J * Σ δ(s(i), s(j)),
    where δ is the Kronecker delta (only contributing when neighboring spins are equal).
    """

    def __init__(self,
                 L: int,
                 T: Tensor,
                 q: int = 3,
                 n_chains: int = 1,
                 J: float = 1.0,
                 device: torch.device = torch.device("cpu"),
                 use_amp: bool = False,
                 large_size_simulate: bool = False,
                 pt_enabled: bool = False) -> None:
        """
        Args:
            L (int): Lattice size.
            T (Tensor): Temperature tensor.
            q (int, optional): Number of states in the Potts model. Defaults to 3.
            n_chains (int, optional): Number of chains. Defaults to 1.
            J (float, optional): Coupling constant. Defaults to 1.0.
            device (torch.device, optional): Compute device. Defaults to CPU.
            use_amp (bool, optional): Enable automatic mixed precision. Defaults to False.
            large_size_simulate (bool, optional): Flag for large-scale simulation. Defaults to False.
            pt_enabled (bool, optional): Enable parallel tempering. Defaults to False.
        """
        if q < 2:
            raise ValueError("q must be ≥2.")
        self.q = q

        super().__init__(
            L=L,
            T=T,
            n_chains=n_chains,
            J=J,
            device=device,
            use_amp=use_amp,
            large_size_simulate=large_size_simulate,
            pt_enabled=pt_enabled,
        )

    def init_spins(self) -> None:
        r"""Initialize the spin configuration for the Potts model.

        Spins take integer values in the range [0, q-1]. The tensor shape is [batch_size, n_chains, L, L].
        """
        spins_init = torch.randint(
            0, self.q,
            (self.batch_size, self.n_chains, self.L, self.L),
            device=self.device,
            dtype=torch.int64
        )
        self.spins = spins_init

    @torch.compile
    def metropolis_update_sub_lattice(self, lattice_color: str, adaptive: bool = False) -> None:
        r"""Perform the Metropolis update on the specified sub-lattice for the Potts model.

        For each site in the sub-lattice, a new state (different from the current state) is proposed.
        The energy difference is computed using the Kronecker delta and the move is accepted/rejected based on
        the Metropolis criterion.

        Args:
            lattice_color (str): 'black' or 'white' indicating which sub-lattice to update.
            adaptive (bool, optional): Not used for the Potts update. Defaults to False.
        """
        T = self.T

        if lattice_color == 'black':
            batch_idx = self.black_batch_idx
            chain_idx = self.black_chain_idx
            i_sites = self.black_i_sites
            j_sites = self.black_j_sites
            neighbors_i = self.black_neighbors_i
            neighbors_j = self.black_neighbors_j
            N = self.num_black
        else:
            batch_idx = self.white_batch_idx
            chain_idx = self.white_chain_idx
            i_sites = self.white_i_sites
            j_sites = self.white_j_sites
            neighbors_i = self.white_neighbors_i
            neighbors_j = self.white_neighbors_j
            N = self.num_white

        s_old = self.spins[batch_idx, chain_idx, i_sites, j_sites]

        # Propose a new state different from the current state.
        s_rand = torch.randint(
            0, self.q,
            (self.batch_size, self.n_chains, N),
            device=self.device,
            dtype=torch.int64
        )
        s_new = torch.where(s_rand == s_old, (s_rand + 1) % self.q, s_rand)

        # Retrieve neighbor spins.
        s_neighbors = self.spins[
            batch_idx.unsqueeze(-1),
            chain_idx.unsqueeze(-1),
            neighbors_i,
            neighbors_j
        ]  # Shape: [B, C, N, 4]

        # Compute the energy difference:
        # E = -J * Σ δ(s, neighbor)  -->  ΔE = -J * [Σ δ(s_new, neighbor) - Σ δ(s_old, neighbor)]
        matches_old = (s_neighbors == s_old.unsqueeze(-1)).sum(dim=-1)
        matches_new = (s_neighbors == s_new.unsqueeze(-1)).sum(dim=-1)
        delta_E = -self.J * (matches_new - matches_old).float()

        T_exp = T.view(self.batch_size, 1).expand(self.batch_size, self.n_chains)
        T_sites = T_exp[batch_idx, chain_idx]  # Shape: [B, C, N]
        p_acc = torch.exp(-delta_E / T_sites)

        rand_vals = torch.rand(self.batch_size, self.n_chains, N, device=self.device)
        accept_mask = (rand_vals < p_acc)

        s_updated = torch.where(accept_mask, s_new, s_old)
        self.spins[batch_idx, chain_idx, i_sites, j_sites] = s_updated

        del s_rand, s_new, s_old, s_neighbors, matches_old, matches_new, delta_E, T_exp, rand_vals, p_acc, accept_mask, s_updated

    def compute_energy(self) -> Tensor:
        r"""Compute the energy of the Potts model.

        The energy is given by:
            E = -J * Σ( δ(s(i), s(up)) + δ(s(i), s(right)) )
        Only the top and right neighbors are considered to avoid double counting.

        Returns:
            Tensor: Energy tensor with shape [batch_size, n_chains].
        """
        s = self.spins.to(self.device)
        s_up = torch.roll(s, shifts=1, dims=2)
        s_right = torch.roll(s, shifts=-1, dims=3)

        match_up = (s == s_up).float()
        match_right = (s == s_right).float()

        E_local = match_up + match_right
        E_batch = -self.J * E_local.sum(dim=(2, 3))

        del s, s_up, s_right, match_up, match_right, E_local
        return E_batch

    def compute_average_energy(self) -> Tensor:
        r"""Compute the average energy per site for the Potts model.

        Returns:
            Tensor: Average energy tensor.
        """
        return self.compute_energy().mean(dim=1) / self.L**2

    def compute_specific_heat_capacity(self) -> Tensor:
        r"""Compute the heat capacity per site for the Potts model.
        The heat capacity is given by:
            C = (1 / T^2) * var(E)
        where var(E) is the variance of the energy.
        The factor of 1 / L^2 is included to normalize the heat capacity per site.

        Returns:
            Tensor: Heat capacity tensor with shape [batch_size].
        """
        T = self.T.to(self.device)
        energy = self.compute_energy()  # Shape: [batch_size, n_chains]
        c = torch.var(energy, dim=1) / (T**2)
        return c / self.L**2

    def compute_magnetization(self) -> Tensor:
        r"""Compute the magnetization per site for the Potts model.
        The magnetization is given by:
            n_α = (1 / L^2) * Σ delta(s(i), s(j))
            Σ n_α = 1
            m = (q * n_α_max - 1) / (q - 1)
        where the sum is over all lattice sites and n_α_max is the maximum occupancy fraction.

        Returns:
            Tensor: Magnetization tensor with shape [batch_size].
        """
        spins = self.spins.to(self.device)
        # One-hot encode → shape (*batch, *dims, q)
        one_hot = F.one_hot(spins, num_classes=self.q).to(torch.float32)
        # Occupancy fractions along spatial dims
        n_alpha = one_hot.mean(dim=(2, 3))  # (*batch, q)
        # Max-colour definition
        m_scalar = (self.q * n_alpha.max(dim=-1).values - 1) / (self.q - 1)
        return m_scalar.mean(dim=1)

    def compute_susceptibility(self) -> Tensor:
        r"""Compute the susceptibility per site for the Potts model.
        The susceptibility is given by:
            χ = (1 / T) * var(m)
        where m is the magnetization per site.

        Returns:
            Tensor: Susceptibility tensor.
        """
        spins = self.spins.to(self.device)
        n_alpha = F.one_hot(spins, num_classes=self.q).to(torch.float32).mean(dim=(2, 3))
        m_scalar = (self.q * n_alpha.max(dim=-1).values - 1) / (self.q - 1)
        return m_scalar.var(dim=1) / self.T.to(self.device)

    def compute_binder_cumulant(self) -> Tensor:
        r"""Compute the Binder cumulant for the Potts model.

        The Binder cumulant is given by:
            U_4 = 1 - (1 / 3) * (《m^4》 / 《m^2》^2)
        where m^2 is the square of the magnetization and m^4 is the fourth moment of the magnetization.

        Returns:
            Tensor: Binder cumulant tensor.
        """
        spins = self.spins.to(self.device)
        n_alpha = F.one_hot(spins, num_classes=self.q).to(torch.float32).mean(dim=(2, 3))
        m_scalar = (self.q * n_alpha.max(dim=-1).values - 1) / (self.q - 1)
        m2 = m_scalar.pow(2).mean(dim=1)
        m4 = m_scalar.pow(4).mean(dim=1)
        return 1.0 - m4.div(3.0 * m2 * m2)

    def compute_entropy(self) -> Tensor:
        r"""Compute the entropy per site for the Potts model.

        The entropy is given by:
            S = -Σ p_i * log(p_i)
        where p_i is the probability of each state.

        Returns:
            Tensor: Entropy tensor with shape [batch_size].
        """
        spins = self.spins.to(self.device)
        n_alpha = F.one_hot(spins, num_classes=self.q).to(torch.float32).mean(dim=(2, 3))
        entropy = -torch.sum(n_alpha * torch.log(n_alpha + 1e-10), dim=-1)
        return entropy.mean(dim=1)
