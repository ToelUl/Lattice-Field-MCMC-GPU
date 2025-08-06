import gc
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Union

from .bases import ScalarLatticeSampler2D

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
        """Computes the total action S = \sum_x [ \kappa*K_x + V_x ]."""
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

        This helper function calculates G_c(t) = <S(t)S(0)>_c as defined in
        Appendix B of the source document (https://arxiv.org/abs/1705.06231).
        The input field tensor is expected to have the shape
         [batch_size, n_samples, L, L].

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

        # S(t) is the spatial average of the field at each time t
        # The time dimension is the third one (dim=-2)
        # S has shape [batch_size, n_samples, L]
        S = field.mean(dim=-1)

        # Get the time-slice at t=0 for correlation calculation
        # S0 has shape [batch_size, n_samples, 1]
        S0 = S[..., 0].unsqueeze(-1)

        # Ensemble average of <S(t)S(0)>
        # Shape: [batch_size, L]
        G_unconnected = (S * S0).mean(dim=1)

        # Ensemble average of <S(t)> and <S(0)>
        # avg_S has shape [batch_size, L]
        avg_S = S.mean(dim=1)
        # avg_S0 has shape [batch_size, 1]
        avg_S0 = S0.mean(dim=1)

        # The connected correlation function G_c(t) = <S(t)S(0)> - <S(t)><S(0)>
        G_c = G_unconnected - (avg_S * avg_S0)

        # --- Symmetrization to reduce noise ---
        # Due to periodic boundary conditions, G_c(t) should be equal to G_c(L-t).
        # We use this property to average the correlator, which significantly
        # reduces statistical errors.
        # Get G_c(L-t)
        G_c_flipped = torch.roll(torch.flip(G_c, dims=[-1]), shifts=1, dims=-1)
        # G_c_flipped[t=0] is still G_c[t=0], other t are G_c[L-t]

        # Average G_c(t) and G_c(L-t)
        G_c_sym = 0.5 * (G_c + G_c_flipped)
        # The t=0 point should not be averaged, so we restore it.
        G_c_sym[:, 0] = G_c[:, 0]
        del G_unconnected, avg_S, avg_S0, G_c_flipped, G_c, S, S0
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

        # 3. Compute the second moment mu2 from G_c(t)
        t = torch.arange(
            self.L, device=field.device, dtype=field.dtype)
        t_sq = t.pow(2)
        # mu2 has shape [batch_size]
        mu2 = 2 * self.L * (G_c * t_sq).sum(dim=-1)

        # 4. Compute the renormalized mass squared
        # Based on formula (B15): m_R^2 = 4 * chi2 / mu2
        # Add a small epsilon to the denominator for numerical stability
        m_R_sq = 4 * chi2 / mu2

        # Return the square root. Use relu to prevent issues from numerical
        # noise that might make m_R_sq slightly negative.
        m_R = torch.sqrt(torch.relu(m_R_sq))
        del G_c, chi2, mu2, t, t_sq, m_R_sq
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
