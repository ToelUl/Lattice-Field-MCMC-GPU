import gc
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, Union
from abc import ABC, abstractmethod

# =============================================================================
# Abstract Base Class: ScalarLatticeSampler2D
# =============================================================================
class ScalarLatticeSampler2D(nn.Module, ABC):
    """An abstract base class for 2D scalar lattice field theory Monte Carlo samplers.

    This class provides a generic framework for performing Monte Carlo
    simulations of 2D scalar lattice models using PyTorch. It handles the core
    mechanics such as device placement, checkerboard updates, parallel
    tempering, and sample collection. Subclasses must implement model-specific
    details.

    The simulation workflow, managed by the `forward` method, consists of a
    thermalization phase to bring the system to equilibrium, followed by a
    production phase where configurations are sampled periodically.

    Attributes:
        L (int): The linear size of the square lattice (L x L).
        kappa (Tensor): A tensor of coupling constants, one for each replica.
        n_chains (int): The number of independent Monte Carlo chains per kappa.
        device (torch.device): The PyTorch device (CPU or GPU) for computation.
        field (Tensor): The current field configurations for all chains.
            Shape: [batch_size, n_chains, L, L].
        batch_size (int): The number of parallel replicas, derived from kappa.
        use_amp (bool): Flag to enable Automatic Mixed Precision for performance.
        large_size_simulate (bool): Flag to move samples to CPU to save VRAM.
        pt_enabled (bool): Flag to enable parallel tempering exchanges.
    """

    def __init__(self,
                 L: int,
                 kappa: Tensor,
                 n_chains: int = 30,
                 device: Optional[torch.device] = None,
                 use_amp: bool = False,
                 large_size_simulate: bool = False,
                 pt_enabled: bool = True) -> None:
        """Initializes the ScalarLatticeSampler2D.

        Args:
            L: The linear size of the lattice.
            kappa: A 1D tensor of coupling constants. Its length determines the
                number of parallel simulation replicas (batch_size).
            n_chains: The number of independent Monte Carlo chains to run for
                each kappa value.
            device: The PyTorch device to use. If None, defaults to CUDA if
                available, otherwise CPU.
            use_amp: If True, enables Automatic Mixed Precision (AMP) for
                potential speed-up on compatible GPUs.
            large_size_simulate: If True, collected samples are immediately
                moved to CPU memory to conserve GPU memory.
            pt_enabled: If True, enables parallel tempering exchanges between
                replicas with adjacent kappa values.
        """
        super().__init__()
        self.L = L
        if kappa.ndim != 1:
            raise ValueError(
                f"Kappa tensor must be 1D, but got shape {kappa.shape}")
        self.batch_size = kappa.shape[0]

        # Determine computation device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.kappa = kappa.to(self.device)
        self.n_chains = n_chains
        self.use_amp = use_amp
        self.large_size_simulate = large_size_simulate
        self.pt_enabled = pt_enabled

        # Validate simulation settings
        if self.large_size_simulate and self.device.type == 'cpu':
             print("Warning: large_size_simulate=True has no effect on CPU.")
             self.large_size_simulate = False
        if self.use_amp and self.device.type not in ['cuda', 'mps']:
            print(f"Warning: use_amp=True is only effective on CUDA/MPS, "
                  f"but device is {self.device.type}.")

        # The field must be initialized by the subclass.
        self.init_field()
        if not hasattr(self, 'field') or self.field.shape != (
            self.batch_size, self.n_chains, self.L, self.L):
             raise NotImplementedError(
                 "Subclass must implement init_field() and define self.field "
                 f"with shape ({self.batch_size}, {self.n_chains}, "
                 f"{self.L}, {self.L})")

        # Pre-compute indices for efficient updates.
        self._prepare_checkerboard_indices()
        print(f"Initialized ScalarLatticeSampler2D on device: {self.device}")
        print(f" L={L}, BatchSize={self.batch_size}, "
              f"ChainsPerKappa={n_chains}, AMP={use_amp}, "
              f"PT={pt_enabled}, LargeSim={large_size_simulate}")

    @abstractmethod
    def init_field(self) -> None:
        """Initializes the lattice field tensor `self.field`.

        This method must be implemented by subclasses. It should create the
        `self.field` tensor with the correct shape [B, C, L, L] and initial
        state appropriate for the specific model being simulated.
        """
        pass

    @abstractmethod
    def metropolis_update_sub_lattice(self, lattice_color: str) -> None:
        """Performs a Metropolis-Hastings update on a single sub-lattice.

        This method must be implemented by subclasses. It defines the core
        update step for the specific model and should operate in-place on
        the `self.field` tensor.

        Args:
            lattice_color: Either 'black' or 'white', indicating which
                sub-lattice to update.
        """
        pass

    @abstractmethod
    def compute_action(self) -> Tensor:
        """Computes the total action of the current field configuration.

        This method must be implemented by subclasses.

        Returns:
            A tensor of shape [batch_size, n_chains] containing the total
            action for each chain.
        """
        pass

    @abstractmethod
    def compute_kinetic_term(self) -> Tensor:
        """Computes the kinetic term of the action.

        This method must be implemented by subclasses. It is required for
        efficient parallel tempering exchanges. It should calculate the part
        of the action that is proportional to kappa.

        Returns:
            A tensor of shape [batch_size, n_chains] containing the kinetic
            term for each chain.
        """
        pass

    def _prepare_checkerboard_indices(self) -> None:
        """Pre-computes indices for efficient checkerboard updates."""
        # Create a grid of coordinates.
        i_coords = torch.arange(self.L, device=self.device)
        j_coords = torch.arange(self.L, device=self.device)
        grid_i, grid_j = torch.meshgrid(i_coords, j_coords, indexing='ij')

        # Create masks for black and white sites.
        black_mask = (grid_i + grid_j) % 2 == 0
        white_mask = ~black_mask
        black_indices_2d = torch.nonzero(black_mask, as_tuple=False)
        white_indices_2d = torch.nonzero(white_mask, as_tuple=False)
        self.num_black = black_indices_2d.shape[0]
        self.num_white = white_indices_2d.shape[0]

        # Pre-compute broadcastable indices for batch and chain dimensions.
        B, C = self.batch_size, self.n_chains
        batch_idx = torch.arange(B, device=self.device).view(B, 1, 1)
        chain_idx = torch.arange(C, device=self.device).view(1, C, 1)

        # Store indices for black sites.
        self.register_buffer('black_batch_idx',
                             batch_idx.expand(B, C, self.num_black))
        self.register_buffer('black_chain_idx',
                             chain_idx.expand(B, C, self.num_black))
        self.register_buffer('black_i_sites', black_indices_2d[:, 0].expand(
            B, C, self.num_black))
        self.register_buffer('black_j_sites', black_indices_2d[:, 1].expand(
            B, C, self.num_black))

        # Store indices for white sites.
        self.register_buffer('white_batch_idx',
                             batch_idx.expand(B, C, self.num_white))
        self.register_buffer('white_chain_idx',
                             chain_idx.expand(B, C, self.num_white))
        self.register_buffer('white_i_sites', white_indices_2d[:, 0].expand(
            B, C, self.num_white))
        self.register_buffer('white_j_sites', white_indices_2d[:, 1].expand(
            B, C, self.num_white))

        # Pre-compute and store neighbor indices.
        black_neighbors_i, black_neighbors_j = self._compute_neighbors(
            black_indices_2d)
        white_neighbors_i, white_neighbors_j = self._compute_neighbors(
            white_indices_2d)
        self.register_buffer(
            'black_neighbors_i',
            black_neighbors_i.view(1, 1, self.num_black, 4).expand(
                B, C, self.num_black, 4))
        self.register_buffer(
            'black_neighbors_j',
            black_neighbors_j.view(1, 1, self.num_black, 4).expand(
                B, C, self.num_black, 4))
        self.register_buffer(
            'white_neighbors_i',
            white_neighbors_i.view(1, 1, self.num_white, 4).expand(
                B, C, self.num_white, 4))
        self.register_buffer(
            'white_neighbors_j',
            white_neighbors_j.view(1, 1, self.num_white, 4).expand(
                B, C, self.num_white, 4))

    def _compute_neighbors(
        self, indices_2d: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes neighbor coordinates for a given set of sites.

        Args:
            indices_2d: A tensor of shape [N, 2] containing the (i, j)
                coordinates of N sites.

        Returns:
            A tuple of two tensors, (neighbor_i, neighbor_j), each of shape
            [N, 4], containing the i and j coordinates of the four neighbors
            (up, down, left, right) for each site.
        """
        L = self.L
        i, j = indices_2d[:, 0], indices_2d[:, 1]
        # Neighbors: (i-1, j), (i+1, j), (i, j-1), (i, j+1) with periodic BCs.
        neighbor_i = torch.stack(
            [(i - 1 + L) % L, (i + 1) % L, i, i], dim=1)
        neighbor_j = torch.stack(
            [j, j, (j - 1 + L) % L, (j + 1) % L], dim=1)
        return neighbor_i, neighbor_j

    def forward(self, n_sweeps: int = 5000, n_therm: int = 2000,
            decorrelate: int = 100, pt_interval: int = 10) -> Tensor:
        """Runs the main Monte Carlo simulation loop.

        This method executes the thermalization and production phases of the
        simulation, including periodic parallel tempering and proposal width
        tuning if enabled.

        Args:
            n_sweeps: The number of production sweeps to perform after
                thermalization.
            n_therm: The number of thermalization sweeps to perform to bring
                the system to equilibrium.
            decorrelate: The interval (in sweeps) at which to save a sample
                of the field configuration.
            pt_interval: The interval (in sweeps) at which to attempt
                parallel tempering exchanges.

        Returns:
            A tensor containing the collected samples, shaped as
            [batch_size, num_collected_samples, L, L].
        """
        num_samples_to_collect = n_sweeps // decorrelate
        if num_samples_to_collect == 0:
            return torch.empty((self.batch_size, 0, self.L, self.L),
                               dtype=self.field.dtype, device=self.device)

        # Determine where to store samples based on memory settings.
        sample_storage_device = torch.device(
            'cpu') if self.large_size_simulate else self.device
        samples = torch.empty(
            (num_samples_to_collect, self.batch_size, self.n_chains, self.L,
             self.L),
            dtype=self.field.dtype, device=sample_storage_device)

        use_amp_effective = self.use_amp and self.device.type in ('cuda', 'mps')
        sample_idx = 0

        # Check if the subclass has implemented the proposal width tuning.
        tuning_enabled = hasattr(self, 'adjust_proposal_width') and hasattr(
            self, 'tune_interval')

        print(f"Starting simulation: {n_therm} thermalization sweeps, "
              f"{n_sweeps} production sweeps.")
        print("Proposal width tuning is "
              f"{'ENABLED' if tuning_enabled else 'DISABLED'}.")

        with torch.no_grad():
            # Use autocast for potential performance gains with mixed precision.
            with torch.autocast(device_type=self.device.type,
                                enabled=use_amp_effective):
                # --- Thermalization Loop ---
                for sweep in range(n_therm):
                    self.one_sweep()
                    if self.pt_enabled and (sweep + 1) % pt_interval == 0:
                        self.parallel_tempering_exchange()
                    if tuning_enabled and (sweep + 1) % self.tune_interval == 0:
                        self.adjust_proposal_width(is_thermalization=True)

                # --- Production Loop ---
                for sweep in range(n_sweeps):
                    self.one_sweep()
                    if self.pt_enabled and (sweep + 1) % pt_interval == 0:
                        self.parallel_tempering_exchange()
                    if tuning_enabled and (sweep + 1) % self.tune_interval == 0:
                         self.adjust_proposal_width(is_thermalization=False)

                    # Collect a sample at the specified interval.
                    if (sweep + 1) % decorrelate == 0 and \
                       sample_idx < num_samples_to_collect:
                        current_field = self.field
                        samples[sample_idx] = current_field.cpu(
                        ) if self.large_size_simulate else current_field
                        sample_idx += 1

        print(f"Simulation finished. Collected {sample_idx * self.n_chains} "
              "samples per kappa.")
        # Reshape samples to [B, C * num_samples, L, L] for easier analysis.
        samples = samples.permute(1, 2, 0, 3, 4).contiguous()
        final_samples = samples.view(self.batch_size, -1, self.L, self.L)
        del samples
        gc.collect()
        return final_samples

    def one_sweep(self) -> None:
        """Performs a single full Monte Carlo sweep.

        A sweep consists of updating all black sites followed by all white
        sites, ensuring that updates within a sub-lattice are independent.
        """
        self.metropolis_update_sub_lattice('black')
        self.metropolis_update_sub_lattice('white')

    def parallel_tempering_exchange(self) -> None:
        """Attempts to swap configurations between adjacent kappa values.

        This enhances sampling efficiency by allowing configurations to travel
        across the kappa space, overcoming potential energy barriers. The swap
        is accepted based on the Metropolis criterion for the combined system.
        """
        if self.batch_size < 2:
            return  # Not enough replicas to perform an exchange.

        try:
            K = self.compute_kinetic_term()  # Shape: [B, C]
        except NotImplementedError:
            print("Warning: Skipping PT because compute_kinetic_term() "
                  "is not implemented.")
            return

        # Randomly choose whether to pair (0,1), (2,3), ... or (1,2), (3,4),...
        B = self.batch_size
        idx = torch.arange(B, device=self.device)
        start_indices = idx[::2] if torch.rand(()) < 0.5 else idx[1::2]
        # Ensure pairs are within bounds.
        valid_pairs = start_indices[start_indices < (B - 1)]
        if valid_pairs.numel() == 0:
            return
        partner_indices = valid_pairs + 1

        # Calculate the change in action for the swap.
        K_a, K_b = K[valid_pairs], K[partner_indices]
        kappa_a = self.kappa[valid_pairs].view(-1, 1)
        kappa_b = self.kappa[partner_indices].view(-1, 1)
        delta = (kappa_a - kappa_b) * (K_a - K_b)

        # Accept or reject the swap based on the Metropolis criterion.
        log_u = torch.log(torch.rand_like(delta, dtype=torch.float32))
        swap_mask = log_u < delta  # Acceptance probability is exp(delta)

        # Perform the swap for the accepted pairs.
        if swap_mask.any():
            # Expand mask to field dimensions for torch.where.
            m = swap_mask.view(swap_mask.shape[0], self.n_chains, 1, 1)
            field_a = self.field[valid_pairs]
            field_b = self.field[partner_indices]
            # Use a temporary tensor to avoid data corruption during the swap.
            field_a_tmp = field_a.clone()
            self.field[valid_pairs] = torch.where(m, field_b, field_a)
            self.field[partner_indices] = torch.where(m, field_a_tmp, field_b)
