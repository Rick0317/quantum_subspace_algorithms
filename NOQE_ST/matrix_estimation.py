"""
Matrix Element Estimation for Shadow-based NOQE
Implements estimation of Hamiltonian and overlap matrices from classical shadows
"""
import numpy as np
from typing import List, Dict, Tuple
from shadow_tomography import ShadowEstimator


class MatrixElementEstimator:
    """Estimate matrix elements from classical shadows"""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits
        self.aux_dim = 2**(num_qubits + 1)  # Auxiliary states have ancilla qubit
        self.estimator = ShadowEstimator()

    def estimate_overlap_magnitude(self,
                                   shadows_i: List[np.ndarray],
                                   shadows_j: List[np.ndarray]) -> float:
        """
        Estimate |S_ij|^2 = Tr(|ψ_i⟩⟨ψ_i| |ψ_j⟩⟨ψ_j|)

        Args:
            shadows_i: Classical shadows for state i
            shadows_j: Classical shadows for state j

        Returns:
            |S_ij|^2
        """
        identity = np.eye(self.dim)
        return self.estimator.estimate_bilinear(shadows_i, shadows_j, identity)

    def estimate_overlap_real(self,
                             shadows_R_i: List[np.ndarray],
                             shadows_R_j: List[np.ndarray],
                             overlap_mag_sq: float) -> float:
        """
        Estimate Re(S_ij) using auxiliary states

        From paper Eq. (22):
        Re(S_ij) = 2 Tr(|ψ^R_i⟩⟨ψ^R_i| |ψ^R_j⟩⟨ψ^R_j|) - (1 + |S_ij|^2)/2

        Args:
            shadows_R_i: Shadows for auxiliary state R_i
            shadows_R_j: Shadows for auxiliary state R_j
            overlap_mag_sq: Previously computed |S_ij|^2

        Returns:
            Re(S_ij)
        """
        # Auxiliary states are in extended space (n+1 qubits)
        identity = np.eye(self.aux_dim)
        aux_overlap = self.estimator.estimate_bilinear(shadows_R_i, shadows_R_j, identity)

        re_sij = 2 * aux_overlap - 0.5 * (1 + overlap_mag_sq)
        return re_sij

    def estimate_overlap_imag(self,
                             shadows_I_i: List[np.ndarray],
                             shadows_R_j: List[np.ndarray],
                             overlap_mag_sq: float) -> float:
        """
        Estimate Im(S_ij) using auxiliary states

        From paper Eq. (23):
        Im(S_ij) = 2 Tr(|ψ^I_i⟩⟨ψ^I_i| |ψ^R_j⟩⟨ψ^R_j|) - (1 + |S_ij|^2)/2

        Args:
            shadows_I_i: Shadows for auxiliary state I_i
            shadows_R_j: Shadows for auxiliary state R_j
            overlap_mag_sq: Previously computed |S_ij|^2

        Returns:
            Im(S_ij)
        """
        # Auxiliary states are in extended space (n+1 qubits)
        identity = np.eye(self.aux_dim)
        aux_overlap = self.estimator.estimate_bilinear(shadows_I_i, shadows_R_j, identity)

        im_sij = 2 * aux_overlap - 0.5 * (1 + overlap_mag_sq)
        return im_sij

    def estimate_overlap_element(self,
                                shadows_i: List[np.ndarray],
                                shadows_j: List[np.ndarray],
                                shadows_R_i: List[np.ndarray],
                                shadows_R_j: List[np.ndarray],
                                shadows_I_i: List[np.ndarray]) -> complex:
        """
        Estimate complete overlap matrix element S_ij

        Args:
            shadows_i: Shadows for reference state i
            shadows_j: Shadows for reference state j
            shadows_R_i: Shadows for auxiliary R state i
            shadows_R_j: Shadows for auxiliary R state j
            shadows_I_i: Shadows for auxiliary I state i

        Returns:
            Complex overlap S_ij
        """
        # Step 1: Estimate |S_ij|^2
        overlap_mag_sq = self.estimate_overlap_magnitude(shadows_i, shadows_j)

        # Step 2: Estimate Re(S_ij)
        re_part = self.estimate_overlap_real(shadows_R_i, shadows_R_j, overlap_mag_sq)

        # Step 3: Estimate Im(S_ij)
        im_part = self.estimate_overlap_imag(shadows_I_i, shadows_R_j, overlap_mag_sq)

        return complex(re_part, im_part)

    def estimate_hamiltonian_element(self,
                                    shadows_i: List[np.ndarray],
                                    shadows_j: List[np.ndarray],
                                    hamiltonian: np.ndarray,
                                    overlap_ij: complex,
                                    is_diagonal: bool = False) -> complex:
        """
        Estimate Hamiltonian matrix element H_ij = <ψ_i|H|ψ_j>

        For diagonal elements (i=j): H_ii = <ψ_i|H|ψ_i> = Tr(ρ_i H)
        This uses simple linear estimation from shadows.

        For off-diagonal elements: We estimate via the bilinear form
        Tr(ρ_i H ρ_j) which approximates <ψ_i|H|ψ_j> for pure states.

        Args:
            shadows_i: Shadows for reference state i
            shadows_j: Shadows for reference state j
            hamiltonian: Hamiltonian operator
            overlap_ij: Previously computed S_ij
            is_diagonal: Whether this is a diagonal element

        Returns:
            H_ij
        """
        if is_diagonal:
            # Diagonal: H_ii = <ψ_i|H|ψ_i> = Tr(ρ_i H)
            # Use linear estimation which is exact for expectation values
            return complex(self.estimator.estimate_linear(shadows_i, hamiltonian), 0)

        # Off-diagonal estimation for H_ij = <ψ_i|H|ψ_j>
        #
        # From the paper, the key relation is:
        #   H_ij · S_ij = Tr(ρ_j ρ_i H)
        #
        # Proof for pure states ρ_i = |ψ_i><ψ_i|, ρ_j = |ψ_j><ψ_j|:
        #   Tr(ρ_j ρ_i H) = Tr(|ψ_j><ψ_j|ψ_i><ψ_i| H)
        #                 = <ψ_j|ψ_i> · Tr(|ψ_j><ψ_i| H)
        #                 = <ψ_j|ψ_i> · <ψ_i|H|ψ_j>
        #                 = S_ji · H_ij
        #
        # Therefore: H_ij = Tr(ρ_j ρ_i H) / S_ji = Tr(ρ_j ρ_i H) / conj(S_ij)
        #
        # The estimate_bilinear function computes Tr(ρ_a O ρ_b) for inputs (shadows_a, shadows_b, O).
        # We need Tr(ρ_j ρ_i H). Using cyclic property of trace:
        #   Tr(ρ_j ρ_i H) = Tr(H ρ_j ρ_i) = Tr(ρ_i H ρ_j)
        #
        # So estimate_bilinear(shadows_i, shadows_j, H) = Tr(ρ_i H ρ_j) = Tr(ρ_j ρ_i H)
        #
        # This gives us: H_ij · S_ji = Tr(ρ_j ρ_i H)
        # Therefore: H_ij = Tr(ρ_j ρ_i H) / S_ji = Tr(ρ_i H ρ_j) / conj(S_ij)

        hij_times_sji = self.estimator.estimate_bilinear(shadows_i, shadows_j, hamiltonian)

        # For orthogonal states (S_ij ≈ 0), the bilinear form gives ~0
        # Cannot extract H_ij by division; return raw estimate
        if abs(overlap_ij) < 1e-6:
            return complex(hij_times_sji, 0)

        # H_ij = Tr(ρ_i H ρ_j) / S_ji = Tr(ρ_i H ρ_j) / conj(S_ij)
        h_ij = hij_times_sji / np.conj(overlap_ij)

        return h_ij


class NOQEMatrixBuilder:
    """Build complete H and S matrices for NOQE"""

    def __init__(self, num_qubits: int, num_references: int):
        self.num_qubits = num_qubits
        self.num_references = num_references
        self.estimator = MatrixElementEstimator(num_qubits)

        # Storage for shadow datasets
        self.reference_shadows: Dict[int, List[np.ndarray]] = {}
        self.auxiliary_R_shadows: Dict[int, List[np.ndarray]] = {}
        self.auxiliary_I_shadows: Dict[int, List[np.ndarray]] = {}

    def add_shadow_data(self,
                       ref_idx: int,
                       ref_shadows: List[np.ndarray],
                       aux_R_shadows: List[np.ndarray],
                       aux_I_shadows: List[np.ndarray]):
        """
        Add shadow data for a reference state

        Args:
            ref_idx: Reference state index
            ref_shadows: Shadows for reference state
            aux_R_shadows: Shadows for auxiliary R state
            aux_I_shadows: Shadows for auxiliary I state
        """
        self.reference_shadows[ref_idx] = ref_shadows
        self.auxiliary_R_shadows[ref_idx] = aux_R_shadows
        self.auxiliary_I_shadows[ref_idx] = aux_I_shadows

    def build_overlap_matrix(self, use_simple_estimation: bool = False) -> np.ndarray:
        """
        Build complete overlap matrix S

        Args:
            use_simple_estimation: If True, use |S_ij|^2 estimation from bilinear form
                                   If False, use auxiliary states (requires proper aux circuits)

        Returns:
            M x M overlap matrix
        """
        M = self.num_references
        S = np.zeros((M, M), dtype=complex)

        for i in range(M):
            # Diagonal elements are 1 (normalized states)
            S[i, i] = 1.0

            for j in range(i + 1, M):
                if use_simple_estimation:
                    # Simple estimation: just use |S_ij|^2 = Tr(rho_i @ rho_j)
                    # For pure orthogonal states, this gives 0
                    # For pure states with overlap s, this gives |s|^2
                    overlap_mag_sq = self.estimator.estimate_overlap_magnitude(
                        self.reference_shadows[i],
                        self.reference_shadows[j]
                    )
                    # Take sqrt and assume real positive (valid for many cases)
                    # For more complex cases, need auxiliary states
                    s_ij = np.sqrt(max(0, overlap_mag_sq))
                else:
                    # Full estimation using auxiliary states
                    s_ij = self.estimator.estimate_overlap_element(
                        self.reference_shadows[i],
                        self.reference_shadows[j],
                        self.auxiliary_R_shadows[i],
                        self.auxiliary_R_shadows[j],
                        self.auxiliary_I_shadows[i]
                    )

                S[i, j] = s_ij
                S[j, i] = np.conj(s_ij)  # Hermitian matrix

        return S

    def build_hamiltonian_matrix(self, hamiltonian: np.ndarray,
                                overlap_matrix: np.ndarray) -> np.ndarray:
        """
        Build complete Hamiltonian matrix H

        Args:
            hamiltonian: Hamiltonian operator
            overlap_matrix: Previously computed overlap matrix

        Returns:
            M x M Hamiltonian matrix
        """
        M = self.num_references
        H = np.zeros((M, M), dtype=complex)

        for i in range(M):
            for j in range(M):
                # Estimate H_ij
                h_ij = self.estimator.estimate_hamiltonian_element(
                    self.reference_shadows[i],
                    self.reference_shadows[j],
                    hamiltonian,
                    overlap_matrix[i, j],
                    is_diagonal=(i == j)
                )

                H[i, j] = h_ij

        return H

    def build_matrices(self, hamiltonian: np.ndarray,
                       use_simple_estimation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build both H and S matrices

        Args:
            hamiltonian: Hamiltonian operator
            use_simple_estimation: If True, use simple |S_ij|^2 estimation
                                   If False, use auxiliary states for complex overlap

        Returns:
            Tuple of (H_matrix, S_matrix)
        """
        S = self.build_overlap_matrix(use_simple_estimation=use_simple_estimation)
        H = self.build_hamiltonian_matrix(hamiltonian, S)

        return H, S
