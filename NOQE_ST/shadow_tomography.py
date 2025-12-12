"""
Shadow Tomography Module for NOQE
Implements classical shadows for quantum state estimation
"""
import numpy as np
from typing import List, Tuple, Optional
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Clifford, random_clifford


class ClassicalShadow:
    """Classical shadow for a quantum state using random Clifford measurements"""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.shadows = []

    def add_measurement(self, unitary_idx: int, measurement_outcome: str):
        """
        Add a measurement outcome to the classical shadow

        Args:
            unitary_idx: Index identifying which Clifford unitary was used
            measurement_outcome: Bitstring of measurement results
        """
        self.shadows.append((unitary_idx, measurement_outcome))

    def get_shadow_snapshot(self, unitary: np.ndarray, outcome: str) -> np.ndarray:
        """
        Construct a single classical shadow snapshot

        Args:
            unitary: The Clifford unitary used for measurement
            outcome: Measurement outcome bitstring

        Returns:
            Shadow density matrix snapshot
        """
        # Construct |b><b| from measurement outcome
        dim = 2**self.num_qubits
        state_vector = np.zeros(dim)
        outcome_int = int(outcome, 2)
        state_vector[outcome_int] = 1.0

        rho_b = np.outer(state_vector, state_vector)

        # Apply inverse channel: M^{-1}(U† |b><b| U)
        # For Clifford measurements: M^{-1}(σ) = (2^n + 1)σ - I
        unitary_dag = unitary.conj().T
        sigma = unitary_dag @ rho_b @ unitary

        rho_hat = (2**self.num_qubits + 1) * sigma - np.eye(dim)

        return rho_hat


class ShadowEstimator:
    """Estimate observables from classical shadows"""

    @staticmethod
    def estimate_linear(shadows: List[np.ndarray], observable: np.ndarray) -> float:
        """
        Estimate Tr(O * rho) from classical shadows

        Args:
            shadows: List of shadow density matrix snapshots
            observable: Observable operator O

        Returns:
            Estimated expectation value
        """
        n_shadows = len(shadows)
        estimate = sum(np.trace(observable @ shadow).real for shadow in shadows) / n_shadows
        return estimate

    @staticmethod
    def estimate_pauli_expectation(shadows: List[np.ndarray],
                                    pauli_string: str,
                                    pauli_coeff: float = 1.0) -> float:
        """
        Estimate expectation of a Pauli string more efficiently.

        For Pauli operators, the shadow estimator simplifies significantly
        and has lower variance than general observable estimation.

        Args:
            shadows: List of shadow density matrix snapshots
            pauli_string: String like "XZIY" representing Pauli operators
            pauli_coeff: Coefficient for this Pauli term

        Returns:
            Estimated expectation value
        """
        # Build Pauli matrix
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

        pauli_matrix = pauli_map[pauli_string[0]]
        for p in pauli_string[1:]:
            pauli_matrix = np.kron(pauli_matrix, pauli_map[p])

        n_shadows = len(shadows)
        estimate = sum(np.trace(pauli_matrix @ shadow).real for shadow in shadows) / n_shadows
        return pauli_coeff * estimate

    @staticmethod
    def estimate_bilinear(shadows_i: List[np.ndarray],
                         shadows_j: List[np.ndarray],
                         observable: np.ndarray,
                         use_median_of_means: bool = False,
                         num_groups: int = 10) -> float:
        """
        Estimate Tr(rho_i * O * rho_j) from classical shadows (VECTORIZED)

        This computes <psi_i|O|psi_j> = Tr(|psi_i><psi_i| O |psi_j><psi_j|)
        which equals Tr(rho_i @ O @ rho_j) using the cyclic property of trace.

        Uses vectorized numpy operations for significant speedup.

        Args:
            shadows_i: Classical shadows for state i
            shadows_j: Classical shadows for state j
            observable: Observable operator O
            use_median_of_means: If True, use median of means estimator
            num_groups: Number of groups for median of means

        Returns:
            Estimated bilinear function value
        """
        n_i = len(shadows_i)
        n_j = len(shadows_j)

        # Stack shadows into 3D arrays: (num_shadows, dim, dim)
        shadows_i_stack = np.stack(shadows_i)  # (n_i, d, d)
        shadows_j_stack = np.stack(shadows_j)  # (n_j, d, d)

        if not use_median_of_means or n_i < num_groups or n_j < num_groups:
            # Vectorized mean estimator
            # We want: sum_n sum_m Tr(shadow_i[n] @ O @ shadow_j[m])
            #
            # Tr(A @ B @ C) = sum_ijk A_ij * B_jk * C_ki
            # For batched computation over all (n,m) pairs:
            #   Tr(shadow_i[n] @ O @ shadow_j[m]) = sum_ijk shadow_i[n,i,j] * O[j,k] * shadow_j[m,k,i]
            #
            # Step 1: Compute O @ shadow_j for all m: result[m,j,i] = sum_k O[j,k] * shadow_j[m,k,i]
            # Step 2: Compute Tr(shadow_i @ result) for all (n,m) pairs

            # Compute O @ shadow_j for all j: (n_j, d, d)
            # O[j,k] @ shadow_j[m,k,l] -> result[m,j,l]
            O_shadows_j = np.einsum('jk,mkl->mjl', observable, shadows_j_stack)  # (n_j, d, d)

            # Compute Tr(shadow_i @ O_shadow_j) for all pairs
            # Tr(A @ B) = sum_ij A[i,j] * B[j,i]
            # For batched: sum_ij shadow_i[n,i,j] * O_shadows_j[m,j,i]
            trace_matrix = np.einsum('nij,mji->nm', shadows_i_stack, O_shadows_j)  # (n_i, n_j)

            total = np.sum(trace_matrix).real
            return total / (n_i * n_j)

        # Median of means with vectorized group computation
        group_size_i = n_i // num_groups
        group_size_j = n_j // num_groups

        group_means = []
        for g in range(num_groups):
            start_i = g * group_size_i
            end_i = start_i + group_size_i
            start_j = g * group_size_j
            end_j = start_j + group_size_j

            group_shadows_i = shadows_i_stack[start_i:end_i]  # (group_size_i, d, d)
            group_shadows_j = shadows_j_stack[start_j:end_j]  # (group_size_j, d, d)

            # Vectorized computation for this group
            O_shadows_j = np.einsum('jk,mkl->mjl', observable, group_shadows_j)
            trace_matrix = np.einsum('nij,mji->nm', group_shadows_i, O_shadows_j)

            count = trace_matrix.size
            if count > 0:
                group_means.append(np.sum(trace_matrix).real / count)

        return float(np.median(group_means))

    @staticmethod
    def u_statistics_order3(shadows: List[np.ndarray]) -> np.ndarray:
        """
        Construct order-3 U-statistics estimator for pure states

        Args:
            shadows: List of classical shadow snapshots

        Returns:
            U-statistics estimator rho^(U3)
        """
        n = len(shadows)
        if n < 3:
            raise ValueError("Need at least 3 shadows for order-3 U-statistics")

        estimator = np.zeros_like(shadows[0])
        count = 0

        # Sum over all distinct triples
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    estimator += shadows[i] @ shadows[j] @ shadows[k]
                    count += 1

        return estimator / count


def generate_random_clifford_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Generate a random Clifford unitary circuit

    Args:
        num_qubits: Number of qubits

    Returns:
        Quantum circuit implementing random Clifford
    """
    qr = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(qr)

    # Use Qiskit's random Clifford generator
    clifford = random_clifford(num_qubits)
    qc.append(clifford.to_instruction(), qr)

    return qc
