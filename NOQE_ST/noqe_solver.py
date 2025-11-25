"""
NOQE Solver - Generalized Eigenvalue Problem
Solves H·c = E·S·c to obtain molecular energies
"""
import numpy as np
from scipy.linalg import eigh
from typing import Tuple, List, Optional


class NOQESolver:
    """
    Solve the generalized eigenvalue problem for NOQE

    From paper Eq. (3): H⃗c = E S⃗c
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def solve(self, H: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve generalized eigenvalue problem

        Args:
            H: Hamiltonian matrix (M x M)
            S: Overlap matrix (M x M)

        Returns:
            Tuple of (eigenvalues, eigenvectors)
            eigenvalues: Array of energies sorted in ascending order
            eigenvectors: Matrix where column k is eigenvector for eigenvalue k
        """
        # Check matrix properties
        if self.verbose:
            print(f"Hamiltonian matrix shape: {H.shape}")
            print(f"Overlap matrix shape: {S.shape}")
            print(f"H Hermiticity error: {np.max(np.abs(H - H.conj().T))}")
            print(f"S Hermiticity error: {np.max(np.abs(S - S.conj().T))}")

        # Ensure Hermitian (symmetrize)
        H = (H + H.conj().T) / 2
        S = (S + S.conj().T) / 2

        # Check if S is positive definite
        s_eigenvals = np.linalg.eigvalsh(S)
        if self.verbose:
            print(f"Overlap matrix eigenvalues: {s_eigenvals}")

        if np.min(s_eigenvals) < 1e-10:
            print("Warning: Overlap matrix is nearly singular")
            # Add small regularization
            S = S + 1e-10 * np.eye(S.shape[0])

        # Solve generalized eigenvalue problem
        # eigh solves: H @ v = lambda @ S @ v
        eigenvalues, eigenvectors = eigh(H, S)

        if self.verbose:
            print(f"\nNOQE Eigenvalues (energies):")
            for i, E in enumerate(eigenvalues):
                print(f"  State {i}: E = {E.real:.6f} Hartree")

        return eigenvalues, eigenvectors

    def get_ground_state_energy(self, H: np.ndarray, S: np.ndarray) -> float:
        """
        Get ground state energy

        Args:
            H: Hamiltonian matrix
            S: Overlap matrix

        Returns:
            Ground state energy (lowest eigenvalue)
        """
        eigenvalues, _ = self.solve(H, S)
        return eigenvalues[0].real

    def get_excited_state_energies(self, H: np.ndarray, S: np.ndarray,
                                   num_states: int) -> np.ndarray:
        """
        Get excited state energies

        Args:
            H: Hamiltonian matrix
            S: Overlap matrix
            num_states: Number of excited states to return

        Returns:
            Array of energies for lowest num_states states
        """
        eigenvalues, _ = self.solve(H, S)
        return eigenvalues[:num_states].real

    def analyze_solution(self, H: np.ndarray, S: np.ndarray,
                        eigenvalues: np.ndarray, eigenvectors: np.ndarray):
        """
        Analyze the NOQE solution

        Args:
            H: Hamiltonian matrix
            S: Overlap matrix
            eigenvalues: Computed eigenvalues
            eigenvectors: Computed eigenvectors
        """
        print("\n" + "="*60)
        print("NOQE Solution Analysis")
        print("="*60)

        M = H.shape[0]

        for k in range(M):
            E_k = eigenvalues[k]
            c_k = eigenvectors[:, k]

            print(f"\nState {k}:")
            print(f"  Energy: {E_k.real:.8f} Hartree")

            # Verify solution: H·c = E·S·c
            Hc = H @ c_k
            ESc = E_k * (S @ c_k)
            residual = np.linalg.norm(Hc - ESc)
            print(f"  Residual ||H·c - E·S·c||: {residual:.2e}")

            # Normalization: c†·S·c = 1
            norm = np.conj(c_k) @ S @ c_k
            print(f"  Normalization c†·S·c: {norm.real:.6f}")

            # Coefficient magnitudes
            print(f"  Coefficients: {np.abs(c_k)}")


class ChemicalAccuracyChecker:
    """Check if energies meet chemical accuracy criteria"""

    # Chemical accuracy: 1.6 mHa ≈ 1 kcal/mol
    CHEMICAL_ACCURACY_HARTREE = 1.6e-3

    @staticmethod
    def check_accuracy(computed_energy: float,
                      exact_energy: float,
                      tolerance: Optional[float] = None) -> bool:
        """
        Check if computed energy is within chemical accuracy

        Args:
            computed_energy: Computed energy value
            exact_energy: Exact reference energy
            tolerance: Custom tolerance (default: chemical accuracy)

        Returns:
            True if within accuracy
        """
        if tolerance is None:
            tolerance = ChemicalAccuracyChecker.CHEMICAL_ACCURACY_HARTREE

        error = abs(computed_energy - exact_energy)

        print(f"\nAccuracy Check:")
        print(f"  Computed energy: {computed_energy:.8f} Ha")
        print(f"  Exact energy:    {exact_energy:.8f} Ha")
        print(f"  Absolute error:  {error:.2e} Ha")
        print(f"  Tolerance:       {tolerance:.2e} Ha")

        within_accuracy = error < tolerance

        if within_accuracy:
            print(f"  ✓ Within chemical accuracy!")
        else:
            print(f"  ✗ Outside chemical accuracy")

        return within_accuracy

    @staticmethod
    def relative_error(computed_energy: float, exact_energy: float) -> float:
        """Calculate relative error percentage"""
        return abs(computed_energy - exact_energy) / abs(exact_energy) * 100
