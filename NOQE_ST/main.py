import numpy as np
from typing import List, Tuple, Optional
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# Import our modules
from shadow_tomography import (
    ClassicalShadow,
    ShadowEstimator,
    generate_random_clifford_circuit
)
from reference_states import ReferenceStateManager
from matrix_estimation import NOQEMatrixBuilder
from error_mitigation import ShadowDistillation, ErrorMitigatedEstimator
from noqe_solver import NOQESolver, ChemicalAccuracyChecker


class ShadowNOQE:
    """
    Shadow Tomography Enhanced NOQE Algorithm

    Key advantages over original NOQE:
    - Measurement scaling: O(M) instead of O(M�)
    - Qubit requirements: N instead of 2N+1
    - Circuit depth: ~g/2 instead of ~g
    - Natural error mitigation via shadow distillation
    """

    def __init__(self,
                 num_qubits: int,
                 num_references: int,
                 num_shadows: int = 1000,
                 use_error_mitigation: bool = True,
                 mitigation_order: int = 3,
                 noise_strength: float = 0.0,
                 backend: str = 'aer_simulator',
                 verbose: bool = True):
        """
        Initialize Shadow-based NOQE

        Args:
            num_qubits: Number of qubits (spin-orbitals)
            num_references: Number of reference states M
            num_shadows: Number of classical shadows per state
            use_error_mitigation: Whether to use shadow distillation
            mitigation_order: Order for shadow distillation (typically 3)
            noise_strength: Noise level for simulation
            backend: Qiskit backend name
            verbose: Print detailed information
        """
        self.num_qubits = num_qubits
        self.num_references = num_references
        self.num_shadows = num_shadows
        self.use_error_mitigation = use_error_mitigation
        self.mitigation_order = mitigation_order
        self.verbose = verbose

        # Initialize components
        self.reference_manager = ReferenceStateManager(num_qubits, num_references)
        self.matrix_builder = NOQEMatrixBuilder(num_qubits, num_references)
        self.solver = NOQESolver(verbose=verbose)

        # Error mitigation
        if use_error_mitigation:
            self.distillation = ShadowDistillation(order=mitigation_order)
        else:
            self.distillation = None

        # Backend setup
        self.backend = AerSimulator()

        if self.verbose:
            print("="*70)
            print("Shadow Tomography Enhanced NOQE")
            print("="*70)
            print(f"Number of qubits: {num_qubits}")
            print(f"Number of reference states: {num_references}")
            print(f"Classical shadows per state: {num_shadows}")
            print(f"Error mitigation: {use_error_mitigation}")
            if use_error_mitigation:
                print(f"Mitigation order: {mitigation_order}")
            print("="*70)

    def add_reference_state(self,
                           uhf_occupation: List[int],
                           ucc_parameters: np.ndarray,
                           excitations: List[Tuple]):
        """
        Add a reference state to NOQE

        Args:
            uhf_occupation: UHF occupation vector
            ucc_parameters: Parameters for UCC ansatz
            excitations: List of excitation tuples
        """
        self.reference_manager.add_reference(
            uhf_occupation, ucc_parameters, excitations
        )

    def collect_classical_shadows(self) -> None:
        """
        Collect classical shadows for all reference and auxiliary states

        This is the core measurement phase using randomized Clifford measurements.
        """
        if self.verbose:
            print("\nCollecting classical shadows...")

        for ref_idx in range(self.num_references):
            if self.verbose:
                print(f"\nReference state {ref_idx}:")

            # Get circuits for this reference
            ref_circuit = self.reference_manager.reference_circuits[ref_idx]
            aux_R_circuit = self.reference_manager.auxiliary_R_circuits[ref_idx]
            aux_I_circuit = self.reference_manager.auxiliary_I_circuits[ref_idx]

            # Collect shadows for each circuit
            # Reference circuits use num_qubits, auxiliary circuits use num_qubits + 1 (ancilla)
            ref_shadows = self._collect_shadows_for_circuit(ref_circuit, "Reference")
            aux_R_shadows = self._collect_shadows_for_circuit(aux_R_circuit, "Auxiliary R",
                                                              num_qubits_override=self.num_qubits + 1)
            aux_I_shadows = self._collect_shadows_for_circuit(aux_I_circuit, "Auxiliary I",
                                                              num_qubits_override=self.num_qubits + 1)

            # Apply error mitigation if enabled
            if self.use_error_mitigation:
                ref_shadows = self.distillation.distill_shadow_list(ref_shadows)
                aux_R_shadows = self.distillation.distill_shadow_list(aux_R_shadows)
                aux_I_shadows = self.distillation.distill_shadow_list(aux_I_shadows)

            # Store shadows
            self.matrix_builder.add_shadow_data(
                ref_idx, ref_shadows, aux_R_shadows, aux_I_shadows
            )

    def _collect_shadows_for_circuit(self,
                                     base_circuit: QuantumCircuit,
                                     label: str,
                                     num_qubits_override: int = None) -> List[np.ndarray]:
        """
        Collect classical shadows for a single circuit using batched execution.

        Args:
            base_circuit: Base quantum circuit
            label: Label for verbose output
            num_qubits_override: If specified, use this instead of self.num_qubits
                                (needed for auxiliary circuits with ancilla)

        Returns:
            List of classical shadow density matrices
        """
        from qiskit.quantum_info import Operator

        n_qubits = num_qubits_override if num_qubits_override else self.num_qubits
        batch_size = 100  # Process circuits in batches

        shadows = []
        clifford_circuits = []
        measurement_circuits = []

        if self.verbose:
            print(f"  {label}: Preparing {self.num_shadows} circuits...")

        # Step 1: Generate all Clifford circuits and measurement circuits
        for _ in range(self.num_shadows):
            clifford_circuit = generate_random_clifford_circuit(n_qubits)
            clifford_circuits.append(clifford_circuit)

            qc = base_circuit.copy()
            qc.compose(clifford_circuit, inplace=True)
            qc.measure_all()
            measurement_circuits.append(qc)

        # Step 2: Transpile and execute in batches
        for batch_start in range(0, self.num_shadows, batch_size):
            batch_end = min(batch_start + batch_size, self.num_shadows)
            batch_circuits = measurement_circuits[batch_start:batch_end]

            # Transpile batch
            transpiled_batch = transpile(batch_circuits, self.backend)

            # Execute batch (1 shot each)
            result = self.backend.run(transpiled_batch, shots=1).result()

            # Process results
            for i, idx in enumerate(range(batch_start, batch_end)):
                counts = result.get_counts(i)
                outcome = list(counts.keys())[0]

                clifford_unitary = Operator(clifford_circuits[idx]).data
                shadow = self._construct_shadow_snapshot(clifford_unitary, outcome, n_qubits)
                shadows.append(shadow)

            if self.verbose:
                print(f"  {label}: {batch_end}/{self.num_shadows} shadows collected")

        return shadows

    def _construct_shadow_snapshot(self,
                                   unitary: np.ndarray,
                                   outcome: str,
                                   n_qubits: int = None) -> np.ndarray:
        """
        Construct a single classical shadow snapshot

        From paper Eq. (14-15):
        �_U,b = M^{-1}(U  |b��b| U) = (2^n + 1)(U  |b��b| U) - I

        Args:
            unitary: Clifford unitary matrix
            outcome: Measurement outcome bitstring

        Returns:
            Classical shadow density matrix
        """
        if n_qubits is None:
            n_qubits = self.num_qubits
        dim = 2**n_qubits

        # Create |b��b| from outcome
        state_vector = np.zeros(dim)
        outcome_int = int(outcome, 2)
        state_vector[outcome_int] = 1.0
        rho_b = np.outer(state_vector, state_vector)

        # Apply inverse channel
        unitary_dag = unitary.conj().T
        sigma = unitary_dag @ rho_b @ unitary

        # Classical shadow reconstruction
        rho_hat = (2**n_qubits + 1) * sigma - np.eye(dim)

        return rho_hat

    def estimate_matrices(self, hamiltonian: np.ndarray,
                          use_exact_off_diagonal: bool = False,
                          use_auxiliary_states: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate Hamiltonian and overlap matrices from classical shadows

        Args:
            hamiltonian: Hamiltonian operator matrix
            use_exact_off_diagonal: If True, compute off-diagonal elements exactly
                                    using statevector simulation (for validation)
            use_auxiliary_states: If True, use auxiliary states for complex overlap
                                  estimation per paper Eq. 22-23

        Returns:
            Tuple of (H_matrix, S_matrix)
        """
        if self.verbose:
            print("\nEstimating matrix elements from shadows...")
            if use_auxiliary_states:
                print("Using auxiliary states for complex overlap estimation...")

        H, S = self.matrix_builder.build_matrices(hamiltonian,
                                                   use_simple_estimation=not use_auxiliary_states)

        if use_exact_off_diagonal:
            # For validation: compute exact off-diagonal elements using statevector
            if self.verbose:
                print("Using exact statevector for off-diagonal elements...")
            H, S = self._compute_exact_matrices(hamiltonian, H, S)

        if self.verbose:
            print(f"Overlap matrix S:")
            print(S)
            print(f"\nHamiltonian matrix H:")
            print(H)

        return H, S

    def _compute_exact_matrices(self, hamiltonian: np.ndarray,
                                H_shadow: np.ndarray,
                                S_shadow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute exact matrix elements using statevector simulation.
        Keeps diagonal elements from shadow estimation, computes off-diagonal exactly.
        """
        M = self.num_references
        H = H_shadow.copy()
        S = S_shadow.copy()

        # Get statevectors for all reference states
        statevectors = []
        for ref_idx in range(M):
            ref_circuit = self.reference_manager.reference_circuits[ref_idx]
            sv = Statevector.from_instruction(ref_circuit)
            statevectors.append(sv.data)

        # Compute exact overlaps and Hamiltonian elements
        for i in range(M):
            for j in range(M):
                psi_i = statevectors[i]
                psi_j = statevectors[j]

                # S_ij = <psi_i|psi_j>
                S[i, j] = np.vdot(psi_i, psi_j)

                # H_ij = <psi_i|H|psi_j>
                H[i, j] = np.vdot(psi_i, hamiltonian @ psi_j)

        return H, S

    def solve_eigenproblem(self, H: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve generalized eigenvalue problem

        Args:
            H: Hamiltonian matrix
            S: Overlap matrix

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        if self.verbose:
            print("\nSolving generalized eigenvalue problem H�c = E�S�c...")

        eigenvalues, eigenvectors = self.solver.solve(H, S)

        return eigenvalues, eigenvectors

    def run(self, hamiltonian: np.ndarray,
            use_auxiliary_states: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run complete Shadow-NOQE algorithm

        Args:
            hamiltonian: Hamiltonian operator
            use_auxiliary_states: If True, use auxiliary states for complex overlap
                                  estimation per paper Eq. 22-23

        Returns:
            Tuple of (energies, eigenvectors)
        """
        if self.verbose:
            print("\n" + "="*70)
            print("Starting Shadow-NOQE Algorithm")
            print("="*70)

        # Step 1: Collect classical shadows
        self.collect_classical_shadows()

        # Step 2: Estimate matrices
        H, S = self.estimate_matrices(hamiltonian, use_auxiliary_states=use_auxiliary_states)

        # Step 3: Solve eigenvalue problem
        energies, eigenvectors = self.solve_eigenproblem(H, S)

        if self.verbose:
            print("\n" + "="*70)
            print("Shadow-NOQE Complete")
            print("="*70)

        return energies, eigenvectors

    def get_ground_state_energy(self, hamiltonian: np.ndarray) -> float:
        """
        Get ground state energy

        Args:
            hamiltonian: Hamiltonian operator

        Returns:
            Ground state energy
        """
        energies, _ = self.run(hamiltonian)
        return energies[0].real


def example_h2_molecule():
    """
    Example: H2 molecule at equilibrium (R ~ 0.74 Angstrom)
    Using STO-3G basis with a realistic molecular Hamiltonian

    The H2 Hamiltonian in the STO-3G basis (4 spin-orbitals) can be written as:
    H = h0*I + h1*(Z0 + Z1) + h2*(Z2 + Z3) + h3*(Z0*Z1 + Z2*Z3)
        + h4*(Z0*Z2 + Z1*Z3) + h5*(Z0*Z3 + Z1*Z2) + h6*(X0*X1*Y2*Y3 + ...)

    For simplicity, we construct the full 16x16 matrix directly.
    """
    print("\n" + "="*70)
    print("Example: H2 Molecule Shadow-NOQE")
    print("="*70)

    # System parameters
    num_qubits = 4  # STO-3G basis: 4 spin-orbitals
    num_references = 2  # Two reference states

    # Initialize Shadow-NOQE
    # Note: More shadows = lower variance but slower
    # 2000-5000 shadows typically gives reasonable accuracy
    noqe = ShadowNOQE(
        num_qubits=num_qubits,
        num_references=num_references,
        num_shadows=10000,
        use_error_mitigation=False,  # Disable for now - distillation may add overhead
        mitigation_order=3,
        verbose=True
    )

    # Define NON-ORTHOGONAL reference states (required for shadow-based NOQE)
    # The paper (Eq. 24) requires H_ij = Tr(ρ_i ρ_j H) / S_ij*, which needs S_ij ≠ 0
    #
    # We use UCC-style rotations to create superpositions:
    # |ψ_1⟩ = cos(θ₁)|1100⟩ + sin(θ₁)|0011⟩
    # |ψ_2⟩ = cos(θ₂)|1100⟩ + sin(θ₂)|0011⟩
    # With different angles, these have non-zero overlap

    # Reference 1: Small rotation from HF state
    uhf1 = [1, 1, 0, 0]  # Start from |1100⟩
    ucc_params1 = np.array([0.2])  # θ₁ = 0.2 radians
    excitations1 = [((0, 1), (2, 3))]  # Double excitation to |0011⟩

    # Reference 2: Larger rotation from HF state
    uhf2 = [1, 1, 0, 0]  # Also start from |1100⟩
    ucc_params2 = np.array([0.5])  # θ₂ = 0.5 radians (different angle)
    excitations2 = [((0, 1), (2, 3))]  # Same excitation operator

    noqe.add_reference_state(uhf1, ucc_params1, excitations1)
    noqe.add_reference_state(uhf2, ucc_params2, excitations2)

    # Realistic H2 Hamiltonian in STO-3G basis at R=0.74 Angstrom
    # These are approximate values from quantum chemistry calculations
    # The Hamiltonian is constructed in the computational basis
    hamiltonian = build_h2_hamiltonian()

    # Run Shadow-NOQE and compare with exact
    noqe.collect_classical_shadows()

    # Get shadow-estimated matrices
    H_shadow, S_shadow = noqe.estimate_matrices(hamiltonian, use_auxiliary_states=False)

    # Compute exact matrices for comparison
    H_exact, S_exact = noqe._compute_exact_matrices(hamiltonian, H_shadow, S_shadow)

    print("\n" + "="*50)
    print("Matrix Element Comparison (Shadow vs Exact)")
    print("="*50)
    print(f"S_01: Shadow = {S_shadow[0,1]:.4f}, Exact = {S_exact[0,1]:.4f}, Error = {abs(S_shadow[0,1] - S_exact[0,1]):.4f}")
    print(f"H_00: Shadow = {H_shadow[0,0].real:.4f}, Exact = {H_exact[0,0].real:.4f}, Error = {abs(H_shadow[0,0] - H_exact[0,0]):.4f}")
    print(f"H_11: Shadow = {H_shadow[1,1].real:.4f}, Exact = {H_exact[1,1].real:.4f}, Error = {abs(H_shadow[1,1] - H_exact[1,1]):.4f}")
    print(f"H_01: Shadow = {H_shadow[0,1].real:.4f}, Exact = {H_exact[0,1].real:.4f}, Error = {abs(H_shadow[0,1] - H_exact[0,1]):.4f}")

    # Test hybrid: shadow S with exact H
    print("\n--- Hybrid test: Shadow S + Exact H ---")
    H_hybrid = H_exact.copy()
    S_hybrid = S_shadow.copy()
    energies_hybrid, _ = noqe.solve_eigenproblem(H_hybrid, S_hybrid)
    print(f"Hybrid ground state: {energies_hybrid[0].real:.6f} Ha")

    # Solve with shadow matrices
    print("\n--- Shadow-based eigenvalue solution ---")
    energies_shadow, _ = noqe.solve_eigenproblem(H_shadow, S_shadow)

    # Solve with exact matrices
    print("\n--- Exact eigenvalue solution ---")
    energies_exact, _ = noqe.solve_eigenproblem(H_exact, S_exact)

    energies = energies_shadow

    print(f"\nShadow ground state: {energies_shadow[0].real:.6f} Ha")
    print(f"Exact NOQE ground state: {energies_exact[0].real:.6f} Ha")

    # Check accuracy
    # Exact H2 ground state energy at equilibrium is about -1.137 Hartree
    exact_energy = np.min(np.linalg.eigvalsh(hamiltonian))
    computed_energy = energies[0].real

    checker = ChemicalAccuracyChecker()
    checker.check_accuracy(computed_energy, exact_energy)

    print(f"\nGround state energy: {computed_energy:.8f} Hartree")
    print(f"Exact ground state energy: {exact_energy:.8f} Hartree")
    print(f"Excited state energies: {energies[1:].real}")


def build_h2_hamiltonian():
    """
    Build a realistic H2 Hamiltonian in the STO-3G basis.

    Uses the standard H2 Hamiltonian coefficients at equilibrium geometry.
    In the Jordan-Wigner encoding with 4 qubits (spin-orbitals).

    Returns:
        16x16 Hamiltonian matrix in computational basis
    """
    # Pauli matrices
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    def kron_n(*matrices):
        """Kronecker product of multiple matrices"""
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result

    # H2 Hamiltonian coefficients at R ~ 0.74 Angstrom (equilibrium)
    # These come from mapping the fermionic Hamiltonian to qubits
    # H = g0*I + g1*Z0 + g2*Z1 + g3*Z2 + g4*Z3
    #     + g5*Z0Z1 + g6*Z0Z2 + g7*Z0Z3 + g8*Z1Z2 + g9*Z1Z3 + g10*Z2Z3
    #     + g11*(X0X1Y2Y3 + Y0Y1X2X3 + Y0X1X2Y3 + X0Y1Y2X3)

    # Coefficients for H2 at equilibrium (approximately)
    g0 = -0.81261  # Nuclear repulsion + constant
    g1 = 0.17120   # Z0 coefficient
    g2 = 0.17120   # Z1 coefficient
    g3 = -0.22279  # Z2 coefficient
    g4 = -0.22279  # Z3 coefficient
    g5 = 0.16862   # Z0Z1
    g6 = 0.12054   # Z0Z2
    g7 = 0.16587   # Z0Z3
    g8 = 0.16587   # Z1Z2
    g9 = 0.12054   # Z1Z3
    g10 = 0.17435  # Z2Z3
    g11 = -0.04532 # Exchange terms

    # Build Hamiltonian
    H = np.zeros((16, 16), dtype=complex)

    # Identity term
    H += g0 * kron_n(I, I, I, I)

    # Single Z terms
    H += g1 * kron_n(Z, I, I, I)
    H += g2 * kron_n(I, Z, I, I)
    H += g3 * kron_n(I, I, Z, I)
    H += g4 * kron_n(I, I, I, Z)

    # ZZ terms
    H += g5 * kron_n(Z, Z, I, I)
    H += g6 * kron_n(Z, I, Z, I)
    H += g7 * kron_n(Z, I, I, Z)
    H += g8 * kron_n(I, Z, Z, I)
    H += g9 * kron_n(I, Z, I, Z)
    H += g10 * kron_n(I, I, Z, Z)

    # Exchange terms (create entanglement, crucial for correlation energy)
    H += g11 * kron_n(X, X, Y, Y)
    H += g11 * kron_n(Y, Y, X, X)
    H += g11 * kron_n(Y, X, X, Y)
    H += g11 * kron_n(X, Y, Y, X)

    return H.real  # Should be real for physical Hamiltonian


if __name__ == "__main__":
    example_h2_molecule()
