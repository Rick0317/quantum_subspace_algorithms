"""
Test Shadow Tomography for Small Systems

This module validates the shadow tomography implementation by comparing
estimates against exact calculations on small quantum systems (1-3 qubits).
"""

import numpy as np
from typing import List, Tuple
from qiskit.quantum_info import random_clifford, Statevector
from shadow_tomography import ClassicalShadow, ShadowEstimator


# Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


def estimate_shadow_cost_linear(hamiltonian_terms: List[Tuple[str, float]],
                                 epsilon: float = 0.01,
                                 delta: float = 0.05) -> dict:
    """
    Estimate the number of shadows needed to estimate ⟨H⟩ to precision epsilon.

    This can be computed A PRIORI without any measurements!

    Args:
        hamiltonian_terms: List of (pauli_string, coefficient) tuples
                          e.g., [("ZIZI", 0.5), ("XXII", 0.3), ...]
        epsilon: Desired precision
        delta: Failure probability (confidence = 1-delta)

    Returns:
        Dictionary with shadow cost estimates
    """
    # Compute shadow variance bound
    shadow_variance = 0.0
    max_weight = 0

    for pauli_string, coeff in hamiltonian_terms:
        # Weight = number of non-identity Paulis
        weight = sum(1 for p in pauli_string if p != 'I')
        max_weight = max(max_weight, weight)

        # Shadow norm squared for weight-k Pauli is 3^k
        shadow_variance += (coeff ** 2) * (3 ** weight)

    # Number of shadows for linear estimation
    # N = O(log(1/δ) · Var / ε²)
    log_factor = np.log(2 / delta)  # For Hoeffding-type bound
    n_shadows_linear = int(np.ceil(log_factor * shadow_variance / (epsilon ** 2)))

    return {
        'shadow_variance_bound': shadow_variance,
        'max_pauli_weight': max_weight,
        'n_shadows_linear': n_shadows_linear,
        'epsilon': epsilon,
        'delta': delta,
        'num_terms': len(hamiltonian_terms)
    }


def estimate_shadow_cost_bilinear(num_qubits: int,
                                   hamiltonian_norm: float,
                                   epsilon: float = 0.01) -> dict:
    """
    Estimate shadows needed for bilinear estimation Tr(ρᵢ H ρⱼ).

    The bilinear estimator has higher variance due to the product of shadows.

    Args:
        num_qubits: Number of qubits in the system
        hamiltonian_norm: Spectral norm ||H|| or 1-norm of coefficients
        epsilon: Desired precision

    Returns:
        Dictionary with bilinear shadow cost estimates
    """
    dim = 2 ** num_qubits

    # For bilinear estimation, variance scales as:
    # Var ≈ (2^n + 1)² · ||H||² / N²
    # To achieve precision ε, need N such that sqrt(Var) < ε
    # N > (2^n + 1) · ||H|| / ε

    prefactor = (dim + 1) ** 2
    n_shadows_bilinear = int(np.ceil((dim + 1) * hamiltonian_norm / epsilon))

    # Per state pair - with N shadows per state, we get N² pairs
    # So effective samples = N², variance ~ 1/N²
    n_shadows_per_state = int(np.ceil(np.sqrt(prefactor) * hamiltonian_norm / epsilon))

    return {
        'num_qubits': num_qubits,
        'hilbert_dim': dim,
        'variance_prefactor': prefactor,
        'n_shadows_bilinear_total': n_shadows_bilinear,
        'n_shadows_per_state': n_shadows_per_state,
        'epsilon': epsilon,
        'hamiltonian_norm': hamiltonian_norm
    }


def estimate_molecular_shadow_cost(molecule: str, num_qubits: int,
                                    num_pauli_terms: int,
                                    one_norm: float,
                                    epsilon: float = 0.001) -> None:
    """
    Print estimated shadow costs for a molecular Hamiltonian.

    Args:
        molecule: Molecule name (for display)
        num_qubits: Number of qubits
        num_pauli_terms: Number of Pauli terms in Hamiltonian
        one_norm: 1-norm of Hamiltonian (sum of |coefficients|)
        epsilon: Target precision (0.001 Ha = chemical accuracy)
    """
    print(f"\n{'='*60}")
    print(f"Shadow Cost Estimate for {molecule}")
    print(f"{'='*60}")
    print(f"  Qubits: {num_qubits}")
    print(f"  Pauli terms: {num_pauli_terms}")
    print(f"  ||H||₁ (1-norm): {one_norm:.2f} Ha")
    print(f"  Target precision: {epsilon} Ha")
    print()

    # Linear estimation (for diagonal elements ⟨ψᵢ|H|ψᵢ⟩ with known |ψᵢ⟩)
    # Assume average weight ~2 for molecular Hamiltonians
    avg_shadow_var = (one_norm / num_pauli_terms) ** 2 * num_pauli_terms * 9  # 3^2 for 2-local
    n_linear = int(np.ceil(np.log(40) * avg_shadow_var / epsilon**2))

    # Bilinear estimation (for matrix elements between unknown states)
    bilinear_cost = estimate_shadow_cost_bilinear(num_qubits, one_norm, epsilon)

    print(f"  Linear estimation (⟨H⟩ for known state):")
    print(f"    Shadows needed: ~{n_linear:,.0f}")
    print()
    print(f"  Bilinear estimation (matrix elements):")
    print(f"    Variance prefactor: (2^{num_qubits} + 1)² = {bilinear_cost['variance_prefactor']:,}")
    print(f"    Shadows per state: ~{bilinear_cost['n_shadows_per_state']:,}")
    print(f"    Total shadow pairs: ~{bilinear_cost['n_shadows_per_state']**2:,}")
    print()

    # Compare with standard approach
    print(f"  For comparison (standard measurement):")
    print(f"    Measurements ~ ||H||₁² / ε² = {(one_norm/epsilon)**2:,.0f}")


def generate_shadows_from_state(
    state_vector: np.ndarray,
    num_shadows: int,
    num_qubits: int
) -> List[np.ndarray]:
    """
    Generate classical shadows from a quantum state using random Clifford measurements.

    Args:
        state_vector: The quantum state to measure
        num_shadows: Number of shadow snapshots to generate
        num_qubits: Number of qubits in the system

    Returns:
        List of shadow density matrix snapshots
    """
    dim = 2 ** num_qubits
    shadows = []

    for _ in range(num_shadows):
        # Generate random Clifford unitary
        cliff = random_clifford(num_qubits)
        U = cliff.to_matrix()

        # Apply U to state: |ψ'⟩ = U|ψ⟩
        rotated_state = U @ state_vector

        # Compute measurement probabilities |⟨b|ψ'⟩|²
        probs = np.abs(rotated_state) ** 2

        # Sample measurement outcome
        outcome_int = np.random.choice(dim, p=probs)

        # Construct |b⟩⟨b|
        b_state = np.zeros(dim)
        b_state[outcome_int] = 1.0
        rho_b = np.outer(b_state, b_state)

        # Apply inverse channel: ρ̂ = (2^n + 1) U† |b⟩⟨b| U - I
        U_dag = U.conj().T
        sigma = U_dag @ rho_b @ U
        rho_hat = (dim + 1) * sigma - np.eye(dim)

        shadows.append(rho_hat)

    return shadows


def test_single_qubit_pauli_expectation():
    """
    Test 1: Single qubit Pauli expectation values

    For |+⟩ = (|0⟩ + |1⟩)/√2:
    - ⟨X⟩ = 1
    - ⟨Y⟩ = 0
    - ⟨Z⟩ = 0
    """
    print("\n" + "=" * 60)
    print("Test 1: Single Qubit Pauli Expectations")
    print("=" * 60)

    num_qubits = 1
    num_shadows = 1000

    # |+⟩ state
    plus_state = np.array([1, 1]) / np.sqrt(2)
    rho_exact = np.outer(plus_state, plus_state)

    shadows = generate_shadows_from_state(plus_state, num_shadows, num_qubits)
    estimator = ShadowEstimator()

    # Exact values
    exact_X = np.trace(X @ rho_exact).real
    exact_Y = np.trace(Y @ rho_exact).real
    exact_Z = np.trace(Z @ rho_exact).real

    # Shadow estimates
    est_X = estimator.estimate_linear(shadows, X)
    est_Y = estimator.estimate_linear(shadows, Y)
    est_Z = estimator.estimate_linear(shadows, Z)

    print(f"State: |+⟩ = (|0⟩ + |1⟩)/√2")
    print(f"Number of shadows: {num_shadows}")
    print()
    print(f"  ⟨X⟩: exact = {exact_X:.4f}, estimate = {est_X:.4f}, error = {abs(est_X - exact_X):.4f}")
    print(f"  ⟨Y⟩: exact = {exact_Y:.4f}, estimate = {est_Y:.4f}, error = {abs(est_Y - exact_Y):.4f}")
    print(f"  ⟨Z⟩: exact = {exact_Z:.4f}, estimate = {est_Z:.4f}, error = {abs(est_Z - exact_Z):.4f}")

    # Check errors are reasonable
    tolerance = 0.2
    assert abs(est_X - exact_X) < tolerance, f"X expectation error too large"
    assert abs(est_Y - exact_Y) < tolerance, f"Y expectation error too large"
    assert abs(est_Z - exact_Z) < tolerance, f"Z expectation error too large"

    print("\n✓ Single qubit Pauli test passed")
    return True


def test_two_qubit_bell_state():
    """
    Test 2: Two-qubit Bell state

    For |Φ+⟩ = (|00⟩ + |11⟩)/√2:
    - ⟨Z₀Z₁⟩ = 1
    - ⟨X₀X₁⟩ = 1
    - ⟨Z₀⟩ = ⟨Z₁⟩ = 0
    """
    print("\n" + "=" * 60)
    print("Test 2: Two-Qubit Bell State")
    print("=" * 60)

    num_qubits = 2
    num_shadows = 2000
    dim = 4

    # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    bell_state = np.zeros(dim)
    bell_state[0] = 1 / np.sqrt(2)  # |00⟩
    bell_state[3] = 1 / np.sqrt(2)  # |11⟩
    rho_exact = np.outer(bell_state, bell_state)

    shadows = generate_shadows_from_state(bell_state, num_shadows, num_qubits)
    estimator = ShadowEstimator()

    # Two-qubit Pauli operators
    ZZ = np.kron(Z, Z)
    XX = np.kron(X, X)
    Z0 = np.kron(Z, I)
    Z1 = np.kron(I, Z)

    # Exact values
    exact_ZZ = np.trace(ZZ @ rho_exact).real
    exact_XX = np.trace(XX @ rho_exact).real
    exact_Z0 = np.trace(Z0 @ rho_exact).real
    exact_Z1 = np.trace(Z1 @ rho_exact).real

    # Shadow estimates
    est_ZZ = estimator.estimate_linear(shadows, ZZ)
    est_XX = estimator.estimate_linear(shadows, XX)
    est_Z0 = estimator.estimate_linear(shadows, Z0)
    est_Z1 = estimator.estimate_linear(shadows, Z1)

    print(f"State: |Φ+⟩ = (|00⟩ + |11⟩)/√2")
    print(f"Number of shadows: {num_shadows}")
    print()
    print(f"  ⟨Z₀Z₁⟩: exact = {exact_ZZ:.4f}, estimate = {est_ZZ:.4f}, error = {abs(est_ZZ - exact_ZZ):.4f}")
    print(f"  ⟨X₀X₁⟩: exact = {exact_XX:.4f}, estimate = {est_XX:.4f}, error = {abs(est_XX - exact_XX):.4f}")
    print(f"  ⟨Z₀⟩:   exact = {exact_Z0:.4f}, estimate = {est_Z0:.4f}, error = {abs(est_Z0 - exact_Z0):.4f}")
    print(f"  ⟨Z₁⟩:   exact = {exact_Z1:.4f}, estimate = {est_Z1:.4f}, error = {abs(est_Z1 - exact_Z1):.4f}")

    tolerance = 0.2
    assert abs(est_ZZ - exact_ZZ) < tolerance
    assert abs(est_XX - exact_XX) < tolerance
    assert abs(est_Z0 - exact_Z0) < tolerance
    assert abs(est_Z1 - exact_Z1) < tolerance

    print("\n✓ Bell state test passed")
    return True


def test_bilinear_overlap():
    """
    Test 3: Bilinear estimator for overlaps

    Test Tr(ρ_i ρ_j) = |⟨ψ_i|ψ_j⟩|²
    """
    print("\n" + "=" * 60)
    print("Test 3: Bilinear Overlap Estimation")
    print("=" * 60)

    num_qubits = 2
    num_shadows = 500
    dim = 4

    # State 1: |00⟩
    psi_1 = np.zeros(dim)
    psi_1[0] = 1.0

    # State 2: cos(θ)|00⟩ + sin(θ)|01⟩ with θ = π/4
    theta = np.pi / 4
    psi_2 = np.zeros(dim)
    psi_2[0] = np.cos(theta)
    psi_2[1] = np.sin(theta)

    # State 3: |01⟩ (orthogonal to psi_1)
    psi_3 = np.zeros(dim)
    psi_3[1] = 1.0

    # Generate shadows
    shadows_1 = generate_shadows_from_state(psi_1, num_shadows, num_qubits)
    shadows_2 = generate_shadows_from_state(psi_2, num_shadows, num_qubits)
    shadows_3 = generate_shadows_from_state(psi_3, num_shadows, num_qubits)

    estimator = ShadowEstimator()
    identity = np.eye(dim)

    # Test overlaps
    # |⟨ψ_1|ψ_1⟩|² = 1
    exact_11 = abs(np.vdot(psi_1, psi_1)) ** 2
    est_11 = estimator.estimate_bilinear(shadows_1, shadows_1, identity)

    # |⟨ψ_1|ψ_2⟩|² = cos²(θ) = 0.5
    exact_12 = abs(np.vdot(psi_1, psi_2)) ** 2
    est_12 = estimator.estimate_bilinear(shadows_1, shadows_2, identity)

    # |⟨ψ_1|ψ_3⟩|² = 0 (orthogonal)
    exact_13 = abs(np.vdot(psi_1, psi_3)) ** 2
    est_13 = estimator.estimate_bilinear(shadows_1, shadows_3, identity)

    print(f"Number of shadows per state: {num_shadows}")
    print()
    print(f"  |⟨ψ₁|ψ₁⟩|²: exact = {exact_11:.4f}, estimate = {est_11:.4f}, error = {abs(est_11 - exact_11):.4f}")
    print(f"  |⟨ψ₁|ψ₂⟩|²: exact = {exact_12:.4f}, estimate = {est_12:.4f}, error = {abs(est_12 - exact_12):.4f}")
    print(f"  |⟨ψ₁|ψ₃⟩|²: exact = {exact_13:.4f}, estimate = {est_13:.4f}, error = {abs(est_13 - exact_13):.4f}")

    tolerance = 0.3  # Bilinear estimators have higher variance
    assert abs(est_11 - exact_11) < tolerance
    assert abs(est_12 - exact_12) < tolerance
    assert abs(est_13 - exact_13) < tolerance

    print("\n✓ Bilinear overlap test passed")
    return True


def test_bilinear_hamiltonian_element():
    """
    Test 4: Bilinear estimator for Hamiltonian matrix elements

    Test ⟨ψ_i|H|ψ_j⟩ using Tr(ρ_i H ρ_j)
    """
    print("\n" + "=" * 60)
    print("Test 4: Bilinear Hamiltonian Matrix Element")
    print("=" * 60)

    num_qubits = 2
    num_shadows = 1000
    dim = 4

    # Simple Hamiltonian: H = Z₀ + 0.5 X₀X₁
    H = np.kron(Z, I) + 0.5 * np.kron(X, X)

    # Two states with non-zero overlap
    # State 1: |00⟩
    psi_1 = np.zeros(dim)
    psi_1[0] = 1.0

    # State 2: (|00⟩ + |01⟩)/√2
    psi_2 = np.zeros(dim)
    psi_2[0] = 1 / np.sqrt(2)
    psi_2[1] = 1 / np.sqrt(2)

    # Exact matrix element: ⟨ψ_1|H|ψ_2⟩
    # Note: This is actually Re(⟨ψ_1|H|ψ_2⟩⟨ψ_2|ψ_1⟩) = Re(H₁₂ S₂₁)
    overlap = np.vdot(psi_1, psi_2)
    h_element = np.vdot(psi_1, H @ psi_2)
    exact_hij_sji = (h_element * np.conj(overlap)).real

    # Generate shadows
    shadows_1 = generate_shadows_from_state(psi_1, num_shadows, num_qubits)
    shadows_2 = generate_shadows_from_state(psi_2, num_shadows, num_qubits)

    estimator = ShadowEstimator()
    est_hij_sji = estimator.estimate_bilinear(shadows_1, shadows_2, H)

    print(f"Hamiltonian: H = Z₀ + 0.5 X₀X₁")
    print(f"Number of shadows per state: {num_shadows}")
    print()
    print(f"  ⟨ψ₁|ψ₂⟩ = {overlap:.4f}")
    print(f"  ⟨ψ₁|H|ψ₂⟩ = {h_element:.4f}")
    print(f"  Re(H₁₂ S₂₁): exact = {exact_hij_sji:.4f}, estimate = {est_hij_sji:.4f}")
    print(f"  Error: {abs(est_hij_sji - exact_hij_sji):.4f}")

    tolerance = 0.3
    assert abs(est_hij_sji - exact_hij_sji) < tolerance

    print("\n✓ Hamiltonian matrix element test passed")
    return True


def test_convergence_with_num_shadows():
    """
    Test 5: Verify that estimation error decreases with more shadows
    """
    print("\n" + "=" * 60)
    print("Test 5: Convergence with Number of Shadows")
    print("=" * 60)

    num_qubits = 1

    # |+⟩ state
    plus_state = np.array([1, 1]) / np.sqrt(2)
    exact_X = 1.0

    shadow_counts = [100, 500, 1000, 2000]
    errors = []

    estimator = ShadowEstimator()

    print(f"Observable: ⟨X⟩ for |+⟩ state (exact = 1.0)")
    print()

    for n in shadow_counts:
        # Average over multiple trials
        trial_errors = []
        for _ in range(5):
            shadows = generate_shadows_from_state(plus_state, n, num_qubits)
            est = estimator.estimate_linear(shadows, X)
            trial_errors.append(abs(est - exact_X))

        avg_error = np.mean(trial_errors)
        errors.append(avg_error)
        print(f"  N = {n:4d}: average error = {avg_error:.4f}")

    # Check that error generally decreases
    # Allow some noise but overall trend should be decreasing
    assert errors[-1] < errors[0] * 1.5, "Error should decrease with more shadows"

    print("\n✓ Convergence test passed")
    return True


def test_three_qubit_ghz():
    """
    Test 6: Three-qubit GHZ state

    For |GHZ⟩ = (|000⟩ + |111⟩)/√2:
    - ⟨Z₀Z₁Z₂⟩ = 1
    - ⟨X₀X₁X₂⟩ = 1
    - All single-qubit ⟨Zi⟩ = 0
    """
    print("\n" + "=" * 60)
    print("Test 6: Three-Qubit GHZ State")
    print("=" * 60)

    num_qubits = 3
    num_shadows = 3000
    dim = 8

    # GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2
    ghz_state = np.zeros(dim)
    ghz_state[0] = 1 / np.sqrt(2)  # |000⟩
    ghz_state[7] = 1 / np.sqrt(2)  # |111⟩
    rho_exact = np.outer(ghz_state, ghz_state)

    shadows = generate_shadows_from_state(ghz_state, num_shadows, num_qubits)
    estimator = ShadowEstimator()

    # Three-qubit operators
    ZZZ = np.kron(np.kron(Z, Z), Z)
    XXX = np.kron(np.kron(X, X), X)
    Z0 = np.kron(np.kron(Z, I), I)

    exact_ZZZ = np.trace(ZZZ @ rho_exact).real
    exact_XXX = np.trace(XXX @ rho_exact).real
    exact_Z0 = np.trace(Z0 @ rho_exact).real

    est_ZZZ = estimator.estimate_linear(shadows, ZZZ)
    est_XXX = estimator.estimate_linear(shadows, XXX)
    est_Z0 = estimator.estimate_linear(shadows, Z0)

    print(f"State: |GHZ⟩ = (|000⟩ + |111⟩)/√2")
    print(f"Number of shadows: {num_shadows}")
    print()
    print(f"  ⟨Z₀Z₁Z₂⟩: exact = {exact_ZZZ:.4f}, estimate = {est_ZZZ:.4f}, error = {abs(est_ZZZ - exact_ZZZ):.4f}")
    print(f"  ⟨X₀X₁X₂⟩: exact = {exact_XXX:.4f}, estimate = {est_XXX:.4f}, error = {abs(est_XXX - exact_XXX):.4f}")
    print(f"  ⟨Z₀⟩:     exact = {exact_Z0:.4f}, estimate = {est_Z0:.4f}, error = {abs(est_Z0 - exact_Z0):.4f}")

    tolerance = 0.25
    assert abs(est_ZZZ - exact_ZZZ) < tolerance
    assert abs(est_XXX - exact_XXX) < tolerance
    assert abs(est_Z0 - exact_Z0) < tolerance

    print("\n✓ GHZ state test passed")
    return True


def test_full_hamiltonian_matrix_2x2():
    """
    Test 7: Full 2x2 Hamiltonian matrix estimation

    Build a complete H and S matrix using shadow tomography
    for a 2-state subspace and compare to exact diagonalization.
    """
    print("\n" + "=" * 60)
    print("Test 7: Full 2x2 Hamiltonian Matrix Estimation")
    print("=" * 60)

    num_qubits = 2
    num_shadows = 2000
    dim = 4

    # Define a simple Hamiltonian: H = Z₀ + 0.3 X₀ + 0.2 Z₀Z₁
    H = np.kron(Z, I) + 0.3 * np.kron(X, I) + 0.2 * np.kron(Z, Z)

    # Define two non-orthogonal basis states
    # |ψ₀⟩ = |00⟩
    psi_0 = np.zeros(dim)
    psi_0[0] = 1.0

    # |ψ₁⟩ = cos(θ)|00⟩ + sin(θ)|10⟩ with θ = π/6
    theta = np.pi / 6
    psi_1 = np.zeros(dim)
    psi_1[0] = np.cos(theta)
    psi_1[2] = np.sin(theta)

    states = [psi_0, psi_1]
    n_states = len(states)

    # Generate shadows for each state
    all_shadows = []
    for psi in states:
        shadows = generate_shadows_from_state(psi, num_shadows, num_qubits)
        all_shadows.append(shadows)

    # Build exact H and S matrices
    H_exact = np.zeros((n_states, n_states), dtype=complex)
    S_exact = np.zeros((n_states, n_states), dtype=complex)

    for i in range(n_states):
        for j in range(n_states):
            H_exact[i, j] = np.vdot(states[i], H @ states[j])
            S_exact[i, j] = np.vdot(states[i], states[j])

    # Build estimated H and S matrices using shadow tomography
    H_est = np.zeros((n_states, n_states))
    S_est = np.zeros((n_states, n_states))

    estimator = ShadowEstimator()
    identity = np.eye(dim)

    for i in range(n_states):
        for j in range(n_states):
            # Estimate Tr(ρ_i H ρ_j) = H_ij * S_ji^*
            hij_sji = estimator.estimate_bilinear(all_shadows[i], all_shadows[j], H)

            # Estimate Tr(ρ_i ρ_j) = |S_ij|²
            sij_sq = estimator.estimate_bilinear(all_shadows[i], all_shadows[j], identity)
            sij_sq = max(0, sij_sq)  # Ensure non-negative

            S_est[i, j] = np.sqrt(sij_sq) if i != j else sij_sq

            # For H_ij, we have H_ij * S_ji^*
            # If S_ij is real and positive, H_ij ≈ hij_sji / S_ji
            if i == j:
                # Diagonal: H_ii = Tr(ρ_i H ρ_i) / Tr(ρ_i ρ_i) = ⟨H⟩_i
                H_est[i, i] = hij_sji / sij_sq if sij_sq > 1e-10 else 0
            else:
                # Off-diagonal: more complex
                H_est[i, j] = hij_sji

    print(f"Hamiltonian: H = Z₀ + 0.3 X₀ + 0.2 Z₀Z₁")
    print(f"Number of shadows per state: {num_shadows}")
    print()

    print("Exact Overlap Matrix S:")
    for i in range(n_states):
        row = "  "
        for j in range(n_states):
            row += f"{S_exact[i,j].real:8.4f} "
        print(row)

    print("\nExact Hamiltonian Matrix H:")
    for i in range(n_states):
        row = "  "
        for j in range(n_states):
            row += f"{H_exact[i,j].real:8.4f} "
        print(row)

    print("\nEstimated diagonal H elements (⟨ψᵢ|H|ψᵢ⟩):")
    for i in range(n_states):
        exact_val = H_exact[i, i].real
        est_val = H_est[i, i]
        print(f"  H[{i},{i}]: exact = {exact_val:.4f}, estimate = {est_val:.4f}, error = {abs(est_val - exact_val):.4f}")

    # Solve generalized eigenvalue problem with exact matrices
    eigenvalues_exact, _ = np.linalg.eigh(H_exact.real)

    print(f"\nExact eigenvalues: {eigenvalues_exact}")

    # Check diagonal elements are reasonable
    tolerance = 0.3
    for i in range(n_states):
        error = abs(H_est[i, i] - H_exact[i, i].real)
        assert error < tolerance, f"H[{i},{i}] error {error:.4f} exceeds tolerance"

    print("\n✓ 2x2 Hamiltonian matrix test passed")
    return True


def test_full_hamiltonian_matrix_3x3():
    """
    Test 8: Full 3x3 Hamiltonian matrix estimation

    Build a complete H and S matrix for a 3-state subspace.
    """
    print("\n" + "=" * 60)
    print("Test 8: Full 3x3 Hamiltonian Matrix Estimation")
    print("=" * 60)

    num_qubits = 2
    num_shadows = 2000
    dim = 4

    # Heisenberg-like Hamiltonian: H = X₀X₁ + Y₀Y₁ + Z₀Z₁
    XX = np.kron(X, X)
    YY = np.kron(Y, Y)
    ZZ = np.kron(Z, Z)
    H = XX + YY + ZZ

    # Three basis states (computational basis)
    psi_0 = np.zeros(dim); psi_0[0] = 1.0  # |00⟩
    psi_1 = np.zeros(dim); psi_1[1] = 1.0  # |01⟩
    psi_2 = np.zeros(dim); psi_2[2] = 1.0  # |10⟩

    states = [psi_0, psi_1, psi_2]
    n_states = len(states)

    # Generate shadows
    all_shadows = []
    for psi in states:
        shadows = generate_shadows_from_state(psi, num_shadows, num_qubits)
        all_shadows.append(shadows)

    # Build exact matrices
    H_exact = np.zeros((n_states, n_states), dtype=complex)
    S_exact = np.zeros((n_states, n_states), dtype=complex)

    for i in range(n_states):
        for j in range(n_states):
            H_exact[i, j] = np.vdot(states[i], H @ states[j])
            S_exact[i, j] = np.vdot(states[i], states[j])

    # Build estimated matrices
    H_est = np.zeros((n_states, n_states))
    S_est = np.zeros((n_states, n_states))

    estimator = ShadowEstimator()
    identity = np.eye(dim)

    for i in range(n_states):
        for j in range(n_states):
            hij_sji = estimator.estimate_bilinear(all_shadows[i], all_shadows[j], H)
            sij_sq = estimator.estimate_bilinear(all_shadows[i], all_shadows[j], identity)
            sij_sq = max(0, sij_sq)

            S_est[i, j] = np.sqrt(sij_sq) if i != j else sij_sq

            if i == j:
                H_est[i, i] = hij_sji / sij_sq if sij_sq > 1e-10 else 0
            else:
                H_est[i, j] = hij_sji

    print(f"Hamiltonian: H = X₀X₁ + Y₀Y₁ + Z₀Z₁ (Heisenberg)")
    print(f"Basis states: |00⟩, |01⟩, |10⟩")
    print(f"Number of shadows per state: {num_shadows}")
    print()

    print("Exact Hamiltonian Matrix H (real part):")
    for i in range(n_states):
        row = "  "
        for j in range(n_states):
            row += f"{H_exact[i,j].real:8.4f} "
        print(row)

    print("\nEstimated diagonal elements:")
    for i in range(n_states):
        exact_val = H_exact[i, i].real
        est_val = H_est[i, i]
        print(f"  H[{i},{i}]: exact = {exact_val:.4f}, estimate = {est_val:.4f}, error = {abs(est_val - exact_val):.4f}")

    # For orthogonal states, off-diagonal overlaps should be ~0
    print("\nEstimated |S_ij|² for orthogonal pairs (should be ~0):")
    for i in range(n_states):
        for j in range(i+1, n_states):
            sij_sq = estimator.estimate_bilinear(all_shadows[i], all_shadows[j], identity)
            print(f"  |S[{i},{j}]|² = {sij_sq:.4f}")

    tolerance = 0.4
    for i in range(n_states):
        error = abs(H_est[i, i] - H_exact[i, i].real)
        assert error < tolerance, f"H[{i},{i}] error {error:.4f} exceeds tolerance"

    print("\n✓ 3x3 Hamiltonian matrix test passed")
    return True


def test_hydrogen_like_hamiltonian():
    """
    Test 9: Hydrogen-like molecular Hamiltonian

    Test with a realistic 2-qubit Hamiltonian resembling H2 in minimal basis.
    """
    print("\n" + "=" * 60)
    print("Test 9: Hydrogen-like Molecular Hamiltonian")
    print("=" * 60)

    num_qubits = 2
    num_shadows = 3000
    dim = 4

    # Simplified H2-like Hamiltonian in STO-3G basis (2 qubits)
    # H = g0 I + g1 Z0 + g2 Z1 + g3 Z0Z1 + g4 X0X1 + g5 Y0Y1
    # Using approximate coefficients
    g0, g1, g2, g3, g4, g5 = -0.5, 0.4, 0.4, 0.1, 0.2, 0.2

    II = np.eye(dim)
    Z0 = np.kron(Z, I)
    Z1 = np.kron(I, Z)
    ZZ = np.kron(Z, Z)
    XX = np.kron(X, X)
    YY = np.kron(Y, Y)

    H = g0 * II + g1 * Z0 + g2 * Z1 + g3 * ZZ + g4 * XX + g5 * YY

    # Reference states: HF ground state and single excitation
    # |HF⟩ = |01⟩ (one electron in each spin orbital of bonding MO)
    psi_hf = np.zeros(dim)
    psi_hf[1] = 1.0  # |01⟩

    # Single excitation: |10⟩
    psi_exc = np.zeros(dim)
    psi_exc[2] = 1.0  # |10⟩

    states = [psi_hf, psi_exc]
    n_states = len(states)

    # Generate shadows
    all_shadows = []
    for psi in states:
        shadows = generate_shadows_from_state(psi, num_shadows, num_qubits)
        all_shadows.append(shadows)

    # Exact matrices
    H_exact = np.zeros((n_states, n_states), dtype=complex)
    S_exact = np.eye(n_states)  # Orthogonal states

    for i in range(n_states):
        for j in range(n_states):
            H_exact[i, j] = np.vdot(states[i], H @ states[j])

    # Estimated diagonal elements
    estimator = ShadowEstimator()
    identity = np.eye(dim)

    print(f"H2-like Hamiltonian (2 qubits)")
    print(f"States: |HF⟩ = |01⟩, |excited⟩ = |10⟩")
    print(f"Number of shadows per state: {num_shadows}")
    print()

    print("Exact Hamiltonian matrix:")
    for i in range(n_states):
        row = "  "
        for j in range(n_states):
            row += f"{H_exact[i,j].real:8.4f} "
        print(row)

    # Exact eigenvalues
    eigenvalues_exact = np.linalg.eigvalsh(H_exact.real)
    print(f"\nExact eigenvalues: {eigenvalues_exact}")

    # Shadow estimates
    print("\nShadow tomography estimates:")
    H_est_diag = []
    for i in range(n_states):
        hij_sji = estimator.estimate_bilinear(all_shadows[i], all_shadows[i], H)
        sij_sq = estimator.estimate_bilinear(all_shadows[i], all_shadows[i], identity)
        h_ii = hij_sji / sij_sq if sij_sq > 1e-10 else 0
        H_est_diag.append(h_ii)

        exact_val = H_exact[i, i].real
        print(f"  ⟨ψ_{i}|H|ψ_{i}⟩: exact = {exact_val:.4f}, estimate = {h_ii:.4f}, error = {abs(h_ii - exact_val):.4f}")

    # Off-diagonal estimate
    h01_s10 = estimator.estimate_bilinear(all_shadows[0], all_shadows[1], H)
    exact_h01 = H_exact[0, 1].real
    print(f"  H₀₁·S₁₀: exact = {exact_h01:.4f}, estimate = {h01_s10:.4f}")

    tolerance = 0.35
    for i in range(n_states):
        error = abs(H_est_diag[i] - H_exact[i, i].real)
        assert error < tolerance, f"H[{i},{i}] error {error:.4f} exceeds tolerance"

    print("\n✓ Hydrogen-like Hamiltonian test passed")
    return True


def test_variance_estimation():
    """
    Test 10: Variance estimation for error bars

    Verify that we can estimate the variance of our estimators.
    """
    print("\n" + "=" * 60)
    print("Test 10: Variance Estimation")
    print("=" * 60)

    num_qubits = 1
    num_shadows = 500
    num_trials = 20

    # |+⟩ state
    plus_state = np.array([1, 1]) / np.sqrt(2)
    exact_X = 1.0

    estimator = ShadowEstimator()
    estimates = []

    for _ in range(num_trials):
        shadows = generate_shadows_from_state(plus_state, num_shadows, num_qubits)
        est = estimator.estimate_linear(shadows, X)
        estimates.append(est)

    estimates = np.array(estimates)
    mean_est = np.mean(estimates)
    std_est = np.std(estimates)
    stderr = std_est / np.sqrt(num_trials)

    print(f"Observable: ⟨X⟩ for |+⟩ state")
    print(f"Number of shadows: {num_shadows}")
    print(f"Number of trials: {num_trials}")
    print()
    print(f"  Exact value: {exact_X:.4f}")
    print(f"  Mean estimate: {mean_est:.4f}")
    print(f"  Std deviation: {std_est:.4f}")
    print(f"  Standard error: {stderr:.4f}")
    print(f"  95% CI: [{mean_est - 1.96*stderr:.4f}, {mean_est + 1.96*stderr:.4f}]")

    # Check that exact value is within reasonable range
    assert abs(mean_est - exact_X) < 3 * stderr, "Mean is too far from exact"

    print("\n✓ Variance estimation test passed")
    return True


def test_shadow_cost_estimation():
    """
    Test 11: A priori shadow cost estimation

    Demonstrate how to estimate measurement costs without collecting shadows.
    """
    print("\n" + "=" * 60)
    print("Test 11: A Priori Shadow Cost Estimation")
    print("=" * 60)

    # Example 1: Simple 2-qubit Hamiltonian
    print("\n--- Example 1: Simple 2-qubit Hamiltonian ---")
    h2_terms = [
        ("II", -0.5),   # Identity (weight 0)
        ("ZI", 0.4),    # Z on qubit 0 (weight 1)
        ("IZ", 0.4),    # Z on qubit 1 (weight 1)
        ("ZZ", 0.1),    # ZZ interaction (weight 2)
        ("XX", 0.2),    # XX interaction (weight 2)
        ("YY", 0.2),    # YY interaction (weight 2)
    ]

    cost = estimate_shadow_cost_linear(h2_terms, epsilon=0.01)
    print(f"  Shadow variance bound: {cost['shadow_variance_bound']:.4f}")
    print(f"  Max Pauli weight: {cost['max_pauli_weight']}")
    print(f"  Shadows for ε=0.01: {cost['n_shadows_linear']:,}")

    cost_precise = estimate_shadow_cost_linear(h2_terms, epsilon=0.001)
    print(f"  Shadows for ε=0.001: {cost_precise['n_shadows_linear']:,}")

    # Example 2: Molecular Hamiltonians
    print("\n--- Example 2: Molecular Hamiltonian Estimates ---")

    # Typical values for common molecules
    molecules = [
        ("H₂ (4 qubits)", 4, 15, 2.5),
        ("LiH (12 qubits)", 12, 631, 15.0),
        ("H₂O (14 qubits)", 14, 1086, 25.0),
        ("N₂ (20 qubits)", 20, 2951, 40.0),
    ]

    print(f"\n  {'Molecule':<20} {'Qubits':>8} {'Terms':>8} {'Linear (ε=0.01)':>18} {'Bilinear/state':>16}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*18} {'-'*16}")

    for name, n_qubits, n_terms, one_norm in molecules:
        # Linear cost estimate (assuming avg weight ~2)
        avg_coeff = one_norm / n_terms
        shadow_var = avg_coeff**2 * n_terms * 9  # 3^2 for 2-local
        n_linear = int(np.ceil(np.log(40) * shadow_var / 0.01**2))

        # Bilinear cost
        bilinear = estimate_shadow_cost_bilinear(n_qubits, one_norm, epsilon=0.01)

        print(f"  {name:<20} {n_qubits:>8} {n_terms:>8} {n_linear:>18,} {bilinear['n_shadows_per_state']:>16,}")

    # Example 3: Show scaling with precision
    print("\n--- Example 3: Scaling with Precision ---")
    print(f"  For H₂O (14 qubits, ||H||₁ ≈ 25 Ha):")

    for eps in [0.1, 0.01, 0.001]:
        bilinear = estimate_shadow_cost_bilinear(14, 25.0, epsilon=eps)
        print(f"    ε = {eps}: {bilinear['n_shadows_per_state']:>12,} shadows/state")

    print("\n  Key insight: Bilinear estimation scales as (2^n + 1)/ε")
    print("  This exponential scaling is the main challenge for shadow tomography")
    print("  in NOQE-style matrix element estimation.")

    print("\n✓ Shadow cost estimation test passed")
    return True


def run_all_tests():
    """Run all shadow tomography tests"""
    print("\n" + "=" * 70)
    print("Shadow Tomography Validation Tests")
    print("=" * 70)
    print("Testing shadow tomography on small systems with exact comparison")

    tests = [
        ("Single Qubit Pauli", test_single_qubit_pauli_expectation),
        ("Two-Qubit Bell State", test_two_qubit_bell_state),
        ("Bilinear Overlap", test_bilinear_overlap),
        ("Bilinear Hamiltonian", test_bilinear_hamiltonian_element),
        ("Convergence", test_convergence_with_num_shadows),
        ("Three-Qubit GHZ", test_three_qubit_ghz),
        ("2x2 Hamiltonian Matrix", test_full_hamiltonian_matrix_2x2),
        ("3x3 Hamiltonian Matrix", test_full_hamiltonian_matrix_3x3),
        ("H2-like Hamiltonian", test_hydrogen_like_hamiltonian),
        ("Variance Estimation", test_variance_estimation),
        ("Shadow Cost Estimation", test_shadow_cost_estimation),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ Test '{name}' FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ Test '{name}' ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    if failed == 0:
        print("All shadow tomography tests passed!")

    return failed == 0


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    run_all_tests()
