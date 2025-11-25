"""
Simple tests to verify the Shadow-NOQE implementation
"""
import numpy as np
from shadow_tomography import ClassicalShadow, ShadowEstimator
from reference_states import ReferenceStatePreparation
from matrix_estimation import MatrixElementEstimator
from error_mitigation import ShadowDistillation
from noqe_solver import NOQESolver


def test_shadow_estimator():
    """Test basic shadow tomography estimation"""
    print("\n" + "="*60)
    print("Testing Shadow Estimator")
    print("="*60)

    # Create a simple 2-qubit test
    num_qubits = 2
    dim = 2**num_qubits

    # Create a simple pure state |ψ⟩ = |01⟩
    psi = np.zeros(dim)
    psi[1] = 1.0
    rho = np.outer(psi, psi)

    # Create fake shadows (just use the true state for testing)
    shadows = [rho for _ in range(100)]

    # Test observable: Z₀
    observable = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ])

    estimator = ShadowEstimator()
    estimate = estimator.estimate_linear(shadows, observable)
    exact = np.trace(observable @ rho).real

    print(f"Estimated value: {estimate:.4f}")
    print(f"Exact value: {exact:.4f}")
    print(f"Error: {abs(estimate - exact):.2e}")

    assert abs(estimate - exact) < 0.1, "Shadow estimation error too large"
    print("✓ Shadow estimator test passed")


def test_u_statistics():
    """Test U-statistics estimator"""
    print("\n" + "="*60)
    print("Testing U-Statistics")
    print("="*60)

    num_qubits = 2
    dim = 2**num_qubits

    # Pure state
    psi = np.array([1, 0, 0, 0])
    rho = np.outer(psi, psi)

    # Create multiple shadows
    shadows = [rho for _ in range(10)]

    estimator = ShadowEstimator()
    rho_u3 = estimator.u_statistics_order3(shadows)

    # For pure state, ρ³ should equal ρ (up to normalization)
    error = np.linalg.norm(rho_u3 / np.trace(rho_u3) - rho)

    print(f"Normalization error: {error:.2e}")
    assert error < 0.1, "U-statistics error too large"
    print("✓ U-statistics test passed")


def test_reference_state_preparation():
    """Test reference state preparation"""
    print("\n" + "="*60)
    print("Testing Reference State Preparation")
    print("="*60)

    num_qubits = 4
    prep = ReferenceStatePreparation(num_qubits)

    # Create UHF state |1100⟩
    occupation = [1, 1, 0, 0]
    qc = prep.prepare_uhf_state(occupation)

    print(f"UHF circuit depth: {qc.depth()}")
    print(f"UHF circuit gates: {qc.count_ops()}")
    print("✓ Reference state preparation test passed")


def test_matrix_element_estimator():
    """Test matrix element estimation"""
    print("\n" + "="*60)
    print("Testing Matrix Element Estimator")
    print("="*60)

    num_qubits = 2
    dim = 2**num_qubits

    # Create two states
    psi_i = np.array([1, 0, 0, 0])
    psi_j = np.array([0, 1, 0, 0])

    rho_i = np.outer(psi_i, psi_i)
    rho_j = np.outer(psi_j, psi_j)

    shadows_i = [rho_i for _ in range(50)]
    shadows_j = [rho_j for _ in range(50)]

    estimator = MatrixElementEstimator(num_qubits)

    # Test overlap
    overlap_mag = estimator.estimate_overlap_magnitude(shadows_i, shadows_j)
    exact_overlap = abs(np.vdot(psi_i, psi_j))**2

    print(f"Estimated |S_ij|²: {overlap_mag:.4f}")
    print(f"Exact |S_ij|²: {exact_overlap:.4f}")
    print(f"Error: {abs(overlap_mag - exact_overlap):.2e}")

    assert abs(overlap_mag - exact_overlap) < 0.1
    print("✓ Matrix element estimator test passed")


def test_shadow_distillation():
    """Test shadow distillation"""
    print("\n" + "="*60)
    print("Testing Shadow Distillation")
    print("="*60)

    # Create a pure state with some noise
    dim = 4
    psi = np.array([1, 0, 0, 0])
    rho_pure = np.outer(psi, psi)

    # Add depolarizing noise
    epsilon = 0.1
    rho_noisy = (1 - epsilon) * rho_pure + epsilon * np.eye(dim) / dim

    distillation = ShadowDistillation(order=3)
    rho_distilled = distillation.apply_distillation(rho_noisy)

    # Check if distillation improved purity
    purity_noisy = np.trace(rho_noisy @ rho_noisy).real
    purity_distilled = np.trace(rho_distilled @ rho_distilled).real

    print(f"Noisy purity: {purity_noisy:.4f}")
    print(f"Distilled purity: {purity_distilled:.4f}")
    print(f"Improvement: {(purity_distilled - purity_noisy):.4f}")

    assert purity_distilled >= purity_noisy, "Distillation should improve purity"
    print("✓ Shadow distillation test passed")


def test_noqe_solver():
    """Test NOQE eigenvalue solver"""
    print("\n" + "="*60)
    print("Testing NOQE Solver")
    print("="*60)

    # Create simple test matrices
    M = 2
    H = np.array([[1.0, 0.1], [0.1, 2.0]])
    S = np.eye(M)

    solver = NOQESolver(verbose=False)
    eigenvalues, eigenvectors = solver.solve(H, S)

    # Check if eigenvalues are correct
    print(f"Eigenvalues: {eigenvalues}")

    # Verify solution: H·c = E·S·c
    for k in range(M):
        E_k = eigenvalues[k]
        c_k = eigenvectors[:, k]
        residual = np.linalg.norm(H @ c_k - E_k * S @ c_k)
        print(f"State {k} residual: {residual:.2e}")
        assert residual < 1e-10, "Eigenvalue equation not satisfied"

    print("✓ NOQE solver test passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("Running Shadow-NOQE Implementation Tests")
    print("="*70)

    tests = [
        test_shadow_estimator,
        test_u_statistics,
        test_reference_state_preparation,
        test_matrix_element_estimator,
        test_shadow_distillation,
        test_noqe_solver
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1

    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70)

    if failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"Some tests failed. Please review the errors above.")


if __name__ == "__main__":
    run_all_tests()
