"""
Test script to understand why optimization fails for single fermion killer.
"""

import numpy as np
from openfermion import (
    FermionOperator,
    QubitOperator,
    jordan_wigner,
    normal_ordered,
)
from scipy.optimize import minimize, differential_evolution, basinhopping
from single_fermion_killer import (
    qubit_operator_1norm,
    construct_combined_O,
    get_fermionic_ladder_operator,
)


def test_gradient_at_zero():
    """
    Check numerical gradient at x=0.
    """
    print("=" * 60)
    print("TEST: Numerical gradient at x=0")
    print("=" * 60)

    n = 4

    # H = a_0† a_1
    H_ferm = FermionOperator('0^ 1', 1.0)
    H_ferm = normal_ordered(H_ferm)
    H_qubit = jordan_wigner(H_ferm)

    target_orbital = 1
    create = False

    def objective(coeffs):
        O = construct_combined_O(coeffs, n, target_orbital, create)
        ladder = get_fermionic_ladder_operator(target_orbital, create)
        killer_ferm = O * ladder
        killer_ferm = normal_ordered(killer_ferm)
        killer_qubit = jordan_wigner(killer_ferm)
        H_reduced = H_qubit - killer_qubit
        return qubit_operator_1norm(H_reduced)

    n_params = n + n**3
    x0 = np.zeros(n_params)

    # Compute numerical gradient at x=0
    eps = 1e-5
    grad = np.zeros(n_params)

    f0 = objective(x0)
    print(f"f(0) = {f0}")

    for i in range(min(10, n_params)):  # First 10 parameters
        x_plus = x0.copy()
        x_plus[i] = eps
        f_plus = objective(x_plus)

        x_minus = x0.copy()
        x_minus[i] = -eps
        f_minus = objective(x_minus)

        grad[i] = (f_plus - f_minus) / (2 * eps)
        print(f"  grad[{i}] = {grad[i]:+.6f}  (f+ = {f_plus:.6f}, f- = {f_minus:.6f})")

    print(f"\nOptimal is at c_0 = 1.0, but grad[0] points away from it.")
    print(f"The issue: at x=0, gradient points in wrong direction!")


def test_different_optimizers():
    """
    Try different optimization methods.
    """
    print("=" * 60)
    print("TEST: Different optimizers")
    print("=" * 60)

    n = 4

    H_ferm = FermionOperator('0^ 1', 1.0)
    H_ferm = normal_ordered(H_ferm)
    H_qubit = jordan_wigner(H_ferm)

    target_orbital = 1
    create = False

    def objective(coeffs):
        O = construct_combined_O(coeffs, n, target_orbital, create)
        ladder = get_fermionic_ladder_operator(target_orbital, create)
        killer_ferm = O * ladder
        killer_ferm = normal_ordered(killer_ferm)
        killer_qubit = jordan_wigner(killer_ferm)
        H_reduced = H_qubit - killer_qubit
        return qubit_operator_1norm(H_reduced)

    n_params = n + n**3
    original_norm = qubit_operator_1norm(H_qubit)
    print(f"Original 1-norm: {original_norm}")

    # Try with non-zero initial guess
    print("\n1. L-BFGS-B with random initial guess:")
    for trial in range(3):
        x0 = np.random.randn(n_params) * 0.5
        result = minimize(objective, x0, method='L-BFGS-B',
                         options={'maxiter': 1000})
        print(f"   Trial {trial+1}: final = {result.fun:.6f}, x[0] = {result.x[0]:.4f}")

    # Try Nelder-Mead (gradient-free)
    print("\n2. Nelder-Mead (gradient-free):")
    x0 = np.random.randn(n_params) * 0.5
    result = minimize(objective, x0, method='Nelder-Mead',
                     options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    print(f"   Final: {result.fun:.6f}, x[0] = {result.x[0]:.4f}")

    # Try COBYLA (gradient-free)
    print("\n3. COBYLA (gradient-free):")
    x0 = np.random.randn(n_params) * 0.5
    result = minimize(objective, x0, method='COBYLA',
                     options={'maxiter': 5000, 'rhobeg': 1.0})
    print(f"   Final: {result.fun:.6f}, x[0] = {result.x[0]:.4f}")

    # Try Powell (gradient-free)
    print("\n4. Powell (gradient-free):")
    x0 = np.random.randn(n_params) * 0.5
    result = minimize(objective, x0, method='Powell',
                     options={'maxiter': 5000})
    print(f"   Final: {result.fun:.6f}, x[0] = {result.x[0]:.4f}")

    # Try differential evolution (global)
    print("\n5. Differential Evolution (global):")
    bounds = [(-3, 3)] * n_params
    result = differential_evolution(objective, bounds, maxiter=500, seed=42,
                                   workers=-1, updating='deferred')
    print(f"   Final: {result.fun:.6f}, x[0] = {result.x[0]:.4f}")


def test_one_body_only():
    """
    Test with only one-body parameters (much smaller search space).
    """
    print("=" * 60)
    print("TEST: One-body only (smaller search space)")
    print("=" * 60)

    n = 4

    H_ferm = FermionOperator('0^ 1', 1.0)
    H_ferm = normal_ordered(H_ferm)
    H_qubit = jordan_wigner(H_ferm)

    target_orbital = 1
    create = False

    def objective_one_body(coeffs_1body):
        """Only use one-body coefficients."""
        full_coeffs = np.zeros(n + n**3)
        full_coeffs[:n] = coeffs_1body

        O = construct_combined_O(full_coeffs, n, target_orbital, create)
        ladder = get_fermionic_ladder_operator(target_orbital, create)
        killer_ferm = O * ladder
        killer_ferm = normal_ordered(killer_ferm)
        killer_qubit = jordan_wigner(killer_ferm)
        H_reduced = H_qubit - killer_qubit
        return qubit_operator_1norm(H_reduced)

    print(f"Original 1-norm: {qubit_operator_1norm(H_qubit)}")
    print(f"One-body params: {n}")

    # Try multiple starting points
    print("\nTrying different starting points:")
    best_result = None
    for trial in range(10):
        x0 = np.random.randn(n) * 2
        result = minimize(objective_one_body, x0, method='Nelder-Mead',
                         options={'maxiter': 2000})
        if best_result is None or result.fun < best_result.fun:
            best_result = result
        print(f"  Trial {trial+1}: final = {result.fun:.6f}, coeffs = {result.x}")

    print(f"\nBest result: {best_result.fun:.6f}")
    print(f"Best coeffs: {best_result.x}")


if __name__ == '__main__':
    test_gradient_at_zero()
    print("\n" * 2)
    test_different_optimizers()
    print("\n" * 2)
    test_one_body_only()
