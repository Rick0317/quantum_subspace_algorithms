"""
Single Fermion Killer for Hamiltonian 1-Norm Reduction.

This module implements a method to reduce the 1-norm of a qubit Hamiltonian
by subtracting terms of the form O·a_i or O·a†_i, where O is parametrized
as a fermionic operator to ensure physical consistency.

The killer terms have the form:
- O·a_i where O = sum_j c_j a_j† gives one-body terms: a_j† a_i
- O·a_i where O = sum_{jkl} c_{jkl} a_k† a_l a_j† gives two-body terms: a_k† a_l a_j† a_i

The 1-norm is defined as the sum of absolute values of Pauli coefficients
after Jordan-Wigner transformation.
"""

import numpy as np
from openfermion import (
    FermionOperator,
    QubitOperator,
    jordan_wigner,
    hermitian_conjugated,
    normal_ordered,
)
from scipy.optimize import minimize


def qubit_operator_1norm(H: QubitOperator) -> float:
    """
    Compute the 1-norm of a QubitOperator.

    The 1-norm is the sum of absolute values of all Pauli term coefficients.

    Args:
        H: QubitOperator

    Returns:
        float: The 1-norm of H
    """
    return sum(abs(coef) for coef in H.terms.values())


def get_fermionic_ladder_operator(spin_orbital: int, create: bool) -> FermionOperator:
    """
    Get the fermionic creation or annihilation operator for a spin-orbital.

    Args:
        spin_orbital: Index of the spin-orbital
        create: If True, return a†_i; if False, return a_i

    Returns:
        FermionOperator: The ladder operator
    """
    action = 1 if create else 0
    return FermionOperator(((spin_orbital, action),), 1.0)


def construct_one_body_O(coefficients: np.ndarray, n_spin_orbitals: int,
                          target_orbital: int, create: bool) -> FermionOperator:
    """
    Construct O such that O·a_i (or O·a†_i) gives one-body terms.

    For annihilation (create=False):
        O = sum_j c_j a_j†  =>  O·a_i = sum_j c_j a_j† a_i

    For creation (create=True):
        O = sum_j c_j a_j   =>  O·a†_i = sum_j c_j a_j a†_i

    Args:
        coefficients: Array of shape (n_spin_orbitals,) for one-body coefficients
        n_spin_orbitals: Number of spin-orbitals
        target_orbital: The orbital i in a_i or a†_i
        create: Whether the target is creation (True) or annihilation (False)

    Returns:
        FermionOperator: The operator O
    """
    O = FermionOperator()

    for j in range(n_spin_orbitals):
        if not np.isclose(coefficients[j], 0.0):
            if create:
                # O·a†_i, so O should have annihilation to pair with creation
                O += FermionOperator(((j, 0),), coefficients[j])  # a_j
            else:
                # O·a_i, so O should have creation to pair with annihilation
                O += FermionOperator(((j, 1),), coefficients[j])  # a_j†

    return O


def construct_two_body_O(coefficients: np.ndarray, n_spin_orbitals: int,
                          target_orbital: int, create: bool) -> FermionOperator:
    """
    Construct O such that O·a_i (or O·a†_i) gives two-body terms.

    For annihilation (create=False):
        O = sum_{jkl} c_{jkl} a_k† a_l a_j†  =>  O·a_i = sum_{jkl} c_{jkl} a_k† a_l a_j† a_i

    For creation (create=True):
        O = sum_{jkl} c_{jkl} a_k† a_l a_j   =>  O·a†_i = sum_{jkl} c_{jkl} a_k† a_l a_j a†_i

    Args:
        coefficients: Array of shape (n_spin_orbitals^3,) flattened from (n, n, n)
        n_spin_orbitals: Number of spin-orbitals
        target_orbital: The orbital i in a_i or a†_i
        create: Whether the target is creation (True) or annihilation (False)

    Returns:
        FermionOperator: The operator O
    """
    O = FermionOperator()
    n = n_spin_orbitals

    # Reshape coefficients to 3D
    c = coefficients.reshape((n, n, n))

    for k in range(n):
        for l in range(n):
            for j in range(n):
                if not np.isclose(c[k, l, j], 0.0):
                    if create:
                        # O·a†_i: O = a_k† a_l a_j
                        term = ((k, 1), (l, 0), (j, 0))
                    else:
                        # O·a_i: O = a_k† a_l a_j†
                        term = ((k, 1), (l, 0), (j, 1))
                    O += FermionOperator(term, c[k, l, j])

    return O


def construct_combined_O(coefficients: np.ndarray, n_spin_orbitals: int,
                         target_orbital: int, create: bool) -> FermionOperator:
    """
    Construct O with both one-body and two-body terms.

    Total parameters: n + n^3 where n = n_spin_orbitals
    - First n coefficients: one-body terms
    - Remaining n^3 coefficients: two-body terms

    Args:
        coefficients: Combined coefficient array
        n_spin_orbitals: Number of spin-orbitals
        target_orbital: The orbital i
        create: Whether target is creation or annihilation

    Returns:
        FermionOperator: The combined operator O
    """
    n = n_spin_orbitals
    one_body_coeffs = coefficients[:n]
    two_body_coeffs = coefficients[n:]

    O_one = construct_one_body_O(one_body_coeffs, n, target_orbital, create)
    O_two = construct_two_body_O(two_body_coeffs, n, target_orbital, create)

    return O_one + O_two


def apply_single_fermion_killer(
    H_qubit: QubitOperator,
    spin_orbital: int,
    create: bool,
    n_spin_orbitals: int,
    verbose: bool = False,
    n_restarts: int = 5,
    use_two_body: bool = True
) -> tuple:
    """
    Find optimal O to minimize 1-norm of H - O·a_i (or H - O·a†_i).

    The killer operator K = O·a_i is constructed from fermionic operators:
    - One-body: O = sum_j c_j a_j†, giving K = sum_j c_j a_j† a_i
    - Two-body: O = sum_{jkl} c_{jkl} a_k† a_l a_j†, giving K = sum_{jkl} c_{jkl} a_k† a_l a_j† a_i

    Uses multiple random restarts with Powell optimizer to escape local minima.

    Args:
        H_qubit: The Hamiltonian as a QubitOperator
        spin_orbital: Index of the spin-orbital i
        create: If True, use a†_i; if False, use a_i
        n_spin_orbitals: Total number of spin-orbitals (= n_qubits)
        verbose: If True, print optimization progress
        n_restarts: Number of random restarts for optimization
        use_two_body: If True, include two-body terms in O (more params but more expressive)

    Returns:
        tuple: (H_reduced, O_optimal, killer_operator, norm_reduction)
            - H_reduced: H - K with optimal O (as QubitOperator)
            - O_optimal: The optimal operator O (as FermionOperator)
            - killer_operator: The full killer K = O·a_i or O·a†_i (as QubitOperator)
            - norm_reduction: Amount by which 1-norm was reduced
    """
    n = n_spin_orbitals
    original_norm = qubit_operator_1norm(H_qubit)

    # Get the ladder operator
    ladder_ferm = get_fermionic_ladder_operator(spin_orbital, create)

    if verbose:
        action_str = "creation (a†)" if create else "annihilation (a)"
        print(f"Applying single fermion killer with {action_str} on spin-orbital {spin_orbital}")
        print(f"Original 1-norm: {original_norm:.6f}")

    # Determine number of parameters
    if use_two_body:
        n_params = n + n ** 3
    else:
        n_params = n

    def build_O(coeffs):
        if use_two_body:
            return construct_combined_O(coeffs, n, spin_orbital, create)
        else:
            # Only one-body: pad with zeros for two-body
            full_coeffs = np.zeros(n + n ** 3)
            full_coeffs[:n] = coeffs
            return construct_combined_O(full_coeffs, n, spin_orbital, create)

    if verbose:
        print(f"Number of parameters: {n_params}")

    def objective(coefficients):
        O_ferm = build_O(coefficients)
        killer_ferm = O_ferm * ladder_ferm
        killer_ferm = normal_ordered(killer_ferm)
        killer_qubit = jordan_wigner(killer_ferm)
        H_reduced = H_qubit - killer_qubit
        return qubit_operator_1norm(H_reduced)

    # Use multiple random restarts to find global minimum
    best_result = None
    best_fun = original_norm

    for restart in range(n_restarts):
        # Random initial guess with scale based on typical coefficient magnitudes
        x0 = np.random.randn(n_params) * 0.5

        # Use Powell method (gradient-free, works well for non-smooth objectives)
        result = minimize(
            objective,
            x0,
            method='Powell',
            options={'maxiter': 2000, 'ftol': 1e-10, 'xtol': 1e-10}
        )

        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result

        if verbose:
            print(f"  Restart {restart+1}/{n_restarts}: 1-norm = {result.fun:.6f}")

    # If no improvement found, return identity
    if best_result is None:
        if verbose:
            print(f"No improvement found")
        return H_qubit, FermionOperator(), QubitOperator(), 0.0

    if verbose:
        print(f"Best 1-norm: {best_fun:.6f}")

    # Construct the optimal operators
    O_optimal = build_O(best_result.x)
    killer_ferm = O_optimal * ladder_ferm
    killer_ferm = normal_ordered(killer_ferm)
    killer_qubit = jordan_wigner(killer_ferm)
    H_reduced = H_qubit - killer_qubit

    norm_reduction = original_norm - qubit_operator_1norm(H_reduced)

    if verbose:
        print(f"1-norm reduction: {norm_reduction:.6f}")
        if norm_reduction > 0:
            print(f"Reduction percentage: {100 * norm_reduction / original_norm:.2f}%")

    return H_reduced, O_optimal, killer_qubit, norm_reduction


def apply_hermitian_fermion_killer(
    H_qubit: QubitOperator,
    spin_orbital: int,
    n_spin_orbitals: int,
    verbose: bool = False,
    n_restarts: int = 5,
    use_two_body: bool = False
) -> tuple:
    """
    Find optimal O to minimize 1-norm of H - (K + K†) where K = O·a_i.

    This version preserves Hermiticity by subtracting both K and its Hermitian
    conjugate. This is necessary for reducing Hermitian Hamiltonians where
    terms come in conjugate pairs.

    The killer operator K = O·a_i is constructed from fermionic operators:
    - One-body: O = sum_j c_j a_j†, giving K = sum_j c_j a_j† a_i
    - Two-body: O = sum_{jkl} c_{jkl} a_k† a_l a_j†

    We subtract K + K† = O·a_i + a_i†·O† to maintain Hermiticity.

    Args:
        H_qubit: The Hamiltonian as a QubitOperator (should be Hermitian)
        spin_orbital: Index of the spin-orbital i
        n_spin_orbitals: Total number of spin-orbitals (= n_qubits)
        verbose: If True, print optimization progress
        n_restarts: Number of random restarts for optimization
        use_two_body: If True, include two-body terms in O

    Returns:
        tuple: (H_reduced, O_optimal, killer_operator, norm_reduction)
            - H_reduced: H - (K + K†) with optimal O (as QubitOperator)
            - O_optimal: The optimal operator O (as FermionOperator)
            - killer_operator: The full Hermitian killer K + K† (as QubitOperator)
            - norm_reduction: Amount by which 1-norm was reduced
    """
    n = n_spin_orbitals
    original_norm = qubit_operator_1norm(H_qubit)

    # Get the annihilation operator a_i
    ladder_ferm = get_fermionic_ladder_operator(spin_orbital, create=False)

    if verbose:
        print(f"Applying Hermitian fermion killer on spin-orbital {spin_orbital}")
        print(f"Original 1-norm: {original_norm:.6f}")

    # Determine number of parameters
    if use_two_body:
        n_params = n + n ** 3
    else:
        n_params = n

    def build_O(coeffs):
        if use_two_body:
            return construct_combined_O(coeffs, n, spin_orbital, create=False)
        else:
            full_coeffs = np.zeros(n + n ** 3)
            full_coeffs[:n] = coeffs
            return construct_combined_O(full_coeffs, n, spin_orbital, create=False)

    if verbose:
        print(f"Number of parameters: {n_params}")

    def objective(coefficients):
        O_ferm = build_O(coefficients)
        # K = O · a_i
        killer_ferm = O_ferm * ladder_ferm
        killer_ferm = normal_ordered(killer_ferm)
        # K† = a_i† · O†
        killer_ferm_dag = hermitian_conjugated(killer_ferm)
        # Hermitian killer: K + K†
        hermitian_killer_ferm = killer_ferm + killer_ferm_dag
        hermitian_killer_qubit = jordan_wigner(hermitian_killer_ferm)
        H_reduced = H_qubit - hermitian_killer_qubit
        return qubit_operator_1norm(H_reduced)

    # Use multiple random restarts to find global minimum
    best_result = None
    best_fun = original_norm

    for restart in range(n_restarts):
        x0 = np.random.randn(n_params) * 0.5

        result = minimize(
            objective,
            x0,
            method='Powell',
            options={'maxiter': 2000, 'ftol': 1e-10, 'xtol': 1e-10}
        )

        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result

        if verbose:
            print(f"  Restart {restart+1}/{n_restarts}: 1-norm = {result.fun:.6f}")

    # If no improvement found, return identity
    if best_result is None:
        if verbose:
            print(f"No improvement found")
        return H_qubit, FermionOperator(), QubitOperator(), 0.0

    if verbose:
        print(f"Best 1-norm: {best_fun:.6f}")

    # Construct the optimal operators
    O_optimal = build_O(best_result.x)
    killer_ferm = O_optimal * ladder_ferm
    killer_ferm = normal_ordered(killer_ferm)
    killer_ferm_dag = hermitian_conjugated(killer_ferm)
    hermitian_killer_ferm = killer_ferm + killer_ferm_dag
    hermitian_killer_qubit = jordan_wigner(hermitian_killer_ferm)
    H_reduced = H_qubit - hermitian_killer_qubit

    norm_reduction = original_norm - qubit_operator_1norm(H_reduced)

    if verbose:
        print(f"1-norm reduction: {norm_reduction:.6f}")
        if norm_reduction > 0:
            print(f"Reduction percentage: {100 * norm_reduction / original_norm:.2f}%")

    return H_reduced, O_optimal, hermitian_killer_qubit, norm_reduction


def find_best_single_fermion_killer(
    H_qubit: QubitOperator,
    n_spin_orbitals: int,
    O_type: str = 'one_body',
    verbose: bool = False
) -> tuple:
    """
    Find the best single fermion killer across all spin-orbitals and creation/annihilation.

    Tries all spin-orbitals with both creation and annihilation operators,
    and returns the one that gives the maximum 1-norm reduction.

    Args:
        H_qubit: The Hamiltonian as a QubitOperator
        n_spin_orbitals: Total number of spin-orbitals
        O_type: Type of O operator ('one_body', 'two_body', or 'combined')
        verbose: If True, print progress

    Returns:
        tuple: (H_reduced, O_optimal, killer_operator, norm_reduction, best_orbital, best_create)
    """
    best_reduction = -np.inf
    best_result = None
    best_orbital = None
    best_create = None

    for spin_orbital in range(n_spin_orbitals):
        for create in [True, False]:
            H_reduced, O_opt, killer, reduction = apply_single_fermion_killer(
                H_qubit, spin_orbital, create, n_spin_orbitals,
                O_type=O_type,
                verbose=False
            )

            if verbose:
                action = "a†" if create else "a"
                print(f"  {action}_{spin_orbital}: reduction = {reduction:.6f}")

            if reduction > best_reduction:
                best_reduction = reduction
                best_result = (H_reduced, O_opt, killer, reduction)
                best_orbital = spin_orbital
                best_create = create

    if verbose:
        action = "a†" if best_create else "a"
        print(f"\nBest killer: O·{action}_{best_orbital} with reduction {best_reduction:.6f}")

    return (*best_result, best_orbital, best_create)


def iterative_fermion_killer(
    H_qubit: QubitOperator,
    n_spin_orbitals: int,
    max_iterations: int = 10,
    min_reduction: float = 1e-6,
    O_type: str = 'one_body',
    verbose: bool = False
) -> tuple:
    """
    Iteratively apply single fermion killers until convergence.

    Args:
        H_qubit: The Hamiltonian as a QubitOperator
        n_spin_orbitals: Total number of spin-orbitals
        max_iterations: Maximum number of killer applications
        min_reduction: Stop if reduction falls below this threshold
        O_type: Type of O operator
        verbose: If True, print progress

    Returns:
        tuple: (H_final, total_reduction, killers_applied)
            - H_final: The reduced Hamiltonian
            - total_reduction: Total 1-norm reduction achieved
            - killers_applied: List of (orbital, create, O, reduction) for each step
    """
    H_current = H_qubit
    original_norm = qubit_operator_1norm(H_qubit)
    total_reduction = 0.0
    killers_applied = []

    if verbose:
        print(f"Starting iterative fermion killer")
        print(f"Original 1-norm: {original_norm:.6f}")
        print("=" * 50)

    for iteration in range(max_iterations):
        result = find_best_single_fermion_killer(
            H_current, n_spin_orbitals,
            O_type=O_type,
            verbose=False
        )
        H_reduced, O_opt, killer, reduction, orbital, create = result

        if reduction < min_reduction:
            if verbose:
                print(f"\nIteration {iteration + 1}: reduction {reduction:.6f} below threshold, stopping")
            break

        H_current = H_reduced
        total_reduction += reduction
        killers_applied.append((orbital, create, O_opt, reduction))

        if verbose:
            action = "a†" if create else "a"
            current_norm = qubit_operator_1norm(H_current)
            print(f"Iteration {iteration + 1}: O·{action}_{orbital}, "
                  f"reduction={reduction:.6f}, current 1-norm={current_norm:.6f}")

    if verbose:
        final_norm = qubit_operator_1norm(H_current)
        print("=" * 50)
        print(f"Final 1-norm: {final_norm:.6f}")
        print(f"Total reduction: {total_reduction:.6f} ({100*total_reduction/original_norm:.1f}%)")

    return H_current, total_reduction, killers_applied
