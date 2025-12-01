"""
Test script to debug single fermion killer.
"""

import numpy as np
from openfermion import (
    FermionOperator,
    QubitOperator,
    jordan_wigner,
    hermitian_conjugated,
    normal_ordered,
)
from single_fermion_killer import (
    qubit_operator_1norm,
    construct_one_body_O,
    construct_two_body_O,
    construct_combined_O,
    get_fermionic_ladder_operator,
    apply_single_fermion_killer,
)


def test_basic_killer():
    """Test the killer on a simple Hamiltonian."""

    # Create a simple 4-qubit (2 spatial orbital) Hamiltonian
    # H = a_0† a_1 + a_1† a_0 (hopping term)
    n_spin_orbitals = 4

    H_ferm = FermionOperator('0^ 1', 1.0) + FermionOperator('1^ 0', 1.0)
    H_ferm += FermionOperator('0^ 2', 0.5) + FermionOperator('2^ 0', 0.5)
    H_ferm = normal_ordered(H_ferm)
    H_qubit = jordan_wigner(H_ferm)

    print("=" * 60)
    print("TEST 1: Simple Hamiltonian")
    print("=" * 60)
    print(f"H_ferm = {H_ferm}")
    print(f"H_qubit = {H_qubit}")
    print(f"H_qubit 1-norm = {qubit_operator_1norm(H_qubit):.6f}")
    print()

    # Test what a killer looks like
    print("Testing killer construction:")
    print("-" * 40)

    # Try a simple one-body killer: O = c * a_0†, then O * a_1 = c * a_0† a_1
    coeffs_1body = np.array([1.0, 0.0, 0.0, 0.0])  # Only c_0 = 1
    O_1body = construct_one_body_O(coeffs_1body, n_spin_orbitals, target_orbital=1, create=False)
    print(f"One-body O (for a_1): {O_1body}")

    ladder = get_fermionic_ladder_operator(1, create=False)  # a_1
    print(f"Ladder operator a_1: {ladder}")

    killer_ferm = O_1body * ladder
    print(f"Killer (before normal order): {killer_ferm}")

    killer_ferm = normal_ordered(killer_ferm)
    print(f"Killer (after normal order): {killer_ferm}")

    killer_qubit = jordan_wigner(killer_ferm)
    print(f"Killer (qubit): {killer_qubit}")
    print(f"Killer qubit 1-norm: {qubit_operator_1norm(killer_qubit):.6f}")
    print()

    # Check if killer has any overlap with H
    H_minus_killer = H_qubit - killer_qubit
    print(f"H - killer 1-norm: {qubit_operator_1norm(H_minus_killer):.6f}")
    print(f"Original H 1-norm: {qubit_operator_1norm(H_qubit):.6f}")
    print()

    # Now try the optimization
    print("Running apply_single_fermion_killer:")
    print("-" * 40)
    H_reduced, O_opt, killer_qubit, reduction = apply_single_fermion_killer(
        H_qubit,
        spin_orbital=1,
        create=False,    # annihilation a_1
        n_spin_orbitals=n_spin_orbitals,
        verbose=True
    )

    print(f"\nOptimal O: {O_opt}")
    print(f"Optimal killer: {killer_qubit}")
    print()


def test_what_can_be_cancelled():
    """
    Test: Given H with terms like a_j† a_i, can we cancel them with killer O·a_i?
    """
    print("=" * 60)
    print("TEST 2: What can killer O·a_i cancel?")
    print("=" * 60)

    n = 4

    # H contains a_0† a_1 term
    H_ferm = FermionOperator('0^ 1', 1.0)  # a_0† a_1
    H_ferm = normal_ordered(H_ferm)
    H_qubit = jordan_wigner(H_ferm)

    print(f"H_ferm = {H_ferm}")
    print(f"H_qubit = {H_qubit}")
    print(f"H 1-norm = {qubit_operator_1norm(H_qubit):.6f}")
    print()

    # Killer O·a_1 where O = c_0 a_0†
    # Then K = c_0 * a_0† * a_1 = c_0 * a_0† a_1 (should match H if c_0 = 1)
    print("Manual killer: O = a_0†, K = a_0† a_1")
    O_manual = FermionOperator('0^', 1.0)
    ladder = FermionOperator('1', 1.0)  # a_1
    K_manual = O_manual * ladder
    K_manual = normal_ordered(K_manual)
    print(f"K_ferm = {K_manual}")

    K_qubit = jordan_wigner(K_manual)
    print(f"K_qubit = {K_qubit}")

    H_minus_K = H_qubit - K_qubit
    print(f"H - K = {H_minus_K}")
    print(f"|H - K| = {qubit_operator_1norm(H_minus_K):.6f}")
    print()

    # Now test optimization
    print("Running optimization for spin_orbital=1, create=False (a_1):")
    H_reduced, O_opt, killer_qubit, reduction = apply_single_fermion_killer(
        H_qubit,
        spin_orbital=1,
        create=False,
        n_spin_orbitals=n,
        verbose=True
    )
    print(f"Optimized O: {O_opt}")
    print()


def test_creation_vs_annihilation():
    """
    Test whether we need creation or annihilation for different terms.
    """
    print("=" * 60)
    print("TEST 3: Creation vs Annihilation")
    print("=" * 60)

    n = 4

    # H = a_0† a_1 + h.c.
    H_ferm = FermionOperator('0^ 1', 1.0) + FermionOperator('1^ 0', 1.0)
    H_ferm = normal_ordered(H_ferm)
    H_qubit = jordan_wigner(H_ferm)

    print(f"H_ferm = {H_ferm}")
    print(f"H 1-norm = {qubit_operator_1norm(H_qubit):.6f}")
    print()

    # Try annihilation on qubit 0
    print("Try killer with a_0 (annihilation):")
    H_reduced, O_opt, killer, reduction = apply_single_fermion_killer(
        H_qubit, spin_orbital=0, create=False, n_spin_orbitals=n, verbose=True
    )
    print()

    # Try creation on qubit 0
    print("Try killer with a†_0 (creation):")
    H_reduced, O_opt, killer, reduction = apply_single_fermion_killer(
        H_qubit, spin_orbital=0, create=True, n_spin_orbitals=n, verbose=True
    )
    print()

    # Try annihilation on qubit 1
    print("Try killer with a_1 (annihilation):")
    H_reduced, O_opt, killer, reduction = apply_single_fermion_killer(
        H_qubit, spin_orbital=1, create=False, n_spin_orbitals=n, verbose=True
    )
    print()

    # Try creation on qubit 1
    print("Try killer with a†_1 (creation):")
    H_reduced, O_opt, killer, reduction = apply_single_fermion_killer(
        H_qubit, spin_orbital=1, create=True, n_spin_orbitals=n, verbose=True
    )
    print()


def test_examine_optimization_landscape():
    """
    Check if the optimization is working by examining the landscape.
    """
    print("=" * 60)
    print("TEST 4: Examine optimization landscape")
    print("=" * 60)

    n = 4

    # Simple H = a_0† a_1
    H_ferm = FermionOperator('0^ 1', 1.0)
    H_ferm = normal_ordered(H_ferm)
    H_qubit = jordan_wigner(H_ferm)

    original_norm = qubit_operator_1norm(H_qubit)
    print(f"H 1-norm = {original_norm:.6f}")
    print()

    # Manually sweep coefficient c_0 in one-body term
    print("Sweeping c_0 coefficient for O = c_0 * a_0†, K = O·a_1:")
    target_orbital = 1
    create = False  # a_1

    for c0 in np.linspace(-2, 2, 21):
        coeffs = np.zeros(n + n**3)
        coeffs[0] = c0  # Set c_0 for a_0†

        O = construct_combined_O(coeffs, n, target_orbital, create)
        ladder = get_fermionic_ladder_operator(target_orbital, create)
        killer_ferm = O * ladder
        killer_ferm = normal_ordered(killer_ferm)
        killer_qubit = jordan_wigner(killer_ferm)

        H_reduced = H_qubit - killer_qubit
        reduced_norm = qubit_operator_1norm(H_reduced)

        print(f"  c_0 = {c0:6.2f}: |H - K| = {reduced_norm:.6f}")


if __name__ == '__main__':
    test_basic_killer()
    print("\n" * 2)
    test_what_can_be_cancelled()
    print("\n" * 2)
    test_creation_vs_annihilation()
    print("\n" * 2)
    test_examine_optimization_landscape()
