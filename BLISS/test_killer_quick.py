"""
Quick test to verify the fixed killer optimizer works.
"""

import numpy as np
from openfermion import (
    FermionOperator,
    jordan_wigner,
    normal_ordered,
)
from single_fermion_killer import (
    qubit_operator_1norm,
    apply_single_fermion_killer,
    apply_hermitian_fermion_killer,
)


def test_simple_hamiltonian():
    """Test on H = a_0† a_1 which should be perfectly cancellable."""
    print("=" * 60)
    print("TEST 1: H = a_0† a_1 (non-Hermitian, should be perfectly cancellable)")
    print("=" * 60)

    n = 4

    H_ferm = FermionOperator('0^ 1', 1.0)
    H_ferm = normal_ordered(H_ferm)
    H_qubit = jordan_wigner(H_ferm)

    original_norm = qubit_operator_1norm(H_qubit)
    print(f"Original 1-norm: {original_norm:.6f}")
    print()

    # Test with annihilation on orbital 1 (should cancel H = a_0† a_1)
    print("Applying NON-Hermitian killer with a_1 (annihilation):")
    H_reduced, O_opt, killer, reduction = apply_single_fermion_killer(
        H_qubit,
        spin_orbital=1,
        create=False,  # annihilation a_1
        n_spin_orbitals=n,
        verbose=True,
        n_restarts=3,
        use_two_body=False  # Only one-body for speed
    )

    print(f"\nFinal reduction: {reduction:.6f}")
    print(f"Expected: ~{original_norm:.6f} (perfect cancellation)")
    print(f"Success: {reduction > 0.9 * original_norm}")


def test_hermitian_hamiltonian_non_hermitian_killer():
    """Test non-Hermitian killer on H = a_0† a_1 + a_1† a_0 (Hermitian hopping term)."""
    print("\n" + "=" * 60)
    print("TEST 2: H = a_0† a_1 + a_1† a_0 with NON-Hermitian killer")
    print("=" * 60)

    n = 4

    H_ferm = FermionOperator('0^ 1', 1.0) + FermionOperator('1^ 0', 1.0)
    H_ferm = normal_ordered(H_ferm)
    H_qubit = jordan_wigner(H_ferm)

    original_norm = qubit_operator_1norm(H_qubit)
    print(f"Original 1-norm: {original_norm:.6f}")
    print()

    # Try non-Hermitian killer (should fail to reduce)
    print("Applying NON-Hermitian killer with a_1 (annihilation):")
    H_reduced, O_opt, killer, reduction = apply_single_fermion_killer(
        H_qubit,
        spin_orbital=1,
        create=False,
        n_spin_orbitals=n,
        verbose=True,
        n_restarts=3,
        use_two_body=False
    )

    print(f"\nReduction achieved: {reduction:.6f}")
    print(f"Expected: 0.0 (can't reduce Hermitian H with non-Hermitian killer)")


def test_hermitian_hamiltonian_hermitian_killer():
    """Test Hermitian killer on H = a_0† a_1 + a_1† a_0 (Hermitian hopping term)."""
    print("\n" + "=" * 60)
    print("TEST 3: H = a_0† a_1 + a_1† a_0 with HERMITIAN killer (K + K†)")
    print("=" * 60)

    n = 4

    H_ferm = FermionOperator('0^ 1', 1.0) + FermionOperator('1^ 0', 1.0)
    H_ferm = normal_ordered(H_ferm)
    H_qubit = jordan_wigner(H_ferm)

    original_norm = qubit_operator_1norm(H_qubit)
    print(f"Original 1-norm: {original_norm:.6f}")
    print()

    # Try Hermitian killer (should succeed!)
    print("Applying HERMITIAN killer (K + K†) on orbital 1:")
    H_reduced, O_opt, killer, reduction = apply_hermitian_fermion_killer(
        H_qubit,
        spin_orbital=1,
        n_spin_orbitals=n,
        verbose=True,
        n_restarts=3,
        use_two_body=False
    )

    print(f"\nReduction achieved: {reduction:.6f}")
    print(f"Expected: ~{original_norm:.6f} (perfect cancellation)")
    print(f"Success: {reduction > 0.9 * original_norm}")


if __name__ == '__main__':
    test_simple_hamiltonian()
    test_hermitian_hamiltonian_non_hermitian_killer()
    test_hermitian_hamiltonian_hermitian_killer()
