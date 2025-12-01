"""
Analyze quantum qubits shared by pairs (i, j) in Q-SENSE basis states.

This script pre-computes all quantum state pairs and identifies:
- Unique quantum states across all matrix elements
- Unique quantum state pairs for off-diagonal elements
- Number of quantum qubits (NQ) for each pair

Usage:
    python analyze_quantum_qubits.py <molecule> <bond_length>

Example:
    python analyze_quantum_qubits.py H2O 1.0
"""

from utils_ferm import (
    orthogonal_transform_obt_tbt,
    obt_phys_spatial_to_spin,
    tbt_phys_spatial_to_spin,
    make_short_H_ferm_op
)
from utils_states import (
    convert_TZ_format_to_sparse_format,
    convert_dense_format_to_sparse_format,
    tz_state_seniority_config,
    compress_state,
)
from utils_m2_factorize import (
    get_indices_mapping_2_wvn,
    factorize_state,
    extract_quantum_states,
)
from openfermion import jordan_wigner

import numpy as np
import pickle
import sys


def main():
    # load Q-SENSE basis states
    molecule    = sys.argv[1]
    bond_length = float(sys.argv[2])

    filename = f'{molecule}_data/Uext_CSF_for_Praveen_Smik_{bond_length}.dump'

    print(f"Analyzing quantum qubits for {molecule} at bond length {bond_length}")
    print("=" * 60)

    with open(filename, 'rb') as f:
        (
            list_list_refCSF,
            list_list_Uext_mp2_CSF,
            list_list_Uext_mp2_ampld,
            list_list_Uext_opt_ampld,
            list_orb_rot,
            x_orbrot,
            Enuc,
            obt_spatial,
            tbt_spatial
        ) = pickle.load(f)

    # rotate orbitals and obtain Hamiltonian operator
    if len(list_orb_rot) != 0:
        obt, tbt = orthogonal_transform_obt_tbt(x_orbrot, list_orb_rot, obt_spatial, tbt_spatial)
    else:
        obt = obt_phys_spatial_to_spin(obt_spatial)
        tbt = tbt_phys_spatial_to_spin(tbt_spatial)

    Hfer = make_short_H_ferm_op(Enuc, obt, tbt)
    Hqub = jordan_wigner(Hfer)

    Nqubits = obt.shape[0]
    Norb = Nqubits // 2
    dim = 2 ** Nqubits

    print(f"Total qubits: {Nqubits}")
    print(f"Orbitals: {Norb}")
    print()

    # obtain relevant information about Q-SENSE states
    UCSF_tz_states = []
    CSF_tz_states = []
    W_amplitudes = []

    for i, ucsf_list in enumerate(list_list_Uext_mp2_CSF):
        for j, ucsf in enumerate(ucsf_list):
            UCSF_tz_states.append(ucsf)
            CSF_tz_states.append(list_list_refCSF[i][j])
            W_amplitudes.append(list_list_Uext_mp2_ampld[i])

    # process information so that we can taper and factorize the Q-SENSE states
    Nstates = len(UCSF_tz_states)
    print(f"Processing {Nstates} Q-SENSE basis states...")
    configs = [tz_state_seniority_config(tz_state) for tz_state in UCSF_tz_states]
    UCSF_information = [get_indices_mapping_2_wvn(CSF_tz_states[i], W_amplitudes[i], Norb) for i in range(Nstates)]

    SW_list = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'W']) for i in range(Nstates)]
    SV_list = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'V']) for i in range(Nstates)]
    SN_list = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'N']) for i in range(Nstates)]
    state_type_list = [UCSF_information[i][1] for i in range(Nstates)]

    # taper and factorize the Q-SENSE basis states
    statevectors = [convert_TZ_format_to_sparse_format(dim, tz_state) for tz_state in UCSF_tz_states]
    tapered_statevectors = [convert_dense_format_to_sparse_format(compress_state(psi.toarray()[0])) for psi in statevectors]
    factorized_tapered_statevectors = [
        factorize_state(tapered_statevectors[i], SW_list[i], SV_list[i], SN_list[i], state_type_list[i])
        for i in range(Nstates)
    ]

    # =============================================================================
    # PHASE 1: Pre-compute all quantum state pairs and identify unique states
    # =============================================================================
    print("\nPhase 1: Pre-computing quantum state pairs for all (i,j) combinations...")

    # Store pre-computed data for each matrix element
    precomputed_data = {}

    # Track unique quantum states by their content
    unique_states = {}  # key: state bytes -> value: state index
    state_index_to_key = {}  # reverse mapping: index -> key
    next_state_idx = 0

    # Track unique state pairs for off-diagonal elements
    unique_state_pairs = set()

    # Track NQ distribution
    nq_counts = {}  # NQ -> count of pairs with that NQ
    nq_zero_pairs = []  # list of (i, j) pairs with NQ=0 (fully classical)

    # Track qubits pulled into quantum region from S_V or S_N
    # For each pair (i,j), track which qubits in the quantum region come from S_V or S_N in each state
    pulled_qubits_analysis = {}  # (i,j) -> {'bra': {'from_V': set, 'from_N': set}, 'ket': {...}}

    def analyze_pulled_qubits(full_Q_block, bra_labels, ket_labels):
        """
        Analyze which qubits in the quantum region come from S_V or S_N for each state.

        Args:
            full_Q_block: tuple of qubit indices in the quantum region
            bra_labels: dict mapping orbital index -> 'W', 'V', or 'N' for bra state
            ket_labels: dict mapping orbital index -> 'W', 'V', or 'N' for ket state

        Returns:
            dict with analysis for bra and ket:
            {
                'bra': {'S_W': set, 'from_V': set, 'from_N': set},
                'ket': {'S_W': set, 'from_V': set, 'from_N': set}
            }
        """
        analysis = {
            'bra': {'S_W': set(), 'from_V': set(), 'from_N': set()},
            'ket': {'S_W': set(), 'from_V': set(), 'from_N': set()}
        }

        for q in full_Q_block:
            # Analyze bra state
            bra_type = bra_labels.get(q, 'N')
            if bra_type == 'W':
                analysis['bra']['S_W'].add(q)
            elif bra_type == 'V':
                analysis['bra']['from_V'].add(q)
            else:  # 'N'
                analysis['bra']['from_N'].add(q)

            # Analyze ket state
            ket_type = ket_labels.get(q, 'N')
            if ket_type == 'W':
                analysis['ket']['S_W'].add(q)
            elif ket_type == 'V':
                analysis['ket']['from_V'].add(q)
            else:  # 'N'
                analysis['ket']['from_N'].add(q)

        return analysis

    def register_state(state_vector):
        """Register a quantum state and return its unique index."""
        nonlocal next_state_idx
        if hasattr(state_vector, 'toarray'):
            state_vector = state_vector.toarray().flatten()
        elif len(state_vector.shape) > 1:
            state_vector = state_vector.flatten()

        key = state_vector.tobytes()
        if key not in unique_states:
            unique_states[key] = next_state_idx
            state_index_to_key[next_state_idx] = (key, state_vector)
            next_state_idx += 1
        return unique_states[key], state_vector

    # Pre-compute diagonal elements
    print("  Processing diagonal elements...")
    for i in range(Nstates):
        ket_f = factorized_tapered_statevectors[i]
        ket_labels = UCSF_information[i][0]
        ket_config = configs[i]

        ketQ, _, NQ, full_Q_block = extract_quantum_states(ket_f, ket_f, ket_labels, ket_labels)

        nq_counts[NQ] = nq_counts.get(NQ, 0) + 1

        if NQ == 0:
            precomputed_data[(i, i)] = {'NQ': 0, 'ket_config': ket_config}
            nq_zero_pairs.append((i, i))
            pulled_qubits_analysis[(i, i)] = None  # No quantum region
        else:
            ketQ_idx, ketQ_dense = register_state(ketQ)
            precomputed_data[(i, i)] = {
                'NQ': NQ,
                'full_Q_block': full_Q_block,
                'ket_config': ket_config,
                'ketQ_idx': ketQ_idx
            }
            # For diagonal, bra = ket, so analyze with same labels
            pulled_qubits_analysis[(i, i)] = analyze_pulled_qubits(full_Q_block, ket_labels, ket_labels)

    # Pre-compute off-diagonal elements
    print("  Processing off-diagonal elements...")
    for i in range(Nstates):
        for j in range(Nstates):
            if i > j:
                bra_f = factorized_tapered_statevectors[i]
                bra_labels = UCSF_information[i][0]
                bra_config = configs[i]

                ket_f = factorized_tapered_statevectors[j]
                ket_labels = UCSF_information[j][0]
                ket_config = configs[j]

                braQ, ketQ, NQ, full_Q_block = extract_quantum_states(bra_f, ket_f, bra_labels, ket_labels)

                nq_counts[NQ] = nq_counts.get(NQ, 0) + 1

                if NQ == 0:
                    precomputed_data[(i, j)] = {
                        'NQ': 0,
                        'bra_config': bra_config,
                        'ket_config': ket_config
                    }
                    nq_zero_pairs.append((i, j))
                    pulled_qubits_analysis[(i, j)] = None  # No quantum region
                else:
                    braQ_idx, braQ_dense = register_state(braQ)
                    ketQ_idx, ketQ_dense = register_state(ketQ)
                    precomputed_data[(i, j)] = {
                        'NQ': NQ,
                        'full_Q_block': full_Q_block,
                        'bra_config': bra_config,
                        'ket_config': ket_config,
                        'braQ_idx': braQ_idx,
                        'ketQ_idx': ketQ_idx
                    }
                    unique_state_pairs.add((braQ_idx, ketQ_idx))
                    # Analyze which qubits come from S_V or S_N for each state
                    pulled_qubits_analysis[(i, j)] = analyze_pulled_qubits(full_Q_block, bra_labels, ket_labels)

    # =============================================================================
    # Print Summary
    # =============================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_matrix_elements = Nstates + (Nstates * (Nstates - 1)) // 2
    print(f"\nBasis states (Nstates): {Nstates}")
    print(f"Total matrix elements: {total_matrix_elements}")
    print(f"  - Diagonal: {Nstates}")
    print(f"  - Off-diagonal (upper triangle): {(Nstates * (Nstates - 1)) // 2}")

    print(f"\nUnique quantum states: {len(unique_states)}")
    print(f"Unique quantum state pairs: {len(unique_state_pairs)}")

    print(f"\nQuantum qubit (NQ) distribution:")
    for nq in sorted(nq_counts.keys()):
        count = nq_counts[nq]
        pct = 100 * count / total_matrix_elements
        print(f"  NQ={nq}: {count} pairs ({pct:.1f}%)")

    print(f"\nFully classical pairs (NQ=0): {len(nq_zero_pairs)}")
    print(f"Pairs requiring quantum estimation: {total_matrix_elements - len(nq_zero_pairs)}")

    # Efficiency analysis
    print("\n" + "=" * 60)
    print("EFFICIENCY ANALYSIS")
    print("=" * 60)
    print(f"\nOriginal Q-SENSE: {total_matrix_elements} matrix element estimations")
    print(f"Shadow Tomography: {len(unique_states)} unique state shadow collections")
    print(f"Reduction factor: {total_matrix_elements / max(len(unique_states), 1):.1f}x fewer quantum measurements")

    # =============================================================================
    # Pulled Qubits Analysis
    # =============================================================================
    print("\n" + "=" * 60)
    print("PULLED QUBITS ANALYSIS")
    print("=" * 60)
    print("\nFor pairs with NQ > 0, analyzing which qubits in the quantum region")
    print("come from S_W, S_V, or S_N for each state (bra |i> and ket |j>).")
    print("\nWhen Q_ij = S_W(i) ∪ S_W(j), qubits in Q_ij \\ S_W(i) are 'pulled' into")
    print("the quantum region from S_V or S_N of state |i>.")

    # Count statistics
    pairs_with_pulled_from_V_bra = 0
    pairs_with_pulled_from_N_bra = 0
    pairs_with_pulled_from_V_ket = 0
    pairs_with_pulled_from_N_ket = 0

    # Detailed examples (show first few)
    examples_shown = 0
    max_examples = 5

    print(f"\n--- Example pairs with pulled qubits ---")
    for (i, j), analysis in pulled_qubits_analysis.items():
        if analysis is None:
            continue  # Skip NQ=0 pairs

        bra_from_V = analysis['bra']['from_V']
        bra_from_N = analysis['bra']['from_N']
        ket_from_V = analysis['ket']['from_V']
        ket_from_N = analysis['ket']['from_N']

        if bra_from_V:
            pairs_with_pulled_from_V_bra += 1
        if bra_from_N:
            pairs_with_pulled_from_N_bra += 1
        if ket_from_V:
            pairs_with_pulled_from_V_ket += 1
        if ket_from_N:
            pairs_with_pulled_from_N_ket += 1

        # Show examples where qubits are pulled from S_V or S_N
        has_pulled = bra_from_V or bra_from_N or ket_from_V or ket_from_N
        if has_pulled and examples_shown < max_examples:
            full_Q = precomputed_data[(i, j)]['full_Q_block']
            NQ = precomputed_data[(i, j)]['NQ']
            print(f"\nPair ({i}, {j}): NQ={NQ}, Q_block={full_Q}")
            print(f"  State |{i}> (bra): S_W={analysis['bra']['S_W']}, from_V={bra_from_V}, from_N={bra_from_N}")
            print(f"  State |{j}> (ket): S_W={analysis['ket']['S_W']}, from_V={ket_from_V}, from_N={ket_from_N}")
            examples_shown += 1

    quantum_pairs = total_matrix_elements - len(nq_zero_pairs)
    print(f"\n--- Summary of pulled qubits ---")
    print(f"Total pairs with NQ > 0: {quantum_pairs}")
    print(f"\nBra state (|i>) analysis:")
    print(f"  Pairs with qubits pulled from S_V: {pairs_with_pulled_from_V_bra} ({100*pairs_with_pulled_from_V_bra/max(quantum_pairs,1):.1f}%)")
    print(f"  Pairs with qubits pulled from S_N: {pairs_with_pulled_from_N_bra} ({100*pairs_with_pulled_from_N_bra/max(quantum_pairs,1):.1f}%)")
    print(f"\nKet state (|j>) analysis:")
    print(f"  Pairs with qubits pulled from S_V: {pairs_with_pulled_from_V_ket} ({100*pairs_with_pulled_from_V_ket/max(quantum_pairs,1):.1f}%)")
    print(f"  Pairs with qubits pulled from S_N: {pairs_with_pulled_from_N_ket} ({100*pairs_with_pulled_from_N_ket/max(quantum_pairs,1):.1f}%)")


if __name__ == '__main__':
    main()
