
from utils_basic import (
    copy_hamiltonian
)
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
    decompress_state,
    create_composite_state
)
from utils_m1_seniority import (
    project_out_seniority_symmetries
)
from utils_m2_factorize import (
    expand_tensor_product,
    expand_tensor_product_for_incomplete_qubit_set,
    get_indices_mapping_2_wvn,
    factorize_state,
    extract_quantum_states,
    evaluate_fully_classical_factors
)
from utils_m3_swap import (
    XorY_augment
)
from utils_m4_partitioning import (
    sorted_insertion_decomposition
)
from utils_results import (
    variance_of_decomp,
    sampling_cost
)
from openfermion import (
    get_sparse_operator,
    jordan_wigner
)

import numpy as np
import pickle
import sys
import os
import multiprocessing as mp

# Add NOQE_ST to path for shadow tomography imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NOQE_ST'))
from shadow_tomography import ShadowEstimator
from matrix_estimation import MatrixElementEstimator


# Shadow tomography configuration
NUM_WORKERS = mp.cpu_count()  # Use all available CPUs


def state_to_key(state_vector: np.ndarray) -> bytes:
    """
    Convert a state vector to a hashable key for caching.

    Args:
        state_vector: Dense state vector

    Returns:
        Bytes representation for use as dictionary key
    """
    if hasattr(state_vector, 'toarray'):
        state_vector = state_vector.toarray().flatten()
    elif len(state_vector.shape) > 1:
        state_vector = state_vector.flatten()
    return state_vector.tobytes()


def get_or_collect_shadows(state_vector: np.ndarray, shadows_cache: dict, num_shadows: int) -> list:
    """
    Get shadows from cache or collect new ones if not cached.

    This avoids redundant quantum measurements when the same state
    appears in multiple matrix elements.

    Args:
        state_vector: Dense state vector
        shadows_cache: Dictionary mapping state keys to shadow lists
        num_shadows: Number of shadows to collect if not cached

    Returns:
        List of classical shadow density matrices
    """
    key = state_to_key(state_vector)
    if key not in shadows_cache:
        shadows_cache[key] = simulate_shadow_collection(state_vector, num_shadows)
    return shadows_cache[key]


def simulate_shadow_collection(state_vector: np.ndarray, num_shadows: int) -> list:
    """
    Simulate classical shadow collection for a quantum state.

    In practice, this would be done on quantum hardware with random Clifford
    measurements. Here we simulate by generating shadow snapshots from the
    known state vector.

    Args:
        state_vector: Dense state vector (1D array)
        num_shadows: Number of shadows to collect

    Returns:
        List of classical shadow density matrices
    """
    from qiskit.quantum_info import random_clifford, Operator

    # Ensure state_vector is 1D
    if hasattr(state_vector, 'toarray'):
        state_vector = state_vector.toarray().flatten()
    elif len(state_vector.shape) > 1:
        state_vector = state_vector.flatten()

    dim = len(state_vector)
    num_qubits = int(np.log2(dim))

    # True density matrix
    rho_true = np.outer(state_vector, np.conj(state_vector))

    shadows = []
    for _ in range(num_shadows):
        # Generate random Clifford
        cliff = random_clifford(num_qubits)
        U = Operator(cliff).data

        # Apply Clifford to state: U|ψ⟩
        rotated_state = U @ state_vector

        # Compute measurement probabilities
        probs = np.abs(rotated_state) ** 2

        # Sample measurement outcome
        outcome = np.random.choice(dim, p=probs)

        # Construct |b⟩⟨b|
        b_state = np.zeros(dim)
        b_state[outcome] = 1.0
        rho_b = np.outer(b_state, b_state)

        # Apply inverse channel: ρ̂ = (2^n + 1)(U†|b⟩⟨b|U) - I
        U_dag = U.conj().T
        sigma = U_dag @ rho_b @ U
        rho_hat = (dim + 1) * sigma - np.eye(dim)

        shadows.append(rho_hat)

    return shadows


def shadow_sampling_cost_diagonal(NQ: int, num_shadows: int) -> float:
    """
    Compute sampling cost for diagonal elements using shadow tomography.

    For diagonal elements ⟨ψ|H|ψ⟩, shadow tomography uses linear estimation
    with sample complexity O(||H||² / ε²).

    Args:
        NQ: Number of quantum qubits
        num_shadows: Number of shadows collected

    Returns:
        Effective standard deviation for sampling cost
    """
    # Shadow tomography variance scales as O(2^n / num_shadows) for linear estimation
    # This is a simplified model - actual variance depends on observable structure
    dim = 2 ** NQ
    # Variance ~ (2^n + 1)² / num_shadows for worst case
    variance = (dim + 1) ** 2 / num_shadows
    return np.sqrt(variance)


def shadow_sampling_cost_offdiagonal(NQ: int, num_shadows: int, overlap_sq: float) -> float:
    """
    Compute sampling cost for off-diagonal elements using shadow tomography.

    For off-diagonal elements ⟨φ|H|ψ⟩, shadow tomography uses bilinear estimation
    with sample complexity dependent on the overlap |⟨φ|ψ⟩|².

    Args:
        NQ: Number of quantum qubits
        num_shadows: Number of shadows per state
        overlap_sq: |⟨φ|ψ⟩|² between the two states

    Returns:
        Effective standard deviation for sampling cost
    """
    dim = 2 ** NQ
    # Bilinear estimation variance scales as O(2^(2n) / num_shadows²)
    # but is modulated by the overlap
    # From the paper: for orthogonal states (overlap~0), H_ij estimation is harder
    variance = (dim + 1) ** 4 / (num_shadows ** 2)

    # When overlap is small, we divide by it to get H_ij, increasing variance
    if overlap_sq > 1e-6:
        variance = variance / overlap_sq

    return np.sqrt(variance)


def _collect_shadows_worker(args):
    """Worker function for multiprocessing - must be at module level."""
    state_idx, state_vector, num_shadows = args
    shadows = simulate_shadow_collection(state_vector, num_shadows)
    return state_idx, shadows


def _estimate_offdiag_worker(args):
    """Worker function for off-diagonal matrix element estimation (full pipeline)."""
    (i, j, bra_f, ket_f, bra_labels, ket_labels, bra_config, ket_config,
     Hqub, Nqubits, shadows_by_idx, precomputed_data, num_shadows) = args
    print(f"    Estimating off-diagonal element ({i}, {j})...")
    # Step 1: Taper Hamiltonian and evaluate classical factors
    Htapered = project_out_seniority_symmetries(Hqub, Nqubits, bra_config, ket_config)
    HQ, _, _, NQ = evaluate_fully_classical_factors(bra_f, ket_f, bra_labels, ket_labels, Htapered)

    if NQ == 0:
        # Purely classical - no quantum estimation needed
        return i, j, HQ.constant, 0.0

    # Step 2: Get shadows and prepare Hamiltonian matrix
    data = precomputed_data[(i, j)]
    shadows_bra = shadows_by_idx[data['braQ_idx']]
    shadows_ket = shadows_by_idx[data['ketQ_idx']]
    HQsparse = get_sparse_operator(HQ, NQ)
    HQ_dense = HQsparse.toarray()

    # Step 3: Shadow tomography estimation
    estimator = ShadowEstimator()

    # Estimate overlap |S_ij|² = Tr(ρ_i ρ_j)
    dim_Q = 2 ** NQ
    identity = np.eye(dim_Q)
    overlap_mag_sq = estimator.estimate_bilinear(shadows_bra, shadows_ket, identity)
    overlap_mag_sq = max(0, overlap_mag_sq)

    # Estimate H_ij using the paper's formula
    hij_times_sji = estimator.estimate_bilinear(shadows_bra, shadows_ket, HQ_dense)

    if overlap_mag_sq > 1e-6:
        s_ij = np.sqrt(overlap_mag_sq)
        h_ij = hij_times_sji / s_ij
    else:
        h_ij = hij_times_sji

    sig_ij = shadow_sampling_cost_offdiagonal(NQ, num_shadows, overlap_mag_sq)

    return i, j, h_ij, sig_ij


def main():

    # load Q-SENSE basis states
    molecule        = sys.argv[1]
    bond_length     = float(sys.argv[2])
    NUM_SHADOWS  = int(sys.argv[3])   

    filename        = f'{molecule}_data/Uext_CSF_for_Praveen_Smik_{bond_length}.dump'
    output_filename = f'main_outputs/{molecule}_{bond_length}_PT_serial'

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
        obt, tbt = orthogonal_transform_obt_tbt(x_orbrot,list_orb_rot,obt_spatial,tbt_spatial)
    else:
        obt = obt_phys_spatial_to_spin(obt_spatial)
        tbt = tbt_phys_spatial_to_spin(tbt_spatial)

    Hfer    = make_short_H_ferm_op(Enuc, obt, tbt)
    Hqub    = jordan_wigner(Hfer)

    Nqubits = obt.shape[0]
    Norb    = Nqubits // 2
    dim     = 2 ** Nqubits

    # obtain relevant information about Q-SENSE states (UCSFs, CSFs, W information) in a linear list

    UCSF_tz_states = []
    CSF_tz_states  = []
    W_amplitudes   = []

    for i, ucsf_list in enumerate(list_list_Uext_mp2_CSF):
        for j, ucsf in enumerate(ucsf_list):
            UCSF_tz_states.append(ucsf)
            CSF_tz_states.append(list_list_refCSF[i][j])
            W_amplitudes.append(list_list_Uext_mp2_ampld[i])

    # process information so that we can taper and factorize the Q-SENSE states

    Nstates          = len(UCSF_tz_states)
    print(f"Processing {Nstates} Q-SENSE basis states...")
    configs          = [tz_state_seniority_config(tz_state) for tz_state in UCSF_tz_states]
    UCSF_information = [get_indices_mapping_2_wvn(CSF_tz_states[i], W_amplitudes[i], Norb) for i in range(Nstates)]

    SW_list          = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'W']) for i in range(Nstates)]
    SV_list          = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'V']) for i in range(Nstates)]
    SN_list          = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'N']) for i in range(Nstates)]
    state_type_list  = [UCSF_information[i][1] for i in range(Nstates)]

    # taper and factorize the Q-SENSE basis states

    statevectors                    = [convert_TZ_format_to_sparse_format(dim, tz_state) for tz_state in UCSF_tz_states]
    tapered_statevectors            = [convert_dense_format_to_sparse_format(compress_state(psi.toarray()[0])) for psi in statevectors]
    factorized_tapered_statevectors = [factorize_state(tapered_statevectors[i], SW_list[i], SV_list[i], SN_list[i], state_type_list[i])
                                       for i in range(Nstates)]


    # =============================================================================
    # PHASE 1: Pre-compute all quantum state pairs and identify unique states
    # =============================================================================
    print("Phase 1: Pre-computing quantum state pairs for all (i,j) combinations...")

    # Store pre-computed data for each matrix element
    # Key: (i, j), Value: dict with NQ, full_Q_block, state indices
    precomputed_data = {}

    # Track unique quantum states by their content
    unique_states = {}  # key: state bytes -> value: state index
    state_index_to_key = {}  # reverse mapping: index -> key
    next_state_idx = 0

    # Track unique state pairs for off-diagonal elements
    unique_state_pairs = set()  # set of (braQ_idx, ketQ_idx) tuples

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

    # Pre-compute diagonal elements (state extraction only, no Hamiltonian processing)
    for i in range(Nstates):
        ket_f      = factorized_tapered_statevectors[i]
        ket_labels = UCSF_information[i][0]
        ket_config = configs[i]

        # Extract quantum states without processing Hamiltonian
        ketQ, _, NQ, full_Q_block = extract_quantum_states(ket_f, ket_f, ket_labels, ket_labels)

        if NQ == 0:
            precomputed_data[(i, i)] = {'NQ': 0, 'ket_config': ket_config}
        else:
            ketQ_idx, ketQ_dense = register_state(ketQ)
            precomputed_data[(i, i)] = {
                'NQ': NQ,
                'full_Q_block': full_Q_block,
                'ket_config': ket_config,
                'ketQ_idx': ketQ_idx
            }

    # Pre-compute off-diagonal elements (state extraction only, no Hamiltonian processing)
    for i in range(Nstates):
        for j in range(Nstates):
            if i > j:
                bra_f      = factorized_tapered_statevectors[i]
                bra_labels = UCSF_information[i][0]
                bra_config = configs[i]

                ket_f      = factorized_tapered_statevectors[j]
                ket_labels = UCSF_information[j][0]
                ket_config = configs[j]

                # Extract quantum states without processing Hamiltonian
                braQ, ketQ, NQ, full_Q_block = extract_quantum_states(bra_f, ket_f, bra_labels, ket_labels)

                if NQ == 0:
                    precomputed_data[(i, j)] = {
                        'NQ': 0,
                        'bra_config': bra_config,
                        'ket_config': ket_config
                    }
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
                    # Track unique state pairs (order matters for bilinear estimation)
                    unique_state_pairs.add((braQ_idx, ketQ_idx))

    print(f"  Found {len(unique_states)} unique quantum states across all matrix elements")
    print(f"  Found {len(unique_state_pairs)} unique quantum state pairs for off-diagonal elements")
    print(f"  Total matrix elements: {len(precomputed_data)}")

    # =============================================================================
    # PHASE 2: Collect shadows for all unique quantum states (PARALLEL)
    # =============================================================================
    import time
    print(f"\nPhase 2: Collecting {NUM_SHADOWS} shadows for each unique quantum state (parallel)...")
    print(f"  Using {NUM_WORKERS} workers for {next_state_idx} unique states")

    # Prepare arguments for parallel processing: list of (state_idx, state_vector, num_shadows)
    shadow_args = [(idx, state_index_to_key[idx][1], NUM_SHADOWS) for idx in range(next_state_idx)]

    # Parallel shadow collection using multiprocessing Pool
    start_time = time.time()
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(_collect_shadows_worker, shadow_args)
    elapsed = time.time() - start_time

    shadows_by_idx = {state_idx: shadows for state_idx, shadows in results}

    print(f"  Completed {next_state_idx} shadow collections in {elapsed:.1f}s ({elapsed/next_state_idx:.2f}s per state avg)")

    # =============================================================================
    # PHASE 3: Estimate matrix elements using pre-collected shadows
    # =============================================================================
    print("\nPhase 3: Estimating matrix elements from shadows...")

    Hsub = np.zeros([Nstates, Nstates], dtype=np.complex128)
    sig_matrix = np.zeros([Nstates, Nstates], dtype=np.complex128)
    estimator = ShadowEstimator()

    # Diagonal elements
    print("  Estimating diagonal elements...")
    for i in range(Nstates):
        with open(output_filename, 'a') as f:
            print(f'{i, i}', file=f)

        data = precomputed_data[(i, i)]
        ket_f = factorized_tapered_statevectors[i]
        ket_labels = UCSF_information[i][0]

        Htapered = project_out_seniority_symmetries(Hqub, Nqubits, data['ket_config'], data['ket_config'])
        HQ, _, _, NQ = evaluate_fully_classical_factors(ket_f, ket_f, ket_labels, ket_labels, Htapered)

        if NQ == 0:
            Hsub[i, i] = HQ.constant
            sig_matrix[i, i] = 0
        else:
            shadows_i = shadows_by_idx[data['ketQ_idx']]
            HQsparse = get_sparse_operator(HQ)
            HQ_dense = HQsparse.toarray()

            h_ii = estimator.estimate_linear(shadows_i, HQ_dense)
            sig_ii = shadow_sampling_cost_diagonal(NQ, NUM_SHADOWS)
            Hsub[i, i] = h_ii
            sig_matrix[i, i] = sig_ii

    # Off-diagonal elements
    print("  Estimating off-diagonal elements (parallel)...")
    ij_pairs = [(i, j) for i in range(Nstates) for j in range(Nstates) if i > j]
    total_pairs = len(ij_pairs)

    # Prepare arguments for parallel processing - include full pipeline data
    offdiag_args = []
    for i, j in ij_pairs:
        print(f"    Preparing off-diagonal pair ({i}, {j})...")
        data = precomputed_data[(i, j)]
        bra_f = factorized_tapered_statevectors[i]
        ket_f = factorized_tapered_statevectors[j]
        bra_labels = UCSF_information[i][0]
        ket_labels = UCSF_information[j][0]
        bra_config = data['bra_config']
        ket_config = data['ket_config']

        offdiag_args.append((
            i, j, bra_f, ket_f, bra_labels, ket_labels, bra_config, ket_config,
            Hqub, Nqubits, shadows_by_idx, precomputed_data, NUM_SHADOWS
        ))

    print(f"    Processing {total_pairs} off-diagonal pairs...")

    # Parallel off-diagonal estimation (full pipeline)
    start_time = time.time()
    with mp.Pool(processes=NUM_WORKERS) as pool:
        offdiag_results = pool.map(_estimate_offdiag_worker, offdiag_args)
    elapsed = time.time() - start_time

    # Fill results into matrices
    for i, j, h_ij, sig_ij in offdiag_results:
        Hsub[i, j] = h_ij
        Hsub[j, i] = np.conj(h_ij)
        sig_matrix[i, j] = sig_ij
        sig_matrix[j, i] = sig_ij

    print(f"    Completed {total_pairs} off-diagonal estimations in {elapsed:.1f}s")

    vals, vecs = np.linalg.eigh(Hsub)
    Egs        = vals[0]
    c          = vecs[:,0]
    cost       = sampling_cost(c, sig_matrix)

    with open(output_filename, 'a') as f:
        print(f'''
        Final Results:
            Method              : PT + Shadow Tomography
            Molecule            : {molecule}
            Bond Length         : {bond_length}
            Ground State Energy : {Egs}
            Sampling Cost       : {cost}
            Num Shadows/State   : {NUM_SHADOWS}
        ''', file=f)


if __name__ == '__main__':
    main()
