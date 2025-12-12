
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
from qiskit.quantum_info import random_clifford, Operator
import time
from datetime import datetime

# Add NOQE_ST to path for shadow tomography imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NOQE_ST'))
from shadow_tomography import ShadowEstimator


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
        List of (Clifford, outcome) tuples where Clifford is stored in compact
        tableau form O(n²) and outcome is the measurement result (integer).
        This saves memory vs storing full 2^n × 2^n unitary matrices.
    """

    # Ensure state_vector is 1D
    if hasattr(state_vector, 'toarray'):
        state_vector = state_vector.toarray().flatten()
    elif len(state_vector.shape) > 1:
        state_vector = state_vector.flatten()

    dim = len(state_vector)
    num_qubits = int(np.log2(dim))

    # IMPORTANT: Normalize state vector for correct probability sampling
    # The quantum states from Q-SENSE factorization may not be normalized
    norm = np.linalg.norm(state_vector)
    if norm > 1e-10:
        state_vector = state_vector / norm

    shadows = []
    for _ in range(num_shadows):
        # Generate random Clifford - stored in compact tableau form O(n²)
        cliff = random_clifford(num_qubits)
        # Convert to unitary only for simulation (not stored)
        U = Operator(cliff).data

        # Apply Clifford to state: U|ψ⟩
        rotated_state = U @ state_vector

        # Compute measurement probabilities
        probs = np.abs(rotated_state) ** 2

        # Sample measurement outcome (computational basis)
        outcome = np.random.choice(dim, p=probs)

        # Store compact representation: (Clifford tableau, outcome)
        # Clifford tableau is O(n²) vs O(4^n) for full unitary matrix
        # Shadow can be reconstructed as: ρ̂ = (2^n + 1)(U†|b⟩⟨b|U) - I
        shadows.append((cliff, outcome))

    return shadows


def reconstruct_shadow(cliff, outcome: int) -> np.ndarray:
    """
    Reconstruct a classical shadow density matrix from (Clifford, outcome) tuple.

    Args:
        cliff: Qiskit Clifford object (compact tableau representation)
        outcome: Measurement outcome (integer index)

    Returns:
        Shadow density matrix ρ̂ = (2^n + 1)(U†|b⟩⟨b|U) - I
    """
    # Convert Clifford tableau to full unitary matrix
    U = Operator(cliff).data
    dim = U.shape[0]

    # Construct |b⟩⟨b|
    b_state = np.zeros(dim)
    b_state[outcome] = 1.0
    rho_b = np.outer(b_state, b_state)

    # Apply inverse channel: ρ̂ = (2^n + 1)(U†|b⟩⟨b|U) - I
    U_dag = U.conj().T
    sigma = U_dag @ rho_b @ U
    rho_hat = (dim + 1) * sigma - np.eye(dim)

    return rho_hat


def reconstruct_shadows(compact_shadows: list) -> list:
    """
    Reconstruct full shadow matrices from compact (Clifford, outcome) representation.

    Args:
        compact_shadows: List of (Clifford, outcome) tuples

    Returns:
        List of full shadow density matrices
    """
    return [reconstruct_shadow(cliff, outcome) for cliff, outcome in compact_shadows]


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


def _estimate_diag_worker(args):
    """Worker function for diagonal matrix element estimation (parallel)."""
    (i, ket_f, ket_labels, ket_config, Hqub, Nqubits,
     shadows_by_idx, precomputed_data, num_shadows, HQ_cache) = args

    # Use precomputed HQ if available, otherwise compute
    if (i, i) in HQ_cache:
        HQ, NQ = HQ_cache[(i, i)]
    else:
        Htapered = project_out_seniority_symmetries(Hqub, Nqubits, ket_config, ket_config)
        HQ, _, _, NQ = evaluate_fully_classical_factors(ket_f, ket_f, ket_labels, ket_labels, Htapered)

    if NQ == 0:
        return i, HQ.constant, 0.0

    data = precomputed_data[(i, i)]
    compact_shadows_i = shadows_by_idx[data['ketQ_idx']]
    # Reconstruct full shadow matrices from compact (U, outcome) format
    shadows_i = reconstruct_shadows(compact_shadows_i)
    HQsparse = get_sparse_operator(HQ)
    HQ_dense = HQsparse.toarray()

    estimator = ShadowEstimator()
    h_ii = estimator.estimate_linear(shadows_i, HQ_dense)
    sig_ii = shadow_sampling_cost_diagonal(NQ, num_shadows)

    return i, h_ii, sig_ii


def _estimate_offdiag_worker_zero_superposition(args):
    """Worker function for off-diagonal matrix element estimation using |0⟩±|ψ⟩ superpositions.

    Uses the formula:
        H_ij = 2 * (Tr(ρ_i^+ ρ_j^+ H) + Tr(ρ_i^- ρ_j^- H))

    Where:
        ρ_i^+ = |+_i⟩⟨+_i| with |+_i⟩ = (|0⟩ + |ψ_i⟩)/√2
        ρ_i^- = |−_i⟩⟨−_i| with |−_i⟩ = (|0⟩ - |ψ_i⟩)/√2

    This uses bilinear shadow estimation on the ± superposition states.
    """
    (i, j, shadows_plus_by_idx, shadows_minus_by_idx, precomputed_data, num_shadows, HQ_cache) = args

    # Use precomputed HQ from cache
    HQ, NQ = HQ_cache[(i, j)]

    if NQ == 0:
        # Purely classical - no quantum estimation needed
        return i, j, HQ.constant, 0.0

    # Get precomputed data
    data = precomputed_data[(i, j)]
    braQ_idx = data['braQ_idx']
    ketQ_idx = data['ketQ_idx']

    # Get compact shadows and reconstruct for the |0⟩±|ψ⟩ superposition states
    shadows_bra_plus = reconstruct_shadows(shadows_plus_by_idx[braQ_idx])    # (|0⟩ + |braQ⟩)/√2
    shadows_bra_minus = reconstruct_shadows(shadows_minus_by_idx[braQ_idx])  # (|0⟩ - |braQ⟩)/√2
    shadows_ket_plus = reconstruct_shadows(shadows_plus_by_idx[ketQ_idx])    # (|0⟩ + |ketQ⟩)/√2
    shadows_ket_minus = reconstruct_shadows(shadows_minus_by_idx[ketQ_idx])  # (|0⟩ - |ketQ⟩)/√2

    HQsparse = get_sparse_operator(HQ, NQ)
    HQ_dense = HQsparse.toarray()

    # Get H_00 = ⟨0|H|0⟩ for the correction term
    H_00 = HQ_dense[0, 0].real

    # Use bilinear shadow estimation (static method from ShadowEstimator)
    # Tr(ρ_i^+ ρ_j^+ H) - bilinear estimation with + superposition shadows
    tr_plus = ShadowEstimator.estimate_bilinear(shadows_bra_plus, shadows_ket_plus, HQ_dense)

    # Tr(ρ_i^- ρ_j^- H) - bilinear estimation with - superposition shadows
    tr_minus = ShadowEstimator.estimate_bilinear(shadows_bra_minus, shadows_ket_minus, HQ_dense)

    # When S_ij = 0 and ⟨0|ψ_i⟩ = ⟨0|ψ_j⟩ = 0:
    # 2 * (tr_plus + tr_minus) = H_00 + H_ij
    # Therefore: H_ij = 2 * (tr_plus + tr_minus) - H_00
    h_ij = 2.0 * (tr_plus + tr_minus) - H_00

    # Variance estimate for bilinear estimation
    # Bilinear variance scales as O(2^(2n) / num_shadows²)
    dim = 2 ** NQ
    variance = 4.0 * 2.0 * (dim + 1) ** 4 / (num_shadows ** 2)  # Factor of 4 from the 2* and 2 terms
    sig_ij = np.sqrt(variance)

    return i, j, h_ij, sig_ij


def main():

    # load Q-SENSE basis states
    molecule        = sys.argv[1]
    bond_length     = float(sys.argv[2])
    NUM_SHADOWS  = int(sys.argv[3])   

    filename        = f'{molecule}_data/Uext_CSF_for_Praveen_Smik_{bond_length}.dump'
    timestamp       = datetime.now().strftime('%Y%m%d_%H%M')
    output_filename = f'main_outputs/{molecule}_{bond_length}_PT_serial_{timestamp}'

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
    # For Hadamard test approach, we also create superposition states |+⟩ = (|bra⟩ + |ket⟩)/√2
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
                    braQ_idx, _ = register_state(braQ)
                    ketQ_idx, _ = register_state(ketQ)

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

    print(f"  Found {len(unique_states)} unique quantum states (including superposition states)")
    print(f"  Found {len(unique_state_pairs)} unique quantum state pairs for off-diagonal elements")
    print(f"  Total matrix elements: {len(precomputed_data)}")

    # =============================================================================
    # PHASE 1b: Precompute HQ matrices for all matrix elements
    # =============================================================================
    print("\nPhase 1b: Precomputing tapered Hamiltonians and HQ matrices...")

    HQ_cache = {}  # Key: (i, j), Value: (HQ, NQ)

    # Precompute for diagonal elements
    for i in range(Nstates):
        ket_f = factorized_tapered_statevectors[i]
        ket_labels = UCSF_information[i][0]
        ket_config = configs[i]

        Htapered = project_out_seniority_symmetries(Hqub, Nqubits, ket_config, ket_config)
        HQ, _, _, NQ = evaluate_fully_classical_factors(ket_f, ket_f, ket_labels, ket_labels, Htapered)
        HQ_cache[(i, i)] = (HQ, NQ)

    # Precompute for off-diagonal elements
    for i in range(Nstates):
        for j in range(Nstates):
            if i > j:
                bra_f = factorized_tapered_statevectors[i]
                bra_labels = UCSF_information[i][0]
                bra_config = configs[i]

                ket_f = factorized_tapered_statevectors[j]
                ket_labels = UCSF_information[j][0]
                ket_config = configs[j]

                Htapered = project_out_seniority_symmetries(Hqub, Nqubits, bra_config, ket_config)
                HQ, _, _, NQ = evaluate_fully_classical_factors(bra_f, ket_f, bra_labels, ket_labels, Htapered)
                HQ_cache[(i, j)] = (HQ, NQ)

    print(f"  Precomputed {len(HQ_cache)} HQ matrices")

    # =============================================================================
    # PHASE 2: Collect shadows for all unique quantum states (PARALLEL)
    # Also collect shadows for (|0⟩ + |ψ⟩)/√2 and (|0⟩ - |ψ⟩)/√2 superpositions
    # =============================================================================
    print(f"\nPhase 2: Collecting {NUM_SHADOWS} shadows for each unique quantum state (parallel)...")
    print(f"  Also collecting shadows for |0⟩±|ψ⟩ superposition states")
    print(f"  Using {NUM_WORKERS} workers for {next_state_idx} unique states")

    # Prepare arguments for parallel processing: list of (state_idx, state_vector, num_shadows)
    shadow_args = [(idx, state_index_to_key[idx][1], NUM_SHADOWS) for idx in range(next_state_idx)]

    # Also prepare superposition states (|0⟩ + |ψ⟩)/√2 and (|0⟩ - |ψ⟩)/√2
    superposition_args_plus = []
    superposition_args_minus = []
    for idx in range(next_state_idx):
        state_vector = state_index_to_key[idx][1]
        dim = len(state_vector)
        # |0⟩ is the computational basis state |00...0⟩
        zero_state = np.zeros(dim)
        zero_state[0] = 1.0
        # Create (|0⟩ + |ψ⟩)/√2 and normalize
        plus_superposition = (zero_state + state_vector) / np.sqrt(2)
        plus_superposition = plus_superposition / np.linalg.norm(plus_superposition)
        # Create (|0⟩ - |ψ⟩)/√2 and normalize
        minus_superposition = (zero_state - state_vector) / np.sqrt(2)
        minus_superposition = minus_superposition / np.linalg.norm(minus_superposition)

        superposition_args_plus.append((f"plus_{idx}", plus_superposition, NUM_SHADOWS))
        superposition_args_minus.append((f"minus_{idx}", minus_superposition, NUM_SHADOWS))

    # Combine all shadow collection tasks
    all_shadow_args = shadow_args + superposition_args_plus + superposition_args_minus
    total_states = len(all_shadow_args)
    print(f"  Total shadow collections: {total_states} ({next_state_idx} original + {next_state_idx} |+⟩ + {next_state_idx} |−⟩)")

    # Parallel shadow collection using multiprocessing Pool
    start_time = time.time()
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(_collect_shadows_worker, all_shadow_args)
    elapsed = time.time() - start_time

    # Separate results into original states and superposition states
    shadows_by_idx = {}
    shadows_plus_by_idx = {}
    shadows_minus_by_idx = {}
    for state_idx, shadows in results:
        if isinstance(state_idx, int):
            shadows_by_idx[state_idx] = shadows
        elif isinstance(state_idx, str) and state_idx.startswith("plus_"):
            orig_idx = int(state_idx.split("_")[1])
            shadows_plus_by_idx[orig_idx] = shadows
        elif isinstance(state_idx, str) and state_idx.startswith("minus_"):
            orig_idx = int(state_idx.split("_")[1])
            shadows_minus_by_idx[orig_idx] = shadows

    print(f"  Completed {total_states} shadow collections in {elapsed:.1f}s ({elapsed/total_states:.2f}s per state avg)")

    # =============================================================================
    # PHASE 3: Estimate matrix elements using pre-collected shadows (PARALLEL)
    # =============================================================================
    print("\nPhase 3: Estimating matrix elements from shadows...")

    Hsub = np.zeros([Nstates, Nstates], dtype=np.complex128)
    sig_matrix = np.zeros([Nstates, Nstates], dtype=np.complex128)

    # Diagonal elements (parallel)
    print("  Estimating diagonal elements (parallel)...")
    diag_args = []
    for i in range(Nstates):
        ket_f = factorized_tapered_statevectors[i]
        ket_labels = UCSF_information[i][0]
        ket_config = configs[i]

        diag_args.append((
            i, ket_f, ket_labels, ket_config, Hqub, Nqubits,
            shadows_by_idx, precomputed_data, NUM_SHADOWS, HQ_cache
        ))

    start_time = time.time()
    with mp.Pool(processes=NUM_WORKERS) as pool:
        diag_results = pool.map(_estimate_diag_worker, diag_args)
    elapsed = time.time() - start_time

    # Fill diagonal results into matrices
    for i, h_ii, sig_ii in diag_results:
        Hsub[i, i] = h_ii
        sig_matrix[i, i] = sig_ii

    print(f"    Completed {Nstates} diagonal estimations in {elapsed:.1f}s")

    # Off-diagonal elements using |0⟩±|ψ⟩ superposition approach (parallel)
    # Uses the formula: H_ij = 2 * (Tr(ρ_i^+ ρ_j^+ H) + Tr(ρ_i^- ρ_j^- H))
    # where ρ_i^± are density matrices for (|0⟩ ± |ψ_i⟩)/√2
    print("  Estimating off-diagonal elements via |0⟩±|ψ⟩ superposition (parallel)...")
    ij_pairs = [(i, j) for i in range(Nstates) for j in range(Nstates) if i > j]
    total_pairs = len(ij_pairs)

    # Prepare arguments for parallel processing with ± superposition shadows
    offdiag_args = []
    for i, j in ij_pairs:
        offdiag_args.append((
            i, j, shadows_plus_by_idx, shadows_minus_by_idx, precomputed_data, NUM_SHADOWS, HQ_cache
        ))

    print(f"    Processing {total_pairs} off-diagonal pairs...")

    # Parallel off-diagonal estimation using |0⟩±|ψ⟩ superposition approach
    start_time = time.time()
    with mp.Pool(processes=NUM_WORKERS) as pool:
        offdiag_results = pool.map(_estimate_offdiag_worker_zero_superposition, offdiag_args)
    elapsed = time.time() - start_time

    # Fill results into matrices
    for i, j, h_ij, sig_ij in offdiag_results:
        Hsub[i, j] = h_ij
        Hsub[j, i] = np.conj(h_ij)
        sig_matrix[i, j] = sig_ij
        sig_matrix[j, i] = sig_ij

    print(f"    Completed {total_pairs} off-diagonal estimations in {elapsed:.1f}s")

    # =============================================================================
    # PHASE 4: Compute EXACT matrix elements for comparison
    # =============================================================================
    print("\nPhase 4: Computing exact matrix elements for comparison...")

    Hsub_exact = np.zeros([Nstates, Nstates], dtype=np.complex128)

    # Exact diagonal elements
    for i in range(Nstates):
        ket_f = factorized_tapered_statevectors[i]
        ket_labels = UCSF_information[i][0]
        ket_config = configs[i]

        Htapered = project_out_seniority_symmetries(Hqub, Nqubits, ket_config, ket_config)
        HQ, ketQ, _, NQ = evaluate_fully_classical_factors(ket_f, ket_f, ket_labels, ket_labels, Htapered)

        if NQ == 0:
            Hsub_exact[i, i] = HQ.constant
        else:
            HQsparse = get_sparse_operator(HQ)
            # Normalize ketQ for consistency with shadow tomography
            ketQ_norm = ketQ / np.linalg.norm(ketQ)
            ketQ_sparse = convert_dense_format_to_sparse_format(ketQ_norm)
            Hsub_exact[i, i] = (ketQ_sparse @ HQsparse @ ketQ_sparse.T)[0, 0]

    # Exact off-diagonal elements
    # NOTE: We do NOT apply the ij_shift here to match the shadow tomography estimation
    # which uses the unshifted Hamiltonian. The shift was originally used for numerical
    # conditioning but affects the result after tapering/projection.
    for i in range(Nstates):
        for j in range(Nstates):
            if i > j:
                bra_f = factorized_tapered_statevectors[i]
                bra_labels = UCSF_information[i][0]
                bra_config = configs[i]

                ket_f = factorized_tapered_statevectors[j]
                ket_labels = UCSF_information[j][0]
                ket_config = configs[j]

                # Use unshifted Hamiltonian to match shadow tomography
                Htapered = project_out_seniority_symmetries(Hqub, Nqubits, bra_config, ket_config)
                HQ, braQ, ketQ, NQ = evaluate_fully_classical_factors(bra_f, ket_f, bra_labels, ket_labels, Htapered)

                if NQ == 0:
                    Hsub_exact[i, j] = HQ.constant
                else:
                    HQsparse = get_sparse_operator(HQ, NQ)
                    # Normalize states for consistency with shadow tomography
                    braQ_norm = braQ / np.linalg.norm(braQ)
                    ketQ_norm = ketQ / np.linalg.norm(ketQ)
                    braQ_sparse = convert_dense_format_to_sparse_format(braQ_norm)
                    ketQ_sparse = convert_dense_format_to_sparse_format(ketQ_norm)
                    Hsub_exact[i, j] = (braQ_sparse @ HQsparse @ ketQ_sparse.T)[0, 0]

                Hsub_exact[j, i] = Hsub_exact[i, j]

    # Compute exact ground state energy
    vals_exact, vecs_exact = np.linalg.eigh(Hsub_exact)
    Egs_exact = vals_exact[0]

    # =============================================================================
    # DIAGNOSTIC: Check S_ij = ⟨ψ_i|ψ_j⟩ and ⟨0|H|0⟩ for the formula assumptions
    # =============================================================================
    print("\n--- Diagnostic: Checking formula assumptions ---")
    for i in range(Nstates):
        for j in range(Nstates):
            if i > j:
                data = precomputed_data[(i, j)]
                if data['NQ'] > 0:
                    braQ_idx = data['braQ_idx']
                    ketQ_idx = data['ketQ_idx']

                    # Get the quantum state vectors
                    braQ = state_index_to_key[braQ_idx][1]
                    ketQ = state_index_to_key[ketQ_idx][1]

                    # Normalize for overlap computation
                    braQ_norm = braQ / np.linalg.norm(braQ)
                    ketQ_norm = ketQ / np.linalg.norm(ketQ)

                    # Compute overlap S_ij = ⟨ψ_i|ψ_j⟩
                    S_ij = np.vdot(braQ_norm, ketQ_norm)

                    # Compute ⟨0|H|0⟩
                    HQ, NQ = HQ_cache[(i, j)]
                    HQsparse = get_sparse_operator(HQ, NQ)
                    HQ_dense = HQsparse.toarray()
                    H_00 = HQ_dense[0, 0].real  # ⟨0|H|0⟩

                    # Compute ⟨0|ψ_i⟩ and ⟨0|ψ_j⟩
                    dim = len(braQ_norm)
                    zero_state = np.zeros(dim)
                    zero_state[0] = 1.0
                    overlap_0_bra = np.vdot(zero_state, braQ_norm)
                    overlap_0_ket = np.vdot(zero_state, ketQ_norm)

                    print(f"  ({i},{j}): S_ij = {S_ij:.6f}, ⟨0|H|0⟩ = {H_00:.6f}, ⟨0|ψ_i⟩ = {overlap_0_bra:.6f}, ⟨0|ψ_j⟩ = {overlap_0_ket:.6f}")

    # Compute shadow tomography results
    vals, vecs = np.linalg.eigh(Hsub)
    Egs        = vals[0]
    c          = vecs[:,0]
    cost       = sampling_cost(c, sig_matrix)

    # =============================================================================
    # Output: Matrix element comparison
    # =============================================================================
    with open(output_filename, 'a') as f:
        print(f"\n{'='*70}", file=f)
        print(f"Matrix Element Comparison: Shadow Tomography vs Exact", file=f)
        print(f"{'='*70}", file=f)
        print(f"Molecule: {molecule}, Bond Length: {bond_length}, Num Shadows: {NUM_SHADOWS}", file=f)
        print(f"\n--- Diagonal Elements ---", file=f)
        print(f"{'(i,i)':<10} {'ST Estimate':>15} {'Exact':>15} {'Error':>15} {'Rel Error %':>12}", file=f)
        print(f"{'-'*10} {'-'*15} {'-'*15} {'-'*15} {'-'*12}", file=f)

        for i in range(Nstates):
            st_val = Hsub[i, i].real
            exact_val = Hsub_exact[i, i].real
            error = abs(st_val - exact_val)
            rel_error = 100 * error / abs(exact_val) if abs(exact_val) > 1e-10 else 0
            print(f"({i},{i})      {st_val:>15.8f} {exact_val:>15.8f} {error:>15.8f} {rel_error:>11.2f}%", file=f)

        print(f"\n--- Off-Diagonal Elements ---", file=f)
        print(f"{'(i,j)':<10} {'ST Estimate':>15} {'Exact':>15} {'Error':>15} {'Rel Error %':>12}", file=f)
        print(f"{'-'*10} {'-'*15} {'-'*15} {'-'*15} {'-'*12}", file=f)

        for i in range(Nstates):
            for j in range(Nstates):
                if i > j:
                    st_val = Hsub[i, j].real
                    exact_val = Hsub_exact[i, j].real
                    error = abs(st_val - exact_val)
                    rel_error = 100 * error / abs(exact_val) if abs(exact_val) > 1e-10 else 0
                    print(f"({i},{j})      {st_val:>15.8f} {exact_val:>15.8f} {error:>15.8f} {rel_error:>11.2f}%", file=f)

        print(f"\n--- Summary Statistics ---", file=f)
        # Compute overall errors
        diag_errors = [abs(Hsub[i,i].real - Hsub_exact[i,i].real) for i in range(Nstates)]
        offdiag_errors = [abs(Hsub[i,j].real - Hsub_exact[i,j].real)
                         for i in range(Nstates) for j in range(Nstates) if i > j]

        print(f"Diagonal elements:     mean error = {np.mean(diag_errors):.8f}, max error = {np.max(diag_errors):.8f}", file=f)
        if offdiag_errors:
            print(f"Off-diagonal elements: mean error = {np.mean(offdiag_errors):.8f}, max error = {np.max(offdiag_errors):.8f}", file=f)

        print(f"\n--- Energy Comparison ---", file=f)
        print(f"Exact Ground State Energy:  {Egs_exact:.10f}", file=f)
        print(f"Shadow Tomography Energy:   {Egs:.10f}", file=f)
        print(f"Energy Error:               {abs(Egs - Egs_exact):.10f}", file=f)

        print(f'''
        Final Results:
            Method              : PT + Shadow Tomography
            Molecule            : {molecule}
            Bond Length         : {bond_length}
            Ground State Energy : {Egs}
            Exact Energy        : {Egs_exact}
            Energy Error        : {abs(Egs - Egs_exact)}
            Sampling Cost       : {cost}
            Num Shadows/State   : {NUM_SHADOWS}
        ''', file=f)


if __name__ == '__main__':
    main()
