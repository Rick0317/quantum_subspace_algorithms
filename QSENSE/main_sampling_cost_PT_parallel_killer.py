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
    evaluate_fully_classical_factors,
    extract_quantum_states,
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
import multiprocess as mp

# Import qubit operator 1-norm utility
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'BLISS'))
from single_fermion_killer import qubit_operator_1norm


def identify_ket_sn_qubits_in_quantum_region(full_Q_block, ket_labels, ket_factorized):
    """
    Identify which qubits in the quantum region come from S_N for the ket state,
    and determine their occupation (0 or 1).

    Args:
        full_Q_block: tuple of spatial orbital indices in the quantum region
        ket_labels: dict mapping orbital index -> 'W', 'V', or 'N' for ket state
        ket_factorized: factorization dict for ket state

    Returns:
        list of tuples: [(spatial_orbital, occupation), ...]
        where occupation is 0 (unoccupied) or 1 (occupied)
    """
    sn_qubits_info = []

    for q in full_Q_block:
        ket_type = ket_labels.get(q, 'N')
        if ket_type == 'N':
            # This qubit comes from S_N in the ket state
            # Get the occupation from the factorized state
            # For S_N orbitals, the key is (q,) and the value is a 2-element array
            # |0⟩ = [1, 0], |1⟩ = [0, 1]
            single_qubit_state = ket_factorized.get((q,), None)
            if single_qubit_state is not None:
                # Check if it's |0⟩ or |1⟩
                if np.abs(single_qubit_state[0]) > 0.5:
                    occupation = 0  # |0⟩ state (unoccupied)
                else:
                    occupation = 1  # |1⟩ state (occupied)
                sn_qubits_info.append((q, occupation))

    return sn_qubits_info


from scipy.optimize import minimize


def optimize_killer_coefficients(pauli_coeffs, sign, n_restarts=3):
    """
    Find optimal killer coefficients o_P for all P_rest simultaneously using numerical optimization.

    We want to minimize the total 1-norm:
        sum_P |c_Z[P] - o_P| + |c_I[P] + sign*o_P| + |c_X[P]| + |c_Y[P]|

    The optimization variables are the o_P values for each P_rest.

    Args:
        pauli_coeffs: dict mapping P_rest -> {'Z': c_Z, 'I': c_I, 'X': c_X, 'Y': c_Y}
        sign: -1 for (Z-1) killer, +1 for (Z+1) killer
        n_restarts: number of random restarts for optimization

    Returns:
        dict mapping P_rest -> optimal o_P value
    """
    # Get list of P_rest keys that have non-zero Z or I terms
    P_rest_list = [p for p, c in pauli_coeffs.items() if abs(c['Z']) > 1e-12 or abs(c['I']) > 1e-12]

    if not P_rest_list:
        return {}

    n_params = len(P_rest_list)

    # Extract coefficient data
    coeffs_data = []
    for p in P_rest_list:
        c = pauli_coeffs[p]
        c_Z = c['Z']
        c_I = c['I']
        # c_Z, c_I are coefficients of the P * Z and P * I terms respectively in the Hamiltonian
        # Handle complex coefficients
        c_Z_real = c_Z.real if isinstance(c_Z, complex) else float(c_Z)
        c_I_real = c_I.real if isinstance(c_I, complex) else float(c_I)
        c_Z_imag = c_Z.imag if isinstance(c_Z, complex) else 0.0
        c_I_imag = c_I.imag if isinstance(c_I, complex) else 0.0
        coeffs_data.append((c_Z_real, c_I_real, c_Z_imag, c_I_imag))

    def objective(o_values):
        total = 0.0
        for i, (c_Z_real, c_I_real, c_Z_imag, c_I_imag) in enumerate(coeffs_data):
            o = o_values[i]
            # |c_Z - o| for complex c_Z with real o
            term1 = np.sqrt((c_Z_real - o)**2 + c_Z_imag**2)
            # |c_I + sign*o| for complex c_I with real o
            term2 = np.sqrt((c_I_real + sign * o)**2 + c_I_imag**2)
            total += term1 + term2
        return total

    # Compute baseline (o = 0 for all)
    baseline = objective(np.zeros(n_params))

    # Try multiple random restarts
    best_result = None
    best_fun = baseline

    for restart in range(n_restarts):
        # Random initial guess
        x0 = np.random.randn(n_params) * 0.1

        result = minimize(
            objective,
            x0,
            method='Powell',
            options={'maxiter': 2000, 'ftol': 1e-10, 'xtol': 1e-10}
        )

        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result

    # Build result dict
    optimal_o = {}
    if best_result is not None and best_fun < baseline - 1e-10:
        for i, p in enumerate(P_rest_list):
            optimal_o[p] = best_result.x[i]
    else:
        # No improvement, return zeros
        for p in P_rest_list:
            optimal_o[p] = 0.0

    return optimal_o


def apply_killers_to_hamiltonian(HQ, NQ, sn_qubits_info, full_Q_block, verbose_file=None):
    """
    Apply optimized killer O*(Z±1) to reduce the 1-norm of HQ.

    For each qubit that comes from S_N in the ket:
    - If occupation == 0: (Z-1)|0⟩ = 0, so we can subtract O*(Z-1) from H
    - If occupation == 1: (Z+1)|1⟩ = 0, so we can subtract O*(Z+1) from H

    We optimize O to minimize the resulting 1-norm. For each Pauli string P on
    the other qubits, if H contains c_Z * P⊗Z and c_I * P⊗I, then subtracting
    o_P * P⊗(Z-1) gives:
        (c_Z - o_P) * P⊗Z + (c_I + o_P) * P⊗I

    The optimal o_P minimizes |c_Z - o_P| + |c_I + o_P|.

    Args:
        HQ: QubitOperator for the quantum Hamiltonian
        NQ: Number of quantum qubits
        sn_qubits_info: list of (spatial_orbital, occupation) from identify_ket_sn_qubits_in_quantum_region
        full_Q_block: tuple of spatial orbital indices in quantum region (for relabeling)
        verbose_file: file path to write verbose output (None to disable)

    Returns:
        HQ_reduced: The reduced Hamiltonian after applying killers
        total_reduction: Total 1-norm reduction achieved
        killers_applied: List of killer info
    """
    from openfermion import QubitOperator

    def log(msg):
        if verbose_file:
            with open(verbose_file, 'a') as f:
                print(msg, file=f)

    if not sn_qubits_info:
        return HQ, 0.0, []

    # Create mapping from spatial orbital to qubit index in NQ-qubit space
    orbital_to_qubit_idx = {orb: idx for idx, orb in enumerate(sorted(full_Q_block))}

    HQ_current = HQ
    total_reduction = 0.0
    killers_applied = []

    original_norm = qubit_operator_1norm(HQ)

    log(f"    Applying optimized killers for {len(sn_qubits_info)} S_N qubits...")
    log(f"    Original HQ 1-norm: {original_norm:.6f}")

    for spatial_orbital, occupation in sn_qubits_info:
        # Get the qubit index in the NQ-qubit Hamiltonian
        qubit_idx = orbital_to_qubit_idx[spatial_orbital]

        norm_before = qubit_operator_1norm(HQ_current)

        # Collect coefficients for each Pauli string P on qubits other than qubit_idx
        # We need c_Z (coefficient of P⊗Z) and c_I (coefficient of P⊗I)
        # Also track c_X and c_Y which cannot be modified by Z-based killers
        pauli_coeffs = {}  # key: P (tuple excluding qubit_idx), value: {'Z': c_Z, 'I': c_I, 'X': c_X, 'Y': c_Y}

        for pauli_term, coeff in HQ_current.terms.items():
            # Separate the action on qubit_idx from the rest
            P_rest = []
            action_on_target = 'I'  # default: identity on qubit_idx

            for qubit, pauli in pauli_term:
                if qubit == qubit_idx:
                    action_on_target = pauli
                else:
                    P_rest.append((qubit, pauli))

            P_rest = tuple(P_rest)

            if P_rest not in pauli_coeffs:
                pauli_coeffs[P_rest] = {'Z': 0.0, 'I': 0.0, 'X': 0.0, 'Y': 0.0}

            pauli_coeffs[P_rest][action_on_target] += coeff

        # Now optimize O for each P_rest
        # H - O*(Z-1) for occupation=0, or H - O*(Z+1) for occupation=1
        # sign = -1 for (Z-1), sign = +1 for (Z+1)
        sign = -1 if occupation == 0 else +1

        # Diagnostic: show what terms exist on the target qubit
        n_Z = sum(1 for p, c in pauli_coeffs.items() if abs(c['Z']) > 1e-12)
        n_I = sum(1 for p, c in pauli_coeffs.items() if abs(c['I']) > 1e-12)
        n_X = sum(1 for p, c in pauli_coeffs.items() if abs(c['X']) > 1e-12)
        n_Y = sum(1 for p, c in pauli_coeffs.items() if abs(c['Y']) > 1e-12)
        log(f"      Qubit {qubit_idx} (occ={occupation}, sign={sign}): {n_Z} Z-terms, {n_I} I-terms, {n_X} X-terms, {n_Y} Y-terms")

        # Show a few sample coefficient pairs for debugging
        sample_count = 0
        for p, c in pauli_coeffs.items():
            if abs(c['Z']) > 1e-12 and abs(c['I']) > 1e-12 and sample_count < 3:
                # For complex coefficients, check if real parts have same sign
                same_sign = (c['Z'].real * c['I'].real > 0) if isinstance(c['Z'], complex) else (c['Z'] * c['I'] > 0)
                log(f"        Sample P={p}: c_Z={c['Z']:.6f}, c_I={c['I']:.6f}, same_sign={same_sign}")
                sample_count += 1

        # For occupation=0: H - O*(Z-1) = H - O*Z + O*I
        #   P⊗Z: c_Z -> c_Z - o_P
        #   P⊗I: c_I -> c_I + o_P
        # For occupation=1: H - O*(Z+1) = H - O*Z - O*I
        #   P⊗Z: c_Z -> c_Z - o_P
        #   P⊗I: c_I -> c_I - o_P

        # Use numerical optimization to find optimal killer coefficients
        optimal_o = optimize_killer_coefficients(pauli_coeffs, sign, n_restarts=3)

        HQ_reduced = QubitOperator()

        for P_rest, coeffs in pauli_coeffs.items():
            c_Z = coeffs['Z']
            c_I = coeffs['I']
            c_X = coeffs['X']
            c_Y = coeffs['Y']

            # Get optimal o_P from optimization (default to 0 if not in dict)
            o_P = optimal_o.get(P_rest, 0.0)

            # Apply the killer: H' = H - o_P * P ⊗ (Z + sign*I)
            new_c_Z = c_Z - o_P
            new_c_I = c_I + sign * o_P

            # Add terms to reduced Hamiltonian
            if abs(new_c_Z) > 1e-12:
                term_Z = tuple(sorted(list(P_rest) + [(qubit_idx, 'Z')], key=lambda x: x[0]))
                HQ_reduced += QubitOperator(term_Z, new_c_Z)

            if abs(new_c_I) > 1e-12:
                HQ_reduced += QubitOperator(P_rest, new_c_I)

            # X and Y terms are unchanged (killer doesn't affect them)
            if abs(c_X) > 1e-12:
                term_X = tuple(sorted(list(P_rest) + [(qubit_idx, 'X')], key=lambda x: x[0]))
                HQ_reduced += QubitOperator(term_X, c_X)

            if abs(c_Y) > 1e-12:
                term_Y = tuple(sorted(list(P_rest) + [(qubit_idx, 'Y')], key=lambda x: x[0]))
                HQ_reduced += QubitOperator(term_Y, c_Y)

        HQ_reduced.compress()
        norm_after = qubit_operator_1norm(HQ_reduced)
        reduction = norm_before - norm_after

        HQ_current = HQ_reduced
        if reduction > 1e-10:
            total_reduction += reduction
            killers_applied.append({
                'spatial_orbital': spatial_orbital,
                'qubit_idx': qubit_idx,
                'occupation': occupation,
                'reduction': reduction
            })

            log(f"      Orbital {spatial_orbital} (qubit {qubit_idx}, occ={occupation}): reduction={reduction:.6f}")

    final_norm = qubit_operator_1norm(HQ_current)
    log(f"    Final HQ 1-norm: {final_norm:.6f}")
    log(f"    Total reduction: {total_reduction:.6f} ({100*total_reduction/max(original_norm,1e-10):.1f}%)")

    return HQ_current, total_reduction, killers_applied

# load Q-SENSE basis states

molecule        = sys.argv[1]
bond_length     = float(sys.argv[2])
filename        = f'{molecule}_data/Uext_CSF_for_Praveen_Smik_{bond_length}.dump'
output_filename = f'main_outputs/{molecule}_{bond_length}_PT_parallel'

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


Hsub       = np.zeros([Nstates, Nstates], dtype=np.complex128)
sig_matrix = np.zeros([Nstates, Nstates], dtype=np.complex128)

for i in range(Nstates):
    with open(output_filename, 'a') as f:
        print(f'{i, i}', file=f)

    ket_f      = factorized_tapered_statevectors[i]
    ket_labels = UCSF_information[i][0]
    ket_config = configs[i]

    Htapered        = project_out_seniority_symmetries(Hqub, Nqubits, ket_config, ket_config)
    HQ, ketQ, _, NQ = evaluate_fully_classical_factors(ket_f, ket_f, ket_labels, ket_labels, Htapered)

    if NQ == 0:
        Hsub[i,i] = HQ.constant

    else:
        HQsparse   = get_sparse_operator(HQ)
        ketQ       = convert_dense_format_to_sparse_format(ketQ)
        Hsub[i,i] = (ketQ @ HQsparse @ ketQ.T)[0,0]

        HQ              -= HQ.constant
        HQ.compress()
        decomp = sorted_insertion_decomposition(HQ, 'fc')
        var_metric       = variance_of_decomp(decomp, ketQ, NQ, general=True)
        sig_matrix[i,i] = np.sqrt(var_metric)

def evaluate_off_diagonal(i, j, Hsub, factorized_tapered_statevectors, UCSF_information, configs, Hqub, Nqubits, output_filename, apply_killer=True):
    with open(output_filename, 'a') as f:
        print(f'{i, j}', file=f)

    ij_shift           = 0.5 * (Hsub[i,i] + Hsub[j,j])

    bra_f              = factorized_tapered_statevectors[i]
    bra_labels         = UCSF_information[i][0]
    bra_config         = configs[i]

    ket_f              = factorized_tapered_statevectors[j]
    ket_labels         = UCSF_information[j][0]
    ket_config         = configs[j]

    # First, extract quantum states to get full_Q_block
    _, _, NQ_check, full_Q_block = extract_quantum_states(bra_f, ket_f, bra_labels, ket_labels)

    Htapered           = project_out_seniority_symmetries(Hqub - ij_shift, Nqubits, bra_config, ket_config)
    HQ, braQ, ketQ, NQ = evaluate_fully_classical_factors(bra_f, ket_f, bra_labels, ket_labels, Htapered)

    if NQ == 0:
        matrix_element   = HQ.constant
        element_variance = 0

    else:
        HQsparse         = get_sparse_operator(HQ, NQ)
        braQ             = convert_dense_format_to_sparse_format(braQ)
        ketQ             = convert_dense_format_to_sparse_format(ketQ)
        matrix_element   = (braQ @ HQsparse @ ketQ.T)[0,0]

        # Apply fermion killer to reduce HQ 1-norm
        HQ_for_variance = HQ
        if apply_killer and full_Q_block:
            # Identify which qubits in NQ come from S_N for the ket state
            sn_qubits_info = identify_ket_sn_qubits_in_quantum_region(full_Q_block, ket_labels, ket_f)

            if sn_qubits_info:
                with open(output_filename, 'a') as f:
                    print(f'  Pair ({i},{j}): Found {len(sn_qubits_info)} S_N qubits in quantum region: {sn_qubits_info}', file=f)

                # Apply killers to reduce HQ
                HQ_reduced, total_reduction, killers_applied = apply_killers_to_hamiltonian(
                    HQ, NQ, sn_qubits_info, full_Q_block, verbose_file=output_filename
                )

                if total_reduction > 1e-10:
                    HQ_for_variance = HQ_reduced
                    with open(output_filename, 'a') as f:
                        print(f'  Pair ({i},{j}): Killer reduced 1-norm by {total_reduction:.6f}', file=f)

        comp             = create_composite_state(braQ, ketQ, NQ)
        HQ_aug           = XorY_augment(HQ_for_variance, NQ)
        decomp           = sorted_insertion_decomposition(HQ_aug, 'fc')
        var_metric       = variance_of_decomp(decomp, comp, NQ + 1, general=True)
        element_variance = np.sqrt(var_metric)

    return matrix_element, element_variance

ij_pairs = []
for i in range(Nstates):
    for j in range(Nstates):
        if i > j:
            ij_pairs.append((i,j))

def worker(args):
    i, j = args
    return evaluate_off_diagonal(i, j, Hsub, factorized_tapered_statevectors, UCSF_information, configs, Hqub, Nqubits, output_filename)

with mp.Pool() as pool:
    results = pool.map(worker, ij_pairs)

for k, (i,j) in enumerate(ij_pairs):
    matrix_element, element_variance = results[k]

    Hsub[i,j]       = matrix_element
    Hsub[j,i]       = matrix_element

    sig_matrix[i,j] = element_variance
    sig_matrix[j,i] = element_variance


vals, vecs = np.linalg.eigh(Hsub)
Egs        = vals[0]
c          = vecs[:,0]
cost       = sampling_cost(c, sig_matrix)

with open(output_filename, 'a') as f:
    print(f'''
        Final Results:
            Method              : {'PT'}
            Molecule            : {molecule}
            Bond Length         : {bond_length}
            Ground State Energy : {Egs}
            Sampling Cost       : {cost}
    ''', file=f)