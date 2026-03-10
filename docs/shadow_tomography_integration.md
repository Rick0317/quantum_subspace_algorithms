# Shadow Tomography Integration with Q-SENSE (PT Method)

## Overview

This document describes the integration of **classical shadow tomography** into the Q-SENSE Perturbation Theory (PT) pipeline for estimating Hamiltonian matrix elements. The implementation replaces the standard measurement-based approach (Hamiltonian partitioning + sorted insertion decomposition) with shadow tomography, following the ideas of Ren et al. (Ref [24] in the Q-SENSE paper) applied to the Q-SENSE framework.

## Background: Q-SENSE PT Method

The Q-SENSE (Quantum SENiority-based Subspace Expansion) algorithm constructs a subspace Hamiltonian matrix **H** from orthogonal basis states built using seniority operators. Each basis state factorizes into classically tractable parts and a quantum part requiring measurement:

- **Diagonal elements**: `H_ii = <phi_i^(Q) | h_ii | phi_i^(Q)>` (linear expectation values)
- **Off-diagonal elements**: `H_ij = <phi_i^(Q) | h_ij | phi_j^(Q)>` (bilinear, between different states)

The ground-state energy is obtained by diagonalizing the subspace Hamiltonian, and the sampling cost metric `epsilon^2 * M` quantifies measurement efficiency.

## Original Approach (`main_sampling_cost_PT_parallel.py`)

The original PT implementation:

1. **Diagonal elements**: Computes `<psi|H|psi>` exactly via sparse matrix-vector multiplication.
2. **Off-diagonal elements**: Uses the **extended swap test** formalism (Eq. 27-28 in the paper), which maps `<phi_mu|h|phi_nu>` to an expectation value of a composite state `|Phi> = (|0>|phi_mu> + |1>|phi_nu>)/sqrt(2)`. The Hamiltonian is augmented with an ancilla qubit via `XorY_augment`.
3. **Sampling cost**: Partitions the effective Hamiltonian into fully-commuting (FC) fragments via the sorted insertion (SI) algorithm, then computes variance-based sampling cost from `variance_of_decomp`.
4. **Constant-term optimization**: Shifts off-diagonal Hamiltonians by `ij_shift = 0.5*(H_ii + H_jj)` to minimize estimator variance (Section III.B.3 of the paper).

## Shadow Tomography Approach (`main_sampling_cost_PT_ST.py`)

The shadow tomography variant replaces the fragment-based measurement strategy with classical shadows. The key modifications are organized into four phases:

### Phase 1: Pre-computation and State Registration

- Extracts quantum states `|phi_i^(Q)>` for all matrix elements using the same Q-SENSE factorization pipeline (seniority tapering, classical/quantum partitioning).
- Registers **unique quantum states** to avoid redundant shadow collection when the same state appears in multiple matrix elements.
- Pre-computes all tapered Hamiltonians `HQ` and caches them (`HQ_cache`).

### Phase 2: Shadow Collection (Parallel)

- For each unique quantum state, collects `NUM_SHADOWS` classical shadows using random Clifford measurements.
- Shadows are stored in **compact form**: `(Clifford tableau, measurement outcome)` pairs, which are O(n^2) per shadow instead of O(4^n) for full unitary matrices.
- For off-diagonal elements, also prepares **|0> +/- |psi> superposition states** and collects their shadows:
  - `|+_i> = (|0> + |psi_i>)/sqrt(2)`
  - `|-_i> = (|0> - |psi_i>)/sqrt(2)`
- Shadow collection is parallelized across all available CPU cores via `multiprocessing.Pool`.

### Phase 3: Matrix Element Estimation

- **Diagonal elements** (`H_ii`): Uses `ShadowEstimator.estimate_linear()` -- averages `Tr(H * rho_hat)` over shadow snapshots. This is the standard linear shadow estimator.
- **Off-diagonal elements** (`H_ij`): Uses a **bilinear shadow estimation** approach with |0>+/-|psi> superpositions:
  ```
  H_ij = 2 * (Tr(rho_i^+ @ rho_j^+ @ H) + Tr(rho_i^- @ rho_j^- @ H)) - H_00
  ```
  where `rho_i^+` and `rho_i^-` are density matrices of the +/- superposition states. This is computed via `ShadowEstimator.estimate_bilinear()` which uses vectorized einsum operations for efficiency.
- Estimation is parallelized via `multiprocessing.Pool`.

### Phase 4: Exact Comparison

- Computes exact matrix elements via direct sparse matrix-vector products for validation.
- Outputs element-by-element comparison (ST estimate vs exact), error statistics, and energy comparison.

## Shadow Tomography Module (`shadow_tomography.py`)

Provides the core shadow tomography primitives:

### `ClassicalShadow`
- Manages shadow collection for a quantum state.
- `get_shadow_snapshot()`: Reconstructs a single shadow density matrix via the inverse channel: `rho_hat = (2^n + 1)(U^dag |b><b| U) - I`

### `ShadowEstimator`
- **`estimate_linear()`**: Estimates `Tr(O * rho)` from shadows -- standard linear estimator. Used for diagonal matrix elements.
- **`estimate_pauli_expectation()`**: Optimized estimator for Pauli string observables (lower variance than general observable estimation).
- **`estimate_bilinear()`**: Estimates `Tr(rho_i @ O @ rho_j)` from two sets of shadows. Uses vectorized numpy/einsum for batched trace computation over all shadow pairs. Supports median-of-means for robust estimation. Used for off-diagonal matrix elements.
- **`u_statistics_order3()`**: Order-3 U-statistics estimator for pure state density matrices.

### Simulation Helpers (in `main_sampling_cost_PT_ST.py`)
- `simulate_shadow_collection()`: Simulates quantum shadow collection classically -- generates random Cliffords, applies to state, samples measurement outcomes. Returns compact `(Clifford, outcome)` tuples.
- `reconstruct_shadow()` / `reconstruct_shadows()`: Converts compact shadow tuples back to full density matrices for estimation.

## Key Differences from Original

| Aspect | Original (PT Parallel) | Shadow Tomography (PT ST) |
|--------|----------------------|--------------------------|
| **Diagonal estimation** | Exact sparse multiplication | Linear shadow estimator |
| **Off-diagonal estimation** | Extended swap test + FC fragments | Bilinear shadow estimation with |0>+/-|psi> superpositions |
| **Sampling cost model** | Variance from fragment decomposition | Analytical variance scaling: O(2^n / N_shadows) diagonal, O(2^(2n) / N_shadows^2) off-diagonal |
| **Constant-term shift** | `ij_shift = 0.5*(H_ii + H_jj)` | Not used (unshifted Hamiltonian) |
| **Parallelism** | joblib `Parallel` for off-diagonal | `multiprocessing.Pool` for both shadow collection and estimation |
| **State reuse** | No caching | Unique state registration avoids redundant shadow collection |
| **New parameter** | N/A | `NUM_SHADOWS` (3rd CLI argument) controls accuracy vs cost tradeoff |

## Motivation

Shadow tomography reduces the measurement scaling with respect to the number of basis states from **quadratic to linear** (Ref [24]), since shadows collected for a state can be reused across all matrix elements involving that state. This is particularly beneficial for the PT method which has a large number of basis states (e.g., 166 for N2). The tradeoff is that classical post-processing cost may grow, and the variance of bilinear (off-diagonal) estimation scales exponentially with qubit count.
