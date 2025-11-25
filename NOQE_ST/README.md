# Shadow Tomography Enhanced Non-Orthogonal Quantum Eigensolver (NOQE)

Implementation of the shadow-tomography-enhanced NOQE algorithm from:

**"An Error Mitigated Non-Orthogonal Quantum Eigensolver via Shadow Tomography"**
by Hang Ren, Yipei Zhang, Wendy M. Billings, Rebecca Tomann, Nikolay V. Tkachenko, Martin Head-Gordon, and K. Birgitta Whaley (2025)

## Overview

This implementation provides a resource-efficient quantum algorithm for electronic structure calculations that offers significant advantages over the original NOQE method:

### Key Improvements

- **Measurement Scaling**: O(M) instead of O(M²) with number of reference states
- **Qubit Requirements**: N qubits instead of 2N+1 qubits
- **Circuit Depth**: Approximately halved (~g/2 instead of ~g gates)
- **Built-in Error Mitigation**: Natural integration with shadow distillation

### Algorithm Summary

1. **Reference State Preparation**: Prepare M non-orthogonal reference states using UHF + UCC ansatz
2. **Auxiliary State Construction**: Create auxiliary states |ψ^R⟩ and |ψ^I⟩ for each reference
3. **Shadow Tomography**: Perform randomized Clifford measurements to collect classical shadows
4. **Matrix Estimation**: Reconstruct H and S matrices from classical shadows using U-statistics
5. **Error Mitigation**: Apply shadow distillation to suppress noise-induced bias
6. **Eigenvalue Solution**: Solve generalized eigenvalue problem H·c = E·S·c

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
NOQE_ST/
├── main.py                  # Main implementation and example
├── shadow_tomography.py     # Classical shadow construction and estimation
├── reference_states.py      # Reference and auxiliary state preparation
├── matrix_estimation.py     # H and S matrix element estimation
├── error_mitigation.py      # Shadow distillation error mitigation
├── noqe_solver.py          # Generalized eigenvalue solver
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Usage

### Basic Example

```python
from main import ShadowNOQE
import numpy as np

# Initialize Shadow-NOQE
noqe = ShadowNOQE(
    num_qubits=4,           # Number of spin-orbitals
    num_references=2,        # Number of reference states
    num_shadows=10000,       # Classical shadows per state
    use_error_mitigation=True,
    mitigation_order=3
)

# Add reference states (UHF + UCC parameters)
uhf_occupation = [1, 1, 0, 0]
ucc_parameters = np.array([0.1])
excitations = [((0, 1), (2, 3))]
noqe.add_reference_state(uhf_occupation, ucc_parameters, excitations)

# Run NOQE
energies, eigenvectors = noqe.run(hamiltonian)
print(f"Ground state energy: {energies[0]} Hartree")
```

### Running the H2 Example

```bash
python main.py
```

This runs the H2 molecule example at the Coulson-Fischer point (R = 1.2 Å) using the STO-3G basis.

## Module Descriptions

### shadow_tomography.py

Implements classical shadows using random Clifford measurements:
- `ClassicalShadow`: Stores shadow measurement data
- `ShadowEstimator`: Estimates observables from shadows
- `generate_random_clifford_circuit()`: Creates random Clifford unitaries

Key equations (from paper):
- Shadow snapshot: ρ̂_{U,b} = (2^n + 1)(U† |b⟩⟨b| U) - I (Eq. 14-15)
- Linear estimation: ⟨O⟩ = Tr(O ρ̂) (Eq. 17)
- Bilinear estimation: Tr(O ρ_i ρ_j) (Eq. 21-24)

### reference_states.py

Prepares reference and auxiliary states:
- `ReferenceStatePreparation`: Creates UHF + UCC reference states
- `prepare_auxiliary_state_R()`: Constructs |ψ^R⟩ = (|0⟩^⊗n + |ψ⟩)/√2 (Eq. 19)
- `prepare_auxiliary_state_I()`: Constructs |ψ^I⟩ = (|0⟩^⊗n + i|ψ⟩)/√2 (Eq. 19)

### matrix_estimation.py

Estimates matrix elements from classical shadows:
- `estimate_overlap_element()`: Computes S_ij using Eq. 21-23
- `estimate_hamiltonian_element()`: Computes H_ij using Eq. 24
- `NOQEMatrixBuilder`: Constructs complete H and S matrices

Uses order-3 U-statistics for optimal sample complexity (Eq. 27-28).

### error_mitigation.py

Implements shadow distillation for error mitigation:
- `ShadowDistillation`: Applies ⟨O⟩_(m) = Tr(O ρ^m) / Tr(ρ^m) (Eq. 33)
- Suppresses noise exponentially: O(ε/(1-ε))^m (Eq. 34)
- No additional quantum overhead

### noqe_solver.py

Solves the generalized eigenvalue problem:
- `NOQESolver.solve()`: Solves H·c = E·S·c (Eq. 3)
- `ChemicalAccuracyChecker`: Validates results (1.6 mHa tolerance)

## Sample Complexity

From the paper's analysis:

### Original NOQE (Hadamard Test)
- Total measurements: O(M² · polylog(D)/ε²)
- Qubits: 2N+1
- Circuit depth: g gates

### Shadow-based NOQE (This Implementation)
- Total measurements: O(M · (1/ε² + D^(5/6)B^(1/6)/ε^(1/3)))
- Qubits: N (50% reduction)
- Circuit depth: ~g/2 (50% reduction)

Where:
- M = number of reference states
- N = number of qubits
- D = 2^N (Hilbert space dimension)
- ε = target precision
- B = bound on Tr(H²)

## Performance

For high-precision regime (ε ≤ 1/D), the sample complexity becomes:
- **O(M/ε²)** - independent of system size!

This makes shadow-based NOQE ideal for:
- Small to moderate quantum systems (N ≤ 20-30 qubits)
- High-precision requirements (chemical accuracy)
- Many reference states (large M)

## Key Features

1. **Resource Efficient**: Half the qubits and gates compared to original NOQE
2. **Measurement Efficient**: Linear scaling O(M) instead of quadratic O(M²)
3. **Error Resilient**: Built-in shadow distillation for noise suppression
4. **Scalable**: Sample complexity independent of system size in high-precision regime
5. **Modular Design**: Clean separation of concerns across modules

## Theoretical Background

### Classical Shadows (Section III)
Classical shadows provide an efficient way to estimate multiple properties of a quantum state from few measurements:
- Sample complexity: O(log(K)B/ε²) for K observables
- Much better than full tomography: O(4^N/ε²)

### U-Statistics Optimization (Section IV.C)
Using order-3 U-statistics for pure states:
- ρ̂^(U3) = (1/n(n-1)(n-2)) Σ_{α≠β≠γ} ρ̂_α ρ̂_β ρ̂_γ (Eq. 27)
- Achieves Heisenberg-limited scaling for suitable parameters

### Shadow Distillation (Section VI)
Error mitigation without quantum overhead:
- Exponential noise suppression: O((ε/(1-ε))^m)
- Outperforms Zero-Noise Extrapolation (ZNE)
- Maintains chemical accuracy under realistic noise

## Citation

If you use this implementation, please cite:

```bibtex
@article{ren2025shadow,
  title={An Error Mitigated Non-Orthogonal Quantum Eigensolver via Shadow Tomography},
  author={Ren, Hang and Zhang, Yipei and Billings, Wendy M. and Tomann, Rebecca and
          Tkachenko, Nikolay V. and Head-Gordon, Martin and Whaley, K. Birgitta},
  journal={arXiv preprint arXiv:2504.16008v2},
  year={2025}
}
```

## License

This implementation is provided for educational and research purposes.

## Future Extensions

Potential improvements and extensions:
1. Integration with real molecular Hamiltonians (PySCF, OpenFermion)
2. Adaptive shadow sampling based on variance estimates
3. Hardware-specific circuit optimizations
4. Parallel shadow collection
5. Extended to other quantum chemistry methods (k-UCC, ADAPT-VQE)

## Contact

For questions or issues, please refer to the original paper or open an issue in this repository.
