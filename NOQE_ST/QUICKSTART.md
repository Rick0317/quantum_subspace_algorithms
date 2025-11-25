# Quick Start Guide

## Installation

```bash
cd /Users/rick/QuantumSubspaceAlgorithms/NOQE_ST
pip install -r requirements.txt
```

## Run Example

```bash
python main.py
```

This will run the H2 molecule example and should produce output showing:
1. Shadow collection progress
2. Matrix estimation
3. Eigenvalue solution
4. Chemical accuracy check

## Run Tests

```bash
python test_implementation.py
```

This will verify all components are working correctly.

## Quick Example

```python
from main import ShadowNOQE
import numpy as np

# 1. Initialize
noqe = ShadowNOQE(
    num_qubits=4,
    num_references=2,
    num_shadows=1000
)

# 2. Define reference states
occupation = [1, 1, 0, 0]  # UHF occupation
ucc_params = np.array([0.1])  # UCC parameters
excitations = [((0, 1), (2, 3))]  # Double excitations

noqe.add_reference_state(occupation, ucc_params, excitations)

# 3. Run (with your Hamiltonian)
energies, _ = noqe.run(hamiltonian)

# 4. Get ground state
print(f"E₀ = {energies[0]:.6f} Hartree")
```

## Key Parameters

- `num_qubits`: Number of spin-orbitals in your system
- `num_references`: Number of UHF reference states (typically 2-6)
- `num_shadows`: Classical shadows per state (10³-10⁶)
  - More shadows → better accuracy
  - Paper uses 10⁶ for chemical accuracy
- `use_error_mitigation`: Enable shadow distillation (recommended: True)
- `mitigation_order`: Order for distillation (recommended: 3)

## Expected Performance

### Sample Complexity
For M reference states and precision ε:
- **High precision regime**: O(M/ε²) measurements
- **Moderate precision**: O(M·D/ε^(1/3)) measurements

### Resource Requirements
Compared to original NOQE:
- **50% fewer qubits**: N instead of 2N+1
- **50% fewer gates**: ~g/2 instead of g
- **Better scaling**: O(M) instead of O(M²)

## Troubleshooting

### Import Errors
Make sure you're in the NOQE_ST directory when running:
```bash
cd /Users/rick/QuantumSubspaceAlgorithms/NOQE_ST
python main.py
```

### Qiskit Issues
If Qiskit isn't installed:
```bash
pip install qiskit qiskit-aer
```

### Memory Issues
For large systems, reduce `num_shadows`:
```python
noqe = ShadowNOQE(num_shadows=1000)  # Instead of 10000
```

### Accuracy Issues
- Increase `num_shadows` (more measurements)
- Enable error mitigation: `use_error_mitigation=True`
- Add more reference states
- Check Hamiltonian is properly normalized

## Next Steps

1. **Custom Hamiltonians**: Replace placeholder Hamiltonian with real molecular data
2. **More References**: Add additional UHF configurations for better accuracy
3. **Optimize Parameters**: Tune UCC parameters using classical methods
4. **Scale Up**: Try larger molecules (up to ~20 qubits)

## Module Overview

```
main.py                   → Main algorithm (ShadowNOQE class)
shadow_tomography.py      → Classical shadows implementation
reference_states.py       → UHF + UCC state preparation
matrix_estimation.py      → H and S matrix construction
error_mitigation.py       → Shadow distillation
noqe_solver.py           → Eigenvalue problem solver
```

## Paper Reference

For theoretical details, see:
- Section IV: Shadow tomography integration
- Section V: Sample complexity analysis
- Section VI: Error mitigation (shadow distillation)
- Section VII: H2 validation example

## Support

For issues or questions about the implementation, refer to:
1. README.md for detailed documentation
2. test_implementation.py for usage examples
3. Original paper for theoretical background
