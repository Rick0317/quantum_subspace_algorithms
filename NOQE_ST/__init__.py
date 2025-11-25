"""
Shadow Tomography Enhanced Non-Orthogonal Quantum Eigensolver (NOQE)

A resource-efficient quantum algorithm for electronic structure calculations
based on classical shadows and non-orthogonal reference states.

Reference:
    Ren et al., "An Error Mitigated Non-Orthogonal Quantum Eigensolver
    via Shadow Tomography" (2025)
"""

from .main import ShadowNOQE
from .shadow_tomography import (
    ClassicalShadow,
    ShadowEstimator,
    generate_random_clifford_circuit
)
from .reference_states import (
    ReferenceStatePreparation,
    ReferenceStateManager
)
from .matrix_estimation import (
    MatrixElementEstimator,
    NOQEMatrixBuilder
)
from .error_mitigation import (
    ShadowDistillation,
    ErrorMitigatedEstimator
)
from .noqe_solver import (
    NOQESolver,
    ChemicalAccuracyChecker
)

__version__ = "1.0.0"
__author__ = "Implementation based on Ren et al. (2025)"

__all__ = [
    # Main algorithm
    "ShadowNOQE",

    # Shadow tomography
    "ClassicalShadow",
    "ShadowEstimator",
    "generate_random_clifford_circuit",

    # Reference states
    "ReferenceStatePreparation",
    "ReferenceStateManager",

    # Matrix estimation
    "MatrixElementEstimator",
    "NOQEMatrixBuilder",

    # Error mitigation
    "ShadowDistillation",
    "ErrorMitigatedEstimator",

    # Solver
    "NOQESolver",
    "ChemicalAccuracyChecker",
]
