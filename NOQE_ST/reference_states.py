"""
Reference State Preparation for NOQE
Implements UHF reference states and auxiliary states for shadow tomography
"""
import numpy as np
from typing import List, Tuple
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate, CXGate


class ReferenceStatePreparation:
    """Prepare reference states for NOQE"""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits

    def prepare_uhf_state(self, occupation: List[int]) -> QuantumCircuit:
        """
        Prepare unrestricted Hartree-Fock state

        Args:
            occupation: List of 0s and 1s indicating occupied orbitals

        Returns:
            Quantum circuit preparing UHF state
        """
        qr = QuantumRegister(self.num_qubits, 'q')
        qc = QuantumCircuit(qr)

        # Initialize occupied orbitals
        for i, occ in enumerate(occupation):
            if occ == 1:
                qc.x(qr[i])

        return qc

    def apply_ucc_ansatz(self, qc: QuantumCircuit,
                        parameters: np.ndarray,
                        excitations: List[Tuple]) -> QuantumCircuit:
        """
        Apply unitary coupled-cluster ansatz (simplified version)

        Args:
            qc: Base quantum circuit
            parameters: UCC parameters
            excitations: List of excitation tuples (occupied, virtual)

        Returns:
            Circuit with UCC ansatz applied
        """
        # Simplified UCCD implementation
        # In practice, use low-rank decomposition as described in paper
        for idx, (occ, virt) in enumerate(excitations):
            if idx < len(parameters):
                theta = parameters[idx]
                # Apply rotation for double excitation
                self._apply_double_excitation(qc, occ, virt, theta)

        return qc

    def _apply_double_excitation(self, qc: QuantumCircuit,
                                 occupied: Tuple[int, int],
                                 virtual: Tuple[int, int],
                                 theta: float):
        """
        Apply a rotation in the {|1100⟩, |0011⟩} subspace.

        Creates: cos(θ)|1100⟩ + sin(θ)|0011⟩

        This implementation works by:
        1. Undo the |1100⟩ preparation to get |0000⟩
        2. Create the desired superposition from |0000⟩
        """
        p, q = occupied  # qubits 0, 1
        r, s = virtual   # qubits 2, 3

        # Step 1: Undo the X gates that created |1100⟩ → back to |0000⟩
        qc.x(p)
        qc.x(q)

        # Step 2: Create superposition cos(θ)|0⟩ + sin(θ)|1⟩ on qubit p
        qc.ry(2 * theta, p)

        # After RY: cos(θ)|0⟩ + sin(θ)|1⟩ on qubit p
        # We want:
        #   cos(θ)|1100⟩ + sin(θ)|0011⟩
        # So: p=0 component (cos) → |1100⟩ → need to flip p,q to 1
        #     p=1 component (sin) → |0011⟩ → need to set r,s to 1

        # When p=1 (sin component), set r=1, s=1
        qc.cx(p, r)
        qc.cx(p, s)

        # When p=0 (cos component), we need p=1, q=1
        # First flip p unconditionally, then use it to control q
        qc.x(p)  # Now: sin(θ)|1⟩ + cos(θ)|0⟩ becomes sin(θ)|0⟩ + cos(θ)|1⟩
        qc.cx(p, q)  # When p=1 (cos component), set q=1

    def prepare_auxiliary_state_R(self, reference_circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Prepare auxiliary state |ψ^R⟩ = (|0⟩⊗n + |ψ⟩)/√2

        Uses an ancilla qubit to create the superposition:
        1. Create ancilla in |+⟩ state
        2. Apply reference state preparation controlled on ancilla = |1⟩
        3. Result: (|0⟩_anc ⊗ |0⟩^⊗n + |1⟩_anc ⊗ |ψ⟩)/√2

        The ancilla is qubit 0, system qubits are 1 to n.

        Args:
            reference_circuit: Circuit preparing reference state |ψ⟩

        Returns:
            Circuit preparing auxiliary state (n+1 qubits: 1 ancilla + n system)
        """
        n = self.num_qubits
        # Create circuit with ancilla (qubit 0) + system qubits (1 to n)
        qr_anc = QuantumRegister(1, 'anc')
        qr_sys = QuantumRegister(n, 'sys')
        qc = QuantumCircuit(qr_anc, qr_sys)

        # Step 1: Put ancilla in |+⟩ state
        qc.h(qr_anc[0])

        # Step 2: Apply controlled version of reference circuit
        # When ancilla=|1⟩, apply the reference state preparation to system qubits
        # This creates: (|0⟩_anc|0...0⟩ + |1⟩_anc|ψ⟩)/√2

        for instr, qargs, cargs in reference_circuit.data:
            # Get the controlled version of each gate
            ctrl_gate = instr.control(1)
            # Map original qubits to system register (offset by 1 for ancilla)
            new_qargs = [qr_anc[0]] + [qr_sys[qarg._index] for qarg in qargs]
            qc.append(ctrl_gate, new_qargs, cargs)

        return qc

    def prepare_auxiliary_state_I(self, reference_circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Prepare auxiliary state |ψ^I⟩ = (|0⟩⊗n + i|ψ⟩)/√2

        Same as R but with S gate on ancilla before controlled operations
        to introduce the i phase on the |1⟩ component.

        Args:
            reference_circuit: Circuit preparing reference state |ψ⟩

        Returns:
            Circuit preparing auxiliary state with phase (n+1 qubits)
        """
        n = self.num_qubits
        qr_anc = QuantumRegister(1, 'anc')
        qr_sys = QuantumRegister(n, 'sys')
        qc = QuantumCircuit(qr_anc, qr_sys)

        # Step 1: Put ancilla in |+⟩ state
        qc.h(qr_anc[0])

        # Step 2: Apply S gate to introduce i phase on |1⟩ component
        # This transforms |+⟩ = (|0⟩ + |1⟩)/√2 to (|0⟩ + i|1⟩)/√2
        qc.s(qr_anc[0])

        # Step 3: Apply controlled version of reference circuit
        for instr, qargs, cargs in reference_circuit.data:
            ctrl_gate = instr.control(1)
            new_qargs = [qr_anc[0]] + [qr_sys[qarg._index] for qarg in qargs]
            qc.append(ctrl_gate, new_qargs, cargs)

        return qc


class ReferenceStateManager:
    """Manage multiple reference states for NOQE"""

    def __init__(self, num_qubits: int, num_references: int):
        self.num_qubits = num_qubits
        self.num_references = num_references
        self.preparation = ReferenceStatePreparation(num_qubits)
        self.reference_circuits = []
        self.auxiliary_R_circuits = []
        self.auxiliary_I_circuits = []

    def add_reference(self, uhf_occupation: List[int],
                     ucc_parameters: np.ndarray,
                     excitations: List[Tuple]):
        """
        Add a reference state

        Args:
            uhf_occupation: UHF occupation vector
            ucc_parameters: UCC ansatz parameters
            excitations: Excitation configurations
        """
        # Prepare base UHF state
        qc_uhf = self.preparation.prepare_uhf_state(uhf_occupation)

        # Apply UCC ansatz
        qc_ref = self.preparation.apply_ucc_ansatz(
            qc_uhf.copy(), ucc_parameters, excitations
        )

        # Prepare auxiliary states
        qc_aux_R = self.preparation.prepare_auxiliary_state_R(qc_ref)
        qc_aux_I = self.preparation.prepare_auxiliary_state_I(qc_ref)

        self.reference_circuits.append(qc_ref)
        self.auxiliary_R_circuits.append(qc_aux_R)
        self.auxiliary_I_circuits.append(qc_aux_I)

    def get_all_circuits_for_shadows(self) -> List[QuantumCircuit]:
        """
        Get all circuits needed for shadow tomography

        Returns:
            List of all circuits (references + auxiliaries)
        """
        all_circuits = []
        all_circuits.extend(self.reference_circuits)
        all_circuits.extend(self.auxiliary_R_circuits)
        all_circuits.extend(self.auxiliary_I_circuits)
        return all_circuits
