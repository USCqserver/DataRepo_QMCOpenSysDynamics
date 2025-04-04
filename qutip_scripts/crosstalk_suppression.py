import numpy as np
import qutip as qt
from typing import List, Callable
from functools import partial

### Define GHZ state circuit ###
def constant_pulse_func(t: float, t_change_points: List[float], Ω_val: float) -> float:
    idx = np.searchsorted(t_change_points, t, side='right') - 1
    return Ω_val if idx % 2 == 0 else 0.0

def constant_pulse(t_start_list: List[float], duration: float, Ω_val: float = None) -> Callable[[float], float]:
    if not t_start_list:
        raise ValueError("t_start_list must not be empty")
    if duration <= 0:
        raise ValueError("duration must be positive")

    if Ω_val is None:
        Ω_val = 1 / (4 * duration)

    t_end_list = [t_start + duration for t_start in t_start_list]
    t_change_points = sorted(t_start_list + t_end_list)

    return partial(constant_pulse_func, t_change_points=t_change_points, Ω_val=Ω_val)

# ZZ-crosstalk in Pauli-X basis
def create_xx_hamiltonian(nqubit, J):
    H = 0
    I = qt.qeye(2)
    X = qt.sigmax()
    
    for i in range(nqubit-1):
        op_list = [I] * nqubit
        op_list[i] = X
        op_list[i+1] = X
        
        H += 2*np.pi * J * qt.tensor(op_list)
    
    return H.to("csr")

def create_dd_sequence(nqubit):
    H = 0
    I = qt.qeye(2)
    Z = qt.sigmaz()

    dd_sequence_1 = [I] * nqubit
    dd_sequence_2 = [I] * nqubit
    
    for i in range(nqubit):
        if i % 2 == 0:
            dd_sequence_1[i] = Z
        else:
            dd_sequence_2[i] = Z

    H_dd_1 = qt.tensor(dd_sequence_1)
    H_dd_1 *= 2*np.pi
    H_dd_2 = qt.tensor(dd_sequence_2)
    H_dd_2 *= 2*np.pi

    return H_dd_1.to("csr"), H_dd_2.to("csr")

def create_circuit_hamiltonian(nqubit, J, circuit_time, dd_num, dd_time):
    H_zz = create_xx_hamiltonian(nqubit, J)
    H_dd_1, H_dd_2 = create_dd_sequence(nqubit)
    
    # Calculate free evolution time between two pulses
    tau = circuit_time / (dd_num*2)
    tau_f = tau/2 - dd_time
    t_dd_time_1 = np.array([tau_f + tau * (i - 1) for i in range(1, 2 * dd_num + 1)])
    t_dd_time_2 = t_dd_time_1 + tau / 2

    # Create pulse functions
    pulse_func_1 = constant_pulse(t_dd_time_1.tolist(), dd_time)
    pulse_func_2 = constant_pulse(t_dd_time_2.tolist(), dd_time)

    H_list = [H_zz]
    H_list.append([H_dd_1, pulse_func_1])
    H_list.append([H_dd_2, pulse_func_2])

    # perform measurements for every 200 ns
    tlist = [float(i) for i in range(0, int(circuit_time) + 1, 200)]

    H = qt.QobjEvo(H_list)
    return H.to("csr"), tlist

def create_lind_ops(nqubit, T1, T2):
    gamma_1 = 1 / T1
    gamma_2 = 1 / T2

    c_ops = []
    # relaxation (T1), which is (sigma_z + im * sigma_y) / 2 in Pauli-x basis
    for i in range(nqubit):
        op_list = [qt.qeye(2)] * nqubit
        op_list[i] = np.sqrt(gamma_1) * ((qt.sigmaz() + qt.sigmay()*1j)/2)
        c_ops.append(qt.tensor(op_list))
    
    # dephasing (T2), which is sigma_x in Pauli-x basis
    for i in range(nqubit):
        op_list = [qt.qeye(2)] * nqubit
        op_list[i] = np.sqrt(gamma_2) * qt.sigmax()

        c_ops.append(qt.tensor(op_list))

    return c_ops

### Define Simulation Parameters ###
nqubit = 6
J = 1e-4
dd_num = 25
dd_time = 10
tf = 5e3
T1 = 1e5
T2 = 5e4
num_trajectory = 1000
seed = 2369

H_free = create_xx_hamiltonian(nqubit, J)
t_free_list = np.linspace(0,1e4, 1001)
H, time_list = create_circuit_hamiltonian(nqubit, J, tf, dd_num, dd_time)
c_ops = create_lind_ops(nqubit, T1, T2)
psi0 = qt.tensor([qt.basis(2, 0)] * (nqubit))

# measurement
def measure_fidelity_me(t: float, rho: qt.Qobj):
    return np.abs(rho[0,0])

def measure_fidelity_mc(t: float, rho: qt.Qobj):
    return np.abs(rho.dag()[0,0]*rho[0,0])

mc_options = {"method":"adams", "order":2, "improved_sampling":True, "map":"parallel", "keep_runs_results": True, "max_step":1}
mc_run = qt.mcsolve(H, psi0.to("csr"), time_list, c_ops, seeds=seed, e_ops=measure_fidelity_mc, options=mc_options, ntraj=num_trajectory)
