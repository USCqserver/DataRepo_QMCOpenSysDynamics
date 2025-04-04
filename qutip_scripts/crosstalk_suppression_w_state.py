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

# ZZ-crosstalk
def create_zz_hamiltonian(nqubit, J):
    H = 0
    I = qt.qeye(2)
    
    for i in range(nqubit-1):
        op_list = [I] * nqubit
        op_list[i] = qt.sigmaz()
        op_list[i+1] = qt.sigmaz()
        
        H += 2*np.pi * J * qt.tensor(op_list)
    
    return H.to("csr")

def create_dd_sequence_x2(nqubit):
    I = qt.qeye(2)

    dd_sequence_X_1 = [I] * nqubit
    dd_sequence_X_2 = [I] * nqubit
    
    for i in range(nqubit):
        if i % 2 == 0:
            dd_sequence_X_1[i] = qt.sigmax()
        else:
            dd_sequence_X_2[i] = qt.sigmax()

    H_dd_X_1 = qt.tensor(dd_sequence_X_1)
    H_dd_X_2 = qt.tensor(dd_sequence_X_2)
    H_dd_X_1 *= 2*np.pi
    H_dd_X_2 *= 2*np.pi

    return H_dd_X_1.to("csr"), H_dd_X_2.to("csr")

def create_dd_sequence_xy4(nqubit):
    I = qt.qeye(2)

    dd_sequence_X_1 = [I] * nqubit
    dd_sequence_Y_1 = [I] * nqubit
    dd_sequence_X_2 = [I] * nqubit
    dd_sequence_Y_2 = [I] * nqubit
    
    for i in range(nqubit):
        if i % 2 == 0:
            dd_sequence_X_1[i] = qt.sigmax()
            dd_sequence_Y_1[i] = qt.sigmay()
        else:
            dd_sequence_X_2[i] = qt.sigmax()
            dd_sequence_Y_2[i] = qt.sigmay()

    H_dd_X_1 = qt.tensor(dd_sequence_X_1)
    H_dd_Y_1 = qt.tensor(dd_sequence_Y_1)
    H_dd_X_2 = qt.tensor(dd_sequence_X_2)
    H_dd_Y_2 = qt.tensor(dd_sequence_Y_2)

    H_dd_X_1 *= 2*np.pi
    H_dd_Y_1 *= 2*np.pi
    H_dd_X_2 *= 2*np.pi
    H_dd_Y_2 *= 2*np.pi

    return H_dd_X_1.to("csr"), H_dd_Y_1.to("csr"), H_dd_X_2.to("csr"), H_dd_Y_2.to("csr")

def create_circuit_hamiltonian_x2(nqubit, J, circuit_time, dd_num, dd_time):
    H_zz = create_zz_hamiltonian(nqubit, J)
    H_dd_X_1, H_dd_X_2 = create_dd_sequence_x2(nqubit)
    
    # Create the time list when dd is applied
    tau = circuit_time / (dd_num*2)
    tau_f = tau/2 - dd_time
    X_dd_time_1 = np.array([tau_f + tau * (i - 1) for i in range(1, 2 * dd_num + 1)])
    X_dd_time_2 = X_dd_time_1 + tau/2

    # Create pulse functions
    pulse_func_1 = constant_pulse(X_dd_time_1.tolist(), dd_time)
    pulse_func_2 = constant_pulse(X_dd_time_2.tolist(), dd_time)

    H_list = [H_zz]
    H_list.append([H_dd_X_1, pulse_func_1])
    H_list.append([H_dd_X_2, pulse_func_2])

    # perform measurements
    measurement_interval = int(circuit_time / dd_num)
    tlist = [float(i) for i in range(0, int(circuit_time) + 1, measurement_interval)]

    H = qt.QobjEvo(H_list)
    return H.to("csr"), tlist

def create_circuit_hamiltonian_xy4(nqubit, J, circuit_time, dd_num, dd_time):
    H_zz = create_zz_hamiltonian(nqubit, J)
    H_dd_X_1, H_dd_Y_1, H_dd_X_2, H_dd_Y_2 = create_dd_sequence_xy4(nqubit)
    
    # Create the time list when dd is applied
    tau = circuit_time / (dd_num*4)
    tau_f = tau/2 - dd_time
    X_dd_time_1 = np.array([tau_f + 2 * tau * (i - 1) for i in range(1, 2 * dd_num + 1)])
    Y_dd_time_1 = X_dd_time_1 + tau
    X_dd_time_2 = X_dd_time_1 + tau/2
    Y_dd_time_2 = Y_dd_time_1 + tau/2

    # Create pulse functions
    pulse_func_X_1 = constant_pulse(X_dd_time_1.tolist(), dd_time)
    pulse_func_Y_1 = constant_pulse(Y_dd_time_1.tolist(), dd_time)
    pulse_func_X_2 = constant_pulse(X_dd_time_2.tolist(), dd_time)
    pulse_func_Y_2 = constant_pulse(Y_dd_time_2.tolist(), dd_time)

    H_list = [H_zz]
    H_list.append([H_dd_X_1, pulse_func_X_1])
    H_list.append([H_dd_Y_1, pulse_func_Y_1])
    H_list.append([H_dd_X_2, pulse_func_X_2])
    H_list.append([H_dd_Y_2, pulse_func_Y_2])

    # perform measurements
    measurement_interval = int(circuit_time / dd_num)
    tlist = [float(i) for i in range(0, int(circuit_time) + 1, measurement_interval)]

    H = qt.QobjEvo(H_list)
    return H.to("csr"), tlist

def create_lind_ops(nqubit, T1, T2):
    gamma_1 = 1 / T1
    gamma_2 = 1 / T2

    c_ops = []
    # relaxation (T1)
    for i in range(nqubit):
        op_list = [qt.qeye(2)] * nqubit
        op_list[i] = np.sqrt(gamma_1) * qt.sigmam()
        c_ops.append(qt.tensor(op_list))
    
    # dephasing (T2)
    for i in range(nqubit):
        op_list = [qt.qeye(2)] * nqubit
        op_list[i] = np.sqrt(gamma_2) * qt.sigmaz()

        c_ops.append(qt.tensor(op_list))

    return c_ops

### Define Simulation Parameters ###
nqubit = 12
J = 1e-4
dd_num = 25
dd_time = 2
tf = 5e3
T1 = 1e5
T2 = 5e4
num_trajectory = 1000
seed = 4214

H_free = create_zz_hamiltonian(nqubit, J)
t_free_list = np.linspace(0,1e4, 1001)
H_dd_x2, x2_time_list = create_circuit_hamiltonian_x2(nqubit, J, tf, dd_num, dd_time)
H_dd_xy4, xy4_time_list = create_circuit_hamiltonian_xy4(nqubit, J, tf, dd_num, dd_time)
c_ops = create_lind_ops(nqubit, T1, T2)
psi0 = qt.w_state(nqubit)

# measurement
def measure_fidelity_me(t: float, rho: qt.Qobj):
    W_state_index = [2**i + 1 for i in range(nqubit)]
    fidelity = 0
    for i in W_state_index:
        for j in W_state_index:
            fidelity += rho[j-1,i-1]
    return np.abs(fidelity / nqubit)

def measure_fidelity_mc(t: float, rho: qt.Qobj):
    W_state_index = [2**i + 1 for i in range(nqubit)]
    fidelity = 0
    for i in W_state_index:
        for j in W_state_index:
            fidelity += rho.dag()[0,i-1]*rho[j-1,0]
    return np.abs(fidelity / nqubit)

mc_options = {"method":"adams", "order":2, "improved_sampling":True, "map":"parallel", "keep_runs_results": True, "max_step":1}
mc_run = qt.mcsolve(H_dd_x2, psi0.to("csr"), x2_time_list, c_ops, seeds=seed, e_ops=measure_fidelity_mc, options=mc_options, ntraj=num_trajectory)
