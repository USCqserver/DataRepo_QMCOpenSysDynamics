import numpy as np
import qutip as qt
from functools import partial

### Define GHZ state circuit ###
def constant_pulse_func(t, t_start, t_end, val):
    return val * ((t_start <= t) & (t < t_end))

def constant_pulse(t_start: float, val: float, duration: float):
    t_end = t_start + duration
    return partial(constant_pulse_func, t_start=t_start, t_end=t_end, val=val)

def create_zz_hamiltonian(nqubit, J):
    H = 0
    I = qt.qeye(2)
    Z = qt.sigmaz()
    
    for i in range(nqubit-1):
        op_list = [I] * nqubit
        op_list[i] = Z
        op_list[i+1] = Z
        
        H += 2*np.pi * J * qt.tensor(op_list)
    
    return H.to("csr")

def create_calib_hamiltonian(nqubit, i):
    H = 0
    I = qt.qeye(2)
    X = qt.sigmax()
    Z = qt.sigmaz()

    op_list_1 = [I] * nqubit
    op_list_1[i] = Z

    op_list_2 = [I] * nqubit
    op_list_2[i+1] = X
    
    op_list_3 = [I] * nqubit

    H += qt.tensor(op_list_1) + qt.tensor(op_list_2) - qt.tensor(op_list_3)
    H *= 2*np.pi

    return H.to("csr")

def create_gate_hamiltonian(nqubit, i):
    H = 0
    I = qt.qeye(2)
    X = qt.sigmax()
    Z = qt.sigmaz()

    op_list = [I] * nqubit
    op_list[i] = Z
    op_list[i+1] = X

    H+= qt.tensor(op_list)
    H *= 2*np.pi

    return H.to("csr")

def create_circuit_hamiltonian(nqubit, J, calib_time, gate_time):
    H_zz = create_zz_hamiltonian(nqubit, J)
    
    amp = 1 / (calib_time * 8)
    omega = 1 / (gate_time * 8)
    total_gate_time = calib_time + gate_time

    H_list = [H_zz]
    tlist = []

    for i in range(nqubit - 1):
        calib_start = i * total_gate_time
        gate_start = calib_start + calib_time

        H_calib = create_calib_hamiltonian(nqubit, i)
        H_gate = create_gate_hamiltonian(nqubit, i)

        H_list.append([H_calib, constant_pulse(calib_start, -amp, calib_time)])
        H_list.append([H_gate, constant_pulse(gate_start, omega, gate_time)])

    tf = (nqubit - 1) * total_gate_time
    # perform measurements for every 10 ns
    tlist = [float(i) for i in range(0, int(tf) + 1, 10)]
    tlist.sort()

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
nqubit = 6
J = 1e-4
calib_time = 10.0
gate_time = 50.0
T1 = 1e5
T2 = 5e4
num_trajectory = 500
seed = 2345

H, time_list = create_circuit_hamiltonian(nqubit, J, calib_time, gate_time)
c_ops = create_lind_ops(nqubit, T1, T2)
vp = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
psi0 = qt.tensor([vp] + [qt.basis(2, 0)] * (nqubit-1))

# measurement
def measure_fidelity_me(t: float, rho: qt.Qobj):
    return (rho[0,0] + rho[0,-1] + rho[-1,0] + rho[-1,-1]) * 0.5

def measure_fidelity_mc(t: float, rho: qt.Qobj):
    return (rho.dag()[0,0]*rho[0,0] + rho.dag()[0,-1]*rho[0,-1] + rho.dag()[-1,0]*rho[-1,0] + rho.dag()[-1,-1]*rho[-1,-1]) * 0.5

mc_options = {"method":"adams", "order":2, "improved_sampling":True, "map":"parallel", "keep_runs_results": True}
mc_run = qt.mcsolve(H, psi0.to("csr"), time_list, c_ops, seeds=seed, e_ops=measure_fidelity_mc, options=mc_options, ntraj=num_trajectory)
