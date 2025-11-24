import numpy as np
from scipy.integrate import solve_ivp

# Square wave input signal
def square_wave(t, duration, amplitude):
    return amplitude if t <= duration else 0

def pfr_transient(t, C, duration, amplitude, k1, k2, ratio, D, u, dz):
    CA, CB, CR, CS = np.split(C, 4)
   
    dCAdz = np.zeros_like(CA)
    d2CAdz2 = np.zeros_like(CA)
    dCAdt = np.zeros_like(CA)
   
    dCBdz = np.zeros_like(CB)
    d2CBdz2 = np.zeros_like(CB)
    dCBdt = np.zeros_like(CB)
   
    dCRdz = np.zeros_like(CR)
    d2CRdz2 = np.zeros_like(CR)
    dCRdt = np.zeros_like(CR)

    dCSdz = np.zeros_like(CS)
    d2CSdz2 = np.zeros_like(CS)
    dCSdt = np.zeros_like(CS)
   
    # Boundary conditions at z = 0 (inlet)
    CA[0] = ratio * square_wave(t, duration, amplitude)
    CB[0] = (1-ratio) * square_wave(t, duration, amplitude)
    CR[0] = 0
    CS[0] = 0
   
    # Interior points
    dCAdz[1:-1] = (CA[2:] - CA[:-2]) / (2 * dz)
    d2CAdz2[1:-1] = (CA[2:] - 2 * CA[1:-1] + CA[:-2]) / (dz**2)
    dCBdz[1:-1] = (CB[2:] - CB[:-2]) / (2 * dz)
    d2CBdz2[1:-1] = (CB[2:] - 2 * CB[1:-1] + CB[:-2]) / (dz**2)
    dCRdz[1:-1] = (CR[2:] - CR[:-2]) / (2 * dz)
    d2CRdz2[1:-1] = (CR[2:] - 2 * CR[1:-1] + CR[:-2]) / (dz**2)
    dCSdz[1:-1] = (CS[2:] - CS[:-2]) / (2 * dz)
    d2CSdz2[1:-1] = (CS[2:] - 2 * CS[1:-1] + CS[:-2]) / (dz**2)
   
    # Zero-flux boundary condition at z = L
    CA[-1] = CA[-2]
    CB[-1] = CB[-2]
    CR[-1] = CR[-2]
    CS[-1] = CS[-2]
   
    # Compute dC/dt using the PDE
    dCAdt[1:-1] = D * d2CAdz2[1:-1] - u * dCAdz[1:-1] - k1 * CA[1:-1] * CB[1:-1]
    dCBdt[1:-1] = D * d2CBdz2[1:-1] - u * dCBdz[1:-1] - k1 * CA[1:-1] * CB[1:-1] - k2 * CB[1:-1] * CR[1:-1]
    dCRdt[1:-1] = D * d2CRdz2[1:-1] - u * dCRdz[1:-1] + k1 * CA[1:-1] * CB[1:-1] - k2 * CB[1:-1] * CR[1:-1]
    dCSdt[1:-1] = D * d2CSdz2[1:-1] - u * dCSdz[1:-1] + k2 * CB[1:-1] * CR[1:-1]
   
    return np.concatenate([dCAdt, dCBdt, dCRdt, dCSdt])

def run_simulation(tau, ratio, k1, k2):
        
    #Reactor and operation conditions
    L = 11.32  # Length of reactor (m)
    d_t = 7.5e-4 #reactor diameter (m)
    
    t_res = tau*60
    u =L/t_res
    D_AB = 1e-9  # Dispersion coefficient
    D = D_AB + ((u**2*(d_t/2)**2)/(48*D_AB))
    
    #Simulatin parameters
    Nz = 1000  # Number of spatial grid points
    dz = L/(Nz-1) #spatial grid distance
    t_simulation = 3.0 * t_res #Total time
    Nt = 1000  # Number of time steps for plotting
    t_eval = np.linspace(0, t_simulation, Nt)  #Time points for evaluation
    
    #Model parameters
    k1 = k1
    k2 = k2
    ratio = ratio
    amplitude = 0.75 #Total concentration (CA0 + CB0)
    duration = (0.04  * t_res)
    
    #Initial conditions
    C0 = np.concatenate([np.zeros(Nz), np.zeros(Nz), np.zeros(Nz), np.zeros(Nz)])
    
    #Solve system
    sol = solve_ivp(pfr_transient, [0, t_simulation], C0, t_eval=t_eval, method='RK45', args=(duration, amplitude, k1, k2, ratio, D, u, dz))
           
    #Extract CB at outlet
    CA_sol, CB_sol, CR_sol, CS_sol = np.split(sol.y, 4)
    CA_out = CA_sol[-1, :]
    CB_out = CB_sol[-1, :]
    CR_out = CR_sol[-1, :]
             
    #Compute Area_Bf
    dt = t_simulation / Nt
    Area_Bf = np.trapz(CB_out, dx = dt)
        
    return Area_Bf

