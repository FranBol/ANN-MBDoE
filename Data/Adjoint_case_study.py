import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np

# ODEfunc definition 
class ODEfunc(nn.Module):
    def __init__(self, k1, k2, ratio, u, D, dz, duration):
        super().__init__()
        self.k1 = nn.Parameter(k1)
        self.k2 = nn.Parameter(k2)
        self.register_buffer('ratio', ratio)
        self.register_buffer('u', u)
        self.register_buffer('D', D)
        self.register_buffer('dz', dz)
        self.register_buffer('duration', duration)
 
    def square_wave(self, t):
        return 0.75 * torch.where(t <= self.duration, 1.0, 0.0)
 
    def forward(self, t, C):
        CA, CB, CR, CS = torch.split(C, 400, dim=0)
 
        dCAdt = torch.zeros_like(CA)
        dCBdt = torch.zeros_like(CB)
        dCRdt = torch.zeros_like(CR)
        dCSdt = torch.zeros_like(CS)
 
        # Inlet BCs
        injection = self.square_wave(t)
        CA = torch.cat([(self.ratio * injection).unsqueeze(0), CA[1:]])
        CB = torch.cat([((1 - self.ratio) * injection).unsqueeze(0), CB[1:]])
 
        # Outlet BCs (zero gradient)
        CA = torch.cat([CA[:-1], CA[-2].unsqueeze(0)]).clamp(min=0.0, max=10)
        CB = torch.cat([CB[:-1], CB[-2].unsqueeze(0)]).clamp(min=0.0, max=10)
        CR = torch.cat([CR[:-1], CR[-2].unsqueeze(0)]).clamp(min=0.0, max=10)
        CS = torch.cat([CS[:-1], CS[-2].unsqueeze(0)]).clamp(min=0.0, max=10)
 
        # Spatial derivatives
        dCAdz   = (CA[2:] - CA[:-2]) / (2 * self.dz)
        d2CAdz2 = (CA[2:] - 2*CA[1:-1] + CA[:-2]) / self.dz**2
        dCBdz   = (CB[2:] - CB[:-2]) / (2 * self.dz)
        d2CBdz2 = (CB[2:] - 2*CB[1:-1] + CB[:-2]) / self.dz**2
        dCRdz   = (CR[2:] - CR[:-2]) / (2 * self.dz)
        d2CRdz2 = (CR[2:] - 2*CR[1:-1] + CR[:-2]) / self.dz**2
        dCSdz   = (CS[2:] - CS[:-2]) / (2 * self.dz)
        d2CSdz2 = (CS[2:] - 2*CS[1:-1] + CS[:-2]) / self.dz**2
 
        # Reaction terms
        r1 = self.k1 * (CA[1:-1] + 1e-8) * (CB[1:-1] + 1e-8)
        r2 = self.k2 * (CB[1:-1] + 1e-8) * (CR[1:-1] + 1e-8)
 
        # Temporal derivatives
        dCAdt[1:-1] = self.D * d2CAdz2 - self.u * dCAdz - r1
        dCBdt[1:-1] = self.D * d2CBdz2 - self.u * dCBdz - r1 - r2
        dCRdt[1:-1] = self.D * d2CRdz2 - self.u * dCRdz + r1 - r2
        dCSdt[1:-1] = self.D * d2CSdz2 - self.u * dCSdz + r2
 
        return torch.cat([dCAdt, dCBdt, dCRdt, dCSdt])
  
# sensitivity via adjoint 
def compute_sensitivity_adjoint(tau, ratio, k1_val, k2_val):
    """
    Solves the PDE for one experimental condition using the torchdiffeq
    adjoint method, then backpropagates to get d(Area_Bf)/dk1 and d(Area_Bf)/dk2
    in a single backward pass 
    """
    # Reactor / flow parameters
    L = torch.tensor(11.32)
    Nz = 400
    dz = L / (Nz - 1)
    t_res = tau * 60.0
    u = L / t_res
    D_AB = 1e-9
    D = D_AB + (u**2 * (3.75e-4)**2) / (48 * D_AB)
    duration = torch.tensor(0.04 * t_res)
     
    k1 = torch.tensor(k1_val,  dtype=torch.float64, requires_grad=True)
    k2 = torch.tensor(k2_val,  dtype=torch.float64, requires_grad=True)
    ratio_t = torch.tensor(ratio,   dtype=torch.float64)
     
    func = ODEfunc(k1, k2, ratio_t, u, D, dz, duration)    
 
    # Time grid and initial conditions
    t_eval = torch.linspace(0, 3.0 * t_res, 2000, dtype=torch.float64)
    C0 = torch.zeros(4 * Nz, dtype=torch.float64)
 
    # Forward solve via adjoint
    sol = odeint(func, C0, t_eval, method='dopri5', adjoint_method='dopri5')
 
    CB_out = sol[:, Nz + Nz - 2]
    dt = t_eval[1] - t_eval[0]
    Area_Bf = torch.trapz(CB_out, dx=dt)
 
    # Single backward pass — gradients w.r.t. both k1 and k2 
    Area_Bf.backward()
 
    sens_k1 = func.k1.grad.item()
    sens_k2 = func.k2.grad.item()
 
    return sens_k1, sens_k2
