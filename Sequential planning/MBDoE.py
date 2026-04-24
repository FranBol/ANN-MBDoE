import torch
import torch.nn as nn
import numpy as np
from scipy import optimize
from scipy.integrate import solve_ivp

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

def objective_function(p):
    k1 = p[0]
    k2 = p[1]
    k1_test = np.full(len(tau_test), k1)
    k2_test = np.full(len(tau_test), k2)    
    X = np.column_stack((tau_test, ratio_test, k1_test, k2_test))    
    X_scaled = np.column_stack(((X[:, 0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0]), 
                            (X[:, 1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0]),
                            (X[:, 2] - bounds[2][0]) / (bounds[2][1] - bounds[2][0]),
                            (X[:, 3] - bounds[3][0]) / (bounds[3][1] - bounds[3][0])))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    area_pred = (model(X_tensor).detach().numpy() * (15.02127404 - 0.008847216) + 0.008847216).T    # Unscale predicted area
    
    log_likelihood = -0.5 * np.sum(((a_test_noise - area_pred) ** 2) / var_exp + np.log(2 * np.pi) + np.log(var_exp))    
    
    return -log_likelihood

def objective_DoE(x):
    tau_new = x[0]
    ratio_new = x[1]
    
    # Append the current tau and ratio values to tau_test and ratio_test
    tau_test_extended = np.append(tau_test, tau_new)
    ratio_test_extended = np.append(ratio_test, ratio_new)
    
    # Evaluate the objective function
    obj = obj_D(tau_test_extended, ratio_test_extended, p, var_exp)
    
    return obj

def model_NN(tau, ratio, p):
    k1, k2 = p
    X = np.column_stack((tau, ratio, k1, k2))    
    X_scaled = np.column_stack(((X[:, 0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0]), 
                            (X[:, 1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0]),
                            (X[:, 2] - bounds[2][0]) / (bounds[2][1] - bounds[2][0]),
                            (X[:, 3] - bounds[3][0]) / (bounds[3][1] - bounds[3][0])))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y = model(X_tensor).detach().numpy()
    
    return y

def sensitivity(tau, ratio, p, model):
    # Define the sensitivity matrix
    s_matrix = np.zeros([len(tau), len(p)])

    for i in range(len(tau)):
        inputs = [tau[i], ratio[i]] + list(p)
        X = np.array(inputs).reshape(1, -1)      
        X_scaled = np.column_stack(((X[:, 0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0]), 
                            (X[:, 1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0]),
                            (X[:, 2] - bounds[2][0]) / (bounds[2][1] - bounds[2][0]),
                            (X[:, 3] - bounds[3][0]) / (bounds[3][1] - bounds[3][0])))
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
        # Extract tau, ratio, k1, k2 from the input X tensor
        tau_tensor = X_tensor[:, 0]
        ratio_tensor = X_tensor[:, 1]
        k1_tensor = X_tensor[:, 2].requires_grad_(True)
        k2_tensor = X_tensor[:, 3].requires_grad_(True)
       
        # Forward pass
        u = model(torch.stack([tau_tensor, ratio_tensor, k1_tensor, k2_tensor], dim=1))            
    
        #Compute gradients
        grad_k1 = torch.autograd.grad(u, k1_tensor, create_graph=True)[0]
        grad_k2 = torch.autograd.grad(u, k2_tensor, create_graph=True)[0]
    
        grad_k1 = grad_k1.detach().numpy()
        grad_k2 = grad_k2.detach().numpy()
        grad_k1 = grad_k1 * (15.02127404 - 0.008847216)/(bounds[2][1] - bounds[2][0])
        grad_k2 = grad_k2 * (15.02127404 - 0.008847216)/(bounds[3][1] - bounds[3][0])
                
        # Store the gradient in the sensitivity matrix
        s_matrix[i] = [grad_k1.item(), grad_k2.item()]
    
    return s_matrix

def FIM_fun(tau, ratio, p, var_exp):
    var_exp_inv = 1/var_exp
    s_matrix = sensitivity(tau, ratio, p, model) 
    FIM = np.zeros((len(s_matrix[0]), len(s_matrix[0])))

    for i in range(len(s_matrix)):
        FIM_i = var_exp_inv * np.outer(s_matrix[i], s_matrix[i])
        FIM += FIM_i
    
    return FIM

def cov_p_fun(tau, ratio, p, var_exp):
    FIM = FIM_fun(tau, ratio, p, var_exp)
    cov_p = np.linalg.inv(FIM)
    
    return cov_p

def obj_D(tau, ratio, p, var_exp):
    FIM = FIM_fun(tau, ratio, p, var_exp) 
    obj = np.linalg.det(np.linalg.inv(FIM))

    return obj






        

        
