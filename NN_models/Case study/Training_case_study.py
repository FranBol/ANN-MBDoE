import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

def custom_loss(model, y_true, y_pred, dadk1_true, dadk2_true, dadk1, dadk2, X):
    
    loss_data = nn.MSELoss()(y_pred, y_true)
    
    tau_tensor = X[:, 0]   
    ratio_tensor = X[:, 1]
    k1_tensor = X[:, 2].requires_grad_(True)
    k2_tensor = X[:, 3].requires_grad_(True)
    
    # Forward pass
    u = model(torch.stack([tau_tensor, ratio_tensor, k1_tensor, k2_tensor], dim=1))    

    du_dk1 = torch.autograd.grad(u.sum(), k1_tensor, create_graph=True)[0]
    du_dk2 = torch.autograd.grad(u.sum(), k2_tensor, create_graph=True)[0]

    # Adjust the gradients to obtain normalized gradients        
    du_dk1_scaled =  (du_dk1 - dadk1.min())/(dadk1.max() - dadk1.min())
    du_dk2_scaled =  (du_dk2 - dadk2.min())/(dadk2.max() - dadk2.min())
        
    loss_sen = nn.MSELoss()(torch.stack([du_dk1_scaled, du_dk2_scaled]), torch.stack([dadk1_true, dadk2_true]))
   
    return  loss_data + loss_sen #0*loss_sen when standard training
