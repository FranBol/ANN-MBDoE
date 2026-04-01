import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.model_selection import train_test_split

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_Xa = nn.Linear(2, 10)       
        self.fc2_Xa = nn.Linear(10, 1) 
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1_Xa(x))        
        Xa = self.fc2_Xa(x)  
        
        return Xa

def custom_loss(y_true, y_pred, Sk_true, X):
    
    loss_data = nn.MSELoss()(y_pred, y_true)    
    
    gradients = torch.autograd.grad(
        outputs=y_pred, 
        inputs=X, 
        grad_outputs=torch.ones_like(y_pred), 
        create_graph=True
    )[0]
    
    # Extract the gradient for k 
    du_dk = gradients[:, 1]
    
    # Normalize the gradients 
    du_dk_scaled = (du_dk - Sk.min()) / (Sk.max() - Sk.min())        
    
    loss_sen = nn.MSELoss()(du_dk_scaled, Sk_true) 
   
    return loss_data + loss_sen #0 * loss_sen when standard training

#Load dataset
df = pd.read_csv('data_example_dispersion.csv')

tau = df['Residence time (s)'].values
k = df['k (s^-1)'].values
Xa = df['Xa'].values
dXadk = df['dXadk'].values

bounds = [(10, 90), (1e-3, 0.1)] #tau (s), k (s-1)
X = np.column_stack((tau, k))
y = Xa.reshape(-1, 1)
Sk = dXadk.reshape(-1, 1)

#Determine the derivative of the scaled values: df/dx * range(x)/range(f(x))
Sk = Sk * (bounds[1][1] - bounds[1][0])

#Normalize X values 
X_scaled = np.column_stack(((X[:, 0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0]), 
                            (X[:, 1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0])))

#Normalize the derivative of the normalized values
Sk_scaled = (Sk - Sk.min(axis=0))/(Sk.max(axis=0) - Sk.min(axis=0))

# First split: Train (70%) and Temp (30%)
X_train, X_temp, y_train, y_temp, Sk_train, Sk_temp = train_test_split(
    X_scaled, y, Sk_scaled, test_size=0.30, random_state=42)

# Second split: Validation (15%) and Test (15%)
X_val, X_test, y_val, y_test, Sk_val, Sk_test = train_test_split(
    X_temp, y_temp, Sk_temp, test_size=0.50, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

Sk_train_tensor = torch.tensor(Sk_train, dtype=torch.float32).squeeze()  
Sk_val_tensor = torch.tensor(Sk_val, dtype=torch.float32).squeeze()  
Sk_test_tensor = torch.tensor(Sk_test, dtype=torch.float32).squeeze()   

# Training loop
num_epochs = 20000
patience = 500  
model     = Model()
optimizer = optim.Adam(model.parameters(), lr=0.01)
    
best_loss        = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(num_epochs):
    # train
    model.train()
    optimizer.zero_grad()
    X_train_tensor.requires_grad_(True)
    y_pred = model(X_train_tensor)
    loss = custom_loss(y_train_tensor, y_pred, Sk_train_tensor, X_train_tensor)
    loss.backward()
    optimizer.step()

    # validate
    X_val_tensor.requires_grad_(True)
    y_val_pred = model(X_val_tensor)
    val_loss = custom_loss(y_val_tensor, y_val_pred, Sk_val_tensor, X_val_tensor)
    val_loss_val = val_loss.item()
        
    if val_loss_val < best_loss:     
        best_loss        = val_loss_val
        patience_counter = 0
        best_model_state = copy.deepcopy(model.state_dict()) 
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stop at epoch {epoch}")
        break

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Train {loss.item():.6f} | Val {val_loss_val:.6f} | ")
    
model.load_state_dict(best_model_state)

# test 
model.eval()
X_test_tensor.requires_grad_(True)
y_test_pred = model(X_test_tensor)
test_loss = custom_loss(y_test_tensor, y_test_pred, Sk_test_tensor, X_test_tensor)
