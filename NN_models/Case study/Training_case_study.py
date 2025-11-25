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

#Load dataset
df = pd.read_csv('data_case_study.csv')

tau = df['Residence Time (min)'].values
ratio = df_simulation['Tyrosine ratio'].values
k1 = df['k1'].values
k2 = df['k2'].values
area = df['area'].values
dadk1 = df['Sens_area_k1'].values
dadk2 = df['Sens_area_k2'].values

X = np.column_stack((tau, ratio, k1, k2))
y = area.reshape(-1, 1)

#Bounds used to produce the dataset
bounds = [(1,10), (0.05, 0.95), (0.01, 0.1), (0.01, 0.1)]

#Determine the derivative of the scaled values: df/dx * range(x)/range(f(x))
dadk1 = dadk1 * (bounds[2][1] - bounds[2][0])/(y.max()-y.min())
dadk2 = dadk2 * (bounds[3][1] - bounds[3][0])/(y.max()-y.min())

#Normalize X values 
X_scaled = np.column_stack(((X[:, 0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0]), 
                            (X[:, 1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0]),
                            (X[:, 2] - bounds[2][0]) / (bounds[2][1] - bounds[2][0]),
                            (X[:, 3] - bounds[3][0]) / (bounds[3][1] - bounds[3][0])))

#Normalize y vlues 
y_scaled = (y - y.min())/(y.max()-y.min())

#Normalize the derivative of the normalized values
dadk1_scaled = (dadk1 - dadk1.min(axis=0))/(dadk1.max(axis=0) - dadk1.min(axis=0))
dadk2_scaled = (dadk2 - dadk2.min(axis=0))/(dadk2.max(axis=0) - dadk2.min(axis=0))

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test, dadk1_train, dadk1_test, dadk2_train, dadk2_test = train_test_split(X_scaled, y_scaled, dadk1_scaled, dadk2_scaled,
                                                                                                      test_size=0.20, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32) 
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)  

dadk1_train_tensor = torch.tensor(dadk1_train, dtype=torch.float32).squeeze()  
dadk1_test_tensor = torch.tensor(dadk1_test, dtype=torch.float32).squeeze()  

dadk2_train_tensor = torch.tensor(dadk2_train, dtype=torch.float32).squeeze()  
dadk2_test_tensor = torch.tensor(dadk2_test, dtype=torch.float32).squeeze()  

# Training loop
num_epochs = 10000
patience = 50
best_loss = float('inf')
patience_counter = 0
    
model = Model()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
        
    # Forward pass
    y_pred = model(X_train_tensor)  
        
    # Compute loss
    loss = custom_loss(model, y_train_tensor, y_pred, dadk1_train_tensor, dadk2_train_tensor, dadk1, dadk2, X_train_tensor)
        
    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    model.eval()
        
    y_test_pred = model(X_test_tensor)
    test_loss = custom_loss(model, y_test_tensor, y_test_pred, dadk1_test_tensor, dadk2_test_tensor, dadk1, dadk2, X_test_tensor)
        
    # Early stopping
    if test_loss < best_loss:
        best_loss = test_loss
        patience_counter = 0
    else:
        patience_counter += 1
                
    if patience_counter >= patience:
        break
        
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}")
