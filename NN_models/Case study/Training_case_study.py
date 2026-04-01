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

def custom_loss(y_true, y_pred, dadk1_true, dadk2_true, X):
    
    loss_data = nn.MSELoss()(y_pred, y_true)    
    
    gradients = torch.autograd.grad(
        outputs=y_pred, 
        inputs=X, 
        grad_outputs=torch.ones_like(y_pred), 
        create_graph=True
    )[0]
    
    # Extract the gradient for k1 and k2 
    du_dk1 = gradients[:, 2]    
    du_dk2 = gradients[:, 3]
    
    # Normalize the gradients 
    du_dk1_scaled = (du_dk1 - dadk1.min()) / (dadk1.max() - dadk1.min())   
    du_dk2_scaled = (du_dk2 - dadk2.min()) / (dadk2.max() - dadk2.min()) 
    
    loss_sen = nn.MSELoss()(torch.stack([du_dk1_scaled, du_dk2_scaled]), torch.stack([dadk1_true, dadk2_true]))
   
    return loss_data + loss_sen # 0 * loss_sen when standard training

#Load dataset
df = pd.read_csv('data_case_study.csv')

tau = df['Residence Time (min)'].values
ratio = df['Tyrosine ratio'].values
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

# First split: Train (70%) and Temp (30%)
X_train, X_temp, y_train, y_temp, dadk1_train, dadk1_temp, dadk2_train, dadk2_temp = train_test_split(
    X_scaled, y_scaled, dadk1_scaled, dadk2_scaled,
    test_size=0.30, random_state=42
)

# Second split: Validation (15%) and Test (15%)
X_val, X_test, y_val, y_test, dadk1_val, dadk1_test, dadk2_val, dadk2_test = train_test_split(
    X_temp, y_temp, dadk1_temp, dadk2_temp,
    test_size=0.50, random_state=42
)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

dadk1_train_tensor = torch.tensor(dadk1_train, dtype=torch.float32).squeeze()
dadk1_val_tensor = torch.tensor(dadk1_val, dtype=torch.float32).squeeze()
dadk1_test_tensor = torch.tensor(dadk1_test, dtype=torch.float32).squeeze()

dadk2_train_tensor = torch.tensor(dadk2_train, dtype=torch.float32).squeeze()
dadk2_val_tensor = torch.tensor(dadk2_val, dtype=torch.float32).squeeze()
dadk2_test_tensor = torch.tensor(dadk2_test, dtype=torch.float32).squeeze()

# Training loop
num_epochs = 10000
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
    loss = custom_loss(y_train_tensor, y_pred, dadk1_train_tensor, dadk2_train_tensor, X_train_tensor)
    loss.backward()
    optimizer.step()

    # validate
    X_val_tensor.requires_grad_(True)
    y_val_pred = model(X_val_tensor)
    val_loss = custom_loss(y_val_tensor, y_val_pred, dadk1_val_tensor, dadk2_val_tensor, X_val_tensor)
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
        print(f" Epoch {epoch:5d} | Train {loss.item():.6f} | Val {val_loss_val:.6f} | ")

model.load_state_dict(best_model_state)

# test 
model.eval()
X_test_tensor.requires_grad_(True)
y_test_pred = model(X_test_tensor)
test_loss = custom_loss(y_test_tensor, y_test_pred, dadk1_test_tensor, dadk2_test_tensor, X_test_tensor)
