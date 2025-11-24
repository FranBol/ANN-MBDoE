import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

def custom_loss(model, y_true, Xa_pred, Sk_true, Sk, X):
    
    loss_data = nn.MSELoss()(Xa_pred, y_true)
    
    tau_tensor = X[:, 0]    
    k_tensor = X[:, 1].requires_grad_(True)
    
    # Forward pass
    u = model(torch.stack([tau_tensor, k_tensor], dim=1))    

    du_dk = torch.autograd.grad(u.sum(), k_tensor, create_graph=True)[0]

    # Adjust the gradients to obtain normalized gradients        
    du_dk_scaled =  (du_dk - Sk.min())/(Sk.max() - Sk.min())
        
    loss_sen = nn.MSELoss()(du_dk_scaled, Sk_true) 
   
    return  loss_data + loss_sen # 0*loss_sen when standard training

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
        
    Xa_pred = model(X_train_tensor)  
        
    loss = custom_loss(model, y_train_tensor, Xa_pred, Sk_train_tensor, Sk, X_train_tensor)
        
    loss.backward()
    optimizer.step()

    model.eval()
        
    Xa_test_pred = model(X_test_tensor)
    test_loss = custom_loss(model, y_test_tensor, Xa_test_pred, Sk_test_tensor, Sk, X_test_tensor)
        
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
