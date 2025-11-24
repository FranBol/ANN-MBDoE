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
