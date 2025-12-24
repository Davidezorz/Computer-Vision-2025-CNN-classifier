import torch
import torch.nn as nn
import torch.optim as optim
import copy
from utils.utils import getDevice
                                                                      

# ╭───────────────────────────────────────────────────────────────────────────╮
# │                       Validation Class (With Patience)                    │
# ╰───────────────────────────────────────────────────────────────────────────╯

class Validation:
    def __init__(self, data_loader, patience=5, min_delta=0.0):
        self.best_loss    = float('inf')
        self.best_weights = None
        self.data_loader  = data_loader
        
        # Early Stopping parameters
        self.patience     = patience
        self.min_delta    = min_delta
        self.counter      = 0
        self.early_stop   = False

    def update(self, criterion, model):
        # 1. Compute current validation loss
        val_loss, val_accuracy = self.computeLoss(criterion, model)
        
        # 2. Check if this is the best loss (minus a small delta threshold)
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0  # Reset counter on improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return val_loss, val_accuracy

    def getWeights(self):
        return self.best_weights

    @torch.no_grad()
    def computeLoss(self, criterion, model):
        device = next(model.parameters()).device
        running_loss = 0.0
        running_accuracy = 0.0
        
        for X, y in self.data_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            y_pred = logits.argmax(dim=-1)
            loss = criterion(logits, y)
            running_loss += loss.item()
            running_accuracy += (y_pred == y).float().mean().item()
            
        avg_loss   = running_loss / len(self.data_loader)
        avg_accuracy = running_accuracy / len(self.data_loader)
        return avg_loss, avg_accuracy





# ╭───────────────────────────────────────────────────────────────────────────╮
# │                              Train Function                               │
# ╰───────────────────────────────────────────────────────────────────────────╯

def train(model, train_loader, val_loader, 
          optim_class = optim.Adam, optim_opt = {},
          lr=1e-4, epochs=10, end_step=0.1, 
          log_interval=10, patience=5, device=None,
          use_amp=True):
      
    device = getDevice(device)                                                  # ◀─╮ Device
    print(f"Training on: {device}")                                             #   ╰ handling
    
    model.to(device)
    model.train()
    
    optimizer = optim_class(model.parameters(), lr=lr, **optim_opt)             # ◀── optimizer setup
    criterion = nn.CrossEntropyLoss()                                           # ◀── loss function setup

    lr_scheduler = optim.lr_scheduler.LinearLR(                                 # ◀─┬ Learning rate scheduler: 
        optimizer,                                                              #   │ Decays LR per batch
        start_factor=1.0,                                                       #   │
        end_factor=end_step,                                                    #   │
        total_iters=epochs * len(train_loader)                                  #   │
    )                                                                           #  ─╯

    losses = {'train': [], 'val': [], 
              'train_accuracy': [], 'val_accuracy': []}
    
    
    validation = Validation(val_loader, patience=patience)                      # Instantiation Validation
    val_loss = float('inf')

    current_loss     = 0.0                                                      # ◀─┬ Loop variables  
    current_accuracy = 0.0                                                      # ◀─┤ 
    steps_since_log  = 0                                                        # ◀─╯ 
    
    # Initialize Scaler for prevent gradients underflow and overflow            #    ╭ Determine if we should use AMP
    use_amp = (device != 'cpu') and use_amp                                     #  ◀─╯ (Automatic Mixed Precision), and
    scaler = torch.amp.GradScaler(device, enabled=use_amp)                      #  ◀─╯ we ensures it does nothing on CPU
    
    print(f"{' ':<8} {' ':<8} {'Train':<8} {'Val':<8} ")
    print(f"{'Epoch':<8} {'Batch':<8} {'Loss':<8} {'Loss':<8} {'Patience':<10}") # set the print format
    print("-" * 80)

    # ---------- actual training starts ----------
    for epoch in range(epochs):
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()                                               # ◀─┬ Reset gradients
            
            with torch.amp.autocast(device_type=device, dtype=torch.float16,    # ◀─┬ This automatically casts operations 
                                    enabled=use_amp):                           #   ╰ to float16 where safe (not on CPU)
                logits = model(X)                                               # ◀─┬ Compute the predictions
                y_pred = logits.argmax(dim=-1)
                loss = criterion(logits, y)                                     # ◀─╯ Compute the loss 
            
            scaler.scale(loss).backward()                                       # ◀─┬ Compute gradients
            scaler.step(optimizer)                                              # ◀─╯ Update parameters
            scaler.update()                                                     # ◀── Adjusts the scale factor
            lr_scheduler.step()                                                    # ◀── Update learning rate
        
            current_loss     += loss.item()                                     # ◀─┬ Update running 
            current_accuracy += (y_pred == y).float().mean().item()             # ◀─┤ trackers
            steps_since_log  += 1                                               # ◀─╯ 

            # ---------- Validation & Logging Step ----------
            if (i + 1) % log_interval == 0:                                     # VALIDATION and LOSS storing
                avg_train_loss = current_loss / steps_since_log
                avg_train_accuracy = current_accuracy / steps_since_log
                
                model.eval()                                                    #   ╭ Run Validation
                val_loss, val_accuracy = validation.update(criterion, model)    # ◀─┤ correctly
                model.train()                                                   #   ╰╯
                
                losses['train'].append(avg_train_loss)                          # ◀─┬ Store metrics
                losses['val'].append(val_loss)                                  # ◀─╯
                losses['train_accuracy'].append(avg_train_accuracy) 
                losses['val_accuracy'].append(val_accuracy) 
            
                current_loss     = 0.0                                          # ◀─┬ Reset running trackers
                current_accuracy = 0.0                                          # ◀─┤
                steps_since_log  = 0                                            # ◀─╯
                
                val_counter = validation.counter
                print(f"\r{epoch+1:<8} {i+1:<8} {avg_train_loss:.5f}", end="")
                print(f"  {val_loss:.5f}  {val_counter}/{patience}   ", end="")

                if validation.early_stop:                                       # ◀─┬ Check Early Stopping
                    print(f"\nEarly stopping at Epoch {epoch+1}, Batch {i+1}.") #   ╭ Break batch loop
                    break                                                       # ◀─╯
        
        if validation.early_stop:                                               # Break epoch loop if early stop triggered
            break
            
    # Final load of best weights
    if validation.getWeights() is not None:
        model.load_state_dict(validation.getWeights())
        print(f"\nTraining complete. Best Validation Loss: {validation.best_loss:.5f}")
    
    return losses