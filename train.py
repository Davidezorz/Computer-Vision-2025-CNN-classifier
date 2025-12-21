import torch
import torch.nn as nn
import torch.optim as optim
import copy




# ╭───────────────────────────────────────────────────────────────────────────╮
# │                             Validation class                              │
# ╰───────────────────────────────────────────────────────────────────────────╯

class Validation():
    
    def __init__(self, X_val, y_val):
        self.best_loss    = float('inf')
        self.best_weights = None
        self.X_val        = X_val
        self.y_val        = y_val
    

    def update(self, criterion, model):
        loss = self.computeLoss(criterion, model).item()
        if loss < self.best_loss:
            state_dict = model.state_dict()
            self.best_weights = copy.deepcopy(state_dict)    # Deep copy the model's state dictionary into RAM
            self.best_loss = loss
        return loss


    def getWeights(self):
        return self.best_weights


    @ torch.no_grad()
    def computeLoss(self, criterion, model):
        pred = model(self.X_val)
        loss = criterion(pred, self.y_val)
        return loss
    




# ╭───────────────────────────────────────────────────────────────────────────╮
# │                              Train function                               │
# ╰───────────────────────────────────────────────────────────────────────────╯

def train(model, trainLoader, X_val, y_val, 
          end_step=0.1, lr=1e-4, epochs=10, step=10):
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0,                     # starts at 100 % of the LR 
        end_factor=end_step,                  # end    at    _% of the LR
        total_iters=epochs*len(trainLoader)
    )

    losses = {'train': [], 'val': []}
    validation = Validation(X_val, y_val)
    val_loss = float('inf')

    for i in range(epochs):
        current_loss, steps_done = 0, 1

        for j, (x_train, y_train) in enumerate(trainLoader):
            optimizer.zero_grad()

            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()

            optimizer.step()
            scheduler.step()
        
            # store losses
            current_loss += loss.item()
            steps_done += 1
            if (j+1) % step == 0:
                model.eval()
                with torch.no_grad():
                    losses['train'].append(current_loss/steps_done)
                    current_loss, steps_done = 0, 1

                    val_loss = validation.update(criterion, model)
                    losses['val'].append(val_loss)
                model.train()

        
        print(f"\repoch: {i+1} / {epochs}", end=' ')
        print(f"loss: {loss.item(): .5f} \t valLoss: {val_loss} \tlr: {scheduler.get_last_lr()}", end=' '*20)
    
    validation.update(criterion, model)

    if validation.getWeights() is not None:
            model.load_state_dict(validation.getWeights())
            print(f"\nTraining complete. Best Validation Loss: {validation.best_loss:.5f}")
    return losses