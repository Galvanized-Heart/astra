import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class AstraModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(-1, 5)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
# Example usage
""" 
trainer = L.Trainer()
trainer.fit(model, train_loader, valid_loader)

trainer.test(model, dataloaders=DataLoader(test_set)) 
"""