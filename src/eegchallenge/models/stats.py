import torch
import pytorch_lightning as pl

class OutputVariance(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.y = []

    def forward(self, x):
        return 0

    def test_step(self, batch, batch_idx):
        self.y.append(batch[1])
        return 0

    def on_test_epoch_end(self):
        self.log("output_variance", torch.var(torch.cat(self.y, dim = 0), correction = 0))
        self.y.clear()