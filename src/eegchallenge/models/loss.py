import torch
import torch.nn as nn

class NRMSELoss(nn.Module): # Issue with this - Can probably remove
    """
    Normalized Root Mean Squared Error (NRMSE) loss function.
    """

    def __init__(self, loss_params=None):
        super(NRMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.loss_params = loss_params

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = self.mse_loss(input, target)
        rmse = torch.sqrt(mse)
        target_std = torch.std(target)
        nrmse = rmse / (target_std)
        if target_std < 1e-6:
            print(f"WARNING: Very small target_std: {target_std.item():.2e}, target range: [{target.min().item():.3f}, {target.max().item():.3f}], target shape: {target.shape}")
        return nrmse

# TODO: Implement loss that can be used as regularization?