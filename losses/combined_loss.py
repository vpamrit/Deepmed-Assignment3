import torch


class CombinedLoss(nn.Module):
    def __init__(self, losses=[], weights=None):
        super(CombinedLosses, self).__init__()

        if weights == None:
            weights = [1] * len(losses)

        self.weights = [weight / sum(weights) for weight in weights]
        self.losses = losses

    def forward(self, net_output, gt):
        mloss = torch.zeros([1])

        for i in range(len(losses)):
            mloss += self.losses[i](net_output, gt) * self.weights[i]

        return mloss


