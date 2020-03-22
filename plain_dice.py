import torch

def dice_loss(pred,target):
    numerator = 2 * torch.sum(pred * target)
    denominator = torch.sum(pred + target)
    return (numerator + 1) / (denominator + 1) #-1

