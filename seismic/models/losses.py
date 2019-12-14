import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight, size_average)

    def forward(self, logits, targets):
        targets = targets.type(torch.cuda.LongTensor)
        return self.loss(logits, targets)


class MultiDice(nn.Module):
    """
    Calculate Dice with averaging per classes and then per batch
    """
    def __init__(self,):
        super(MultiDice, self).__init__()

    def forward(self, outputs, targets):
        smooth = 1e-15
        prediction = outputs.softmax(dim=1)

        dices = []

        for val in range(1, 8):
            target = (targets == val).float().squeeze()
            ch_pred = prediction[:, val]
            intersection = torch.sum(ch_pred * target, dim=(1,2))
            union = torch.sum(ch_pred, dim=(1,2)) + torch.sum(target, dim=(1,2))
            dice_part = (2 * intersection + smooth) / (union + smooth)
            dices.append(dice_part.mean())
        dices = torch.Tensor(dices)
        # dices.append(dice_part.mean())
        # return torch.mean(dice_part) # shouldn't there be torch.mean(dices) ???
        return torch.mean(dices)
