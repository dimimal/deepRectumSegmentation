import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = (
                grad_output
                * 2
                * (target * self.union - self.inter)
                / (self.union * self.union)
            )
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

# def dice_loss(input, target):
#     """TODO: Docstring for dice_loss.
#     :returns: TODO

#     """
#     eps = 1e-8
#      # compute the actual dice score
#     dims = (1, 2, 3)
#     target = F.one_hot(target, 2)
#     target = target.permute(0, 3, 1, 2)
#     intersection = torch.sum(input * target, dims)
#     union = torch.sum(input + target, dims)

#     dsc = 2 * intersection / (union + eps)
    # return torch.mean(1. - dsc)

def dice_loss(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    # print(target.shape)
    # print(target)
    # print(target.dtype)

    target = target.type(torch.int64)
    target = F.one_hot(target, 2)
    target = target.permute(0, 3, 1, 2)

    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques = np.unique(target.cpu().numpy())
    assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    # TODO In order to work I have to adress it as a 2 class problem 0: bg, 1: fg
    probs = F.softmax(input, dim=1)
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)  # b,c,h
    num = torch.sum(num, dim=2)

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)  # b,c,h
    den1 = torch.sum(den1, dim=2)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)  # b,c,h
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2 * (num / (den1 + den2))
    dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg
    dice_total = 1. - (torch.sum(dice_eso) / dice_eso.size(0))  # divide by batch_sz

    return dice_total


# def dice_coeff(input, target):
#     """Dice coeff for batches"""
#     if input.is_cuda:
#         s = torch.FloatTensor(1).cuda().zero_()
#     else:
#         s = torch.FloatTensor(1).zero_()

#     for i, c in enumerate(zip(input, target)):
#         s = s + DiceCoeff().forward(c[0], c[1])

#     return s / (i + 1)


def dice_coeff(pred, target, reduce=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smooth = 1e-8

    # MEseure only the FG
    target = target.type(torch.int64)
    # TODO Expand for 3D Input volumetric data in future!
    target = F.one_hot(target, 2).type(torch.float32)
    target = target.permute(0, 3, 1, 2)

    pred = torch.softmax(pred, dim=1)
    pred = (pred>0.5).float()
    # pred = torch.argmax(pred, dim=1)
    num = pred * target  # b,c,h,w--p*g
    # print(num.shape)
    num = torch.sum(num, dim=3)  # b,c,h
    num = torch.sum(num, dim=2)

    den1 = pred * pred  # --p^2
    den1 = torch.sum(den1, dim=3)  # b,c,h
    den1 = torch.sum(den1, dim=2)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)  # b,c,h
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2 * (num / (den1 + den2))
    # print(dice.shape)
    dice_eso = dice[:, 1:]

    # print(dice_eso.shape)
    # num = pred.size(0)
    # m1 = pred.view(num, -1).float()  # Flatten
    # m2 = target.view(num, -1).float()  # Flatten
    # intersection = (m1 * m2).sum().float()
    # dice_score = (2.0 * intersection / (m1.sum() + m2.sum() + smooth))
    # print(dice_score.shape)
    if reduce:
        dice_total = torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
    else:
        dice_total = dice_eso
    return dice_total

def iou_metric(pred, target):
    """TODO: Docstring for iou_metric.
    :returns: TODO

    """
    SMOOTH = 1e-6
    pred = torch.argmax(pred, dim=1)
    pred = pred.type(torch.int64).detach().requires_grad_(False)
    target = target.type(torch.int64)

    intersection = (pred & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (pred | target).float().sum((1, 2))         # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    iou = torch.sum(iou) / pred.size(0)
    return iou
