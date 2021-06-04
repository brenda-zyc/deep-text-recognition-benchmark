import ep_loss
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn

if __name__ == '__main__':
    batch_size = 192
    num_classes = 38
    num_steps = 26

    myloss = ep_loss.EPLoss()
    pred = torch.rand(size=(batch_size, num_steps, num_classes))
    target = torch.randint(38, size=(batch_size, num_steps))
    R = torch.rand(size=(batch_size, num_steps, 3))
    I = torch.rand(size=(batch_size, num_steps, num_classes))

    loss = myloss(pred, R, I, target)