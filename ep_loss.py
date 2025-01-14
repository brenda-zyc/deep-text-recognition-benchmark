import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EPLoss(nn.Module):
    def __init__(self):
        super(EPLoss, self).__init__()

    def forward(self, pred, R, I, target):
        # pred: batch_size, num_steps, num_classes
        # target: batch_size, max_length+1=num_steps
        # R: batch_size, num_steps, 3  (C, I, D)
        # I: batch_size, num_steps, num_classes

        batch_size, n_y, num_classes = pred.size()  # num_steps (26?)
        n_T = target.size()[1]  # target中存的是每个position中的index， index在[0, num_classes-1]中  # max_length+1

        p_I = torch.zeros(batch_size, n_T, n_y).to(device)  # compute from R[:, :, 1]
        p_D = torch.zeros(batch_size, n_T, n_y).to(device)  # compute from R[:, :, 2]
        p_C = torch.zeros(batch_size, n_T, n_y).to(device)  # compute from R[:, :, 0]
        EOS = torch.tensor(1, dtype=torch.long).to(device)  # long (b/c 'eos' <-> 1)

        # compute p_C, p_I, p_D
        for j in range(n_y):
            R_C = R[:, j, 0]
            R_I = R[:, j, 1]
            R_D = R[:, j, 2].unsqueeze(1).repeat(1, n_y)  # 192, 26
            p_D[:, :, j] = torch.where(target == EOS, EOS.float(), R_D)  # num_batch, num_steps(n_T), num_steps(n_y)

            for i in range(n_T):
                target_onehot = F.one_hot(target[:, i], num_classes=num_classes)  # batch_size, num_classes
                pd_T, _ = torch.max(pred[:, j, :] * target_onehot, dim=1)  # batch_size
                I_T, _ = torch.max(I[:, j, :] * target_onehot, dim=1)  # batch_size

                p_C[:, i, j] = R_C * pd_T  # batch_size, n_T, n_y
                if j == n_y - 1:
                    p_I[:, i, j] = I_T  # batch_size, n_T, n_y
                else:
                    p_I[:, i, j] = R_I * I_T  # batch_size, n_T, n_y

        prev_row = torch.zeros(batch_size, n_y).to(device)
        prev_row[:, 0] = 1
        # todo: check the correctness of batch implementation, combine for loop above with for loop below
        for j in range(1, n_y):  # i=0, i.e. target has 0 tokens
            prev_row[:, j] = prev_row[:, j - 1] * p_D[:, 0, j]  # p_D is the matrix for p(D_ij)
        for i in range(1, n_T):
            prev_col = prev_row[:, 0] * p_I[:, i - 1, 0]  # (b)*(b) -> (b)
            for j in range(1, n_y):
                curr_col = prev_row[:, j - 1] * p_C[:, i - 1, j - 1] + prev_row[:, j] * p_I[:, i - 1,
                                                                                        j] + prev_col * p_D[:,
                                                                                                        i,
                                                                                                        j - 1]  # right shift prev_col
                prev_row[:, j - 1] = prev_col  # no longer need i-1, j-1, so update prev_row[j-1] to i, j-1
                prev_col = curr_col  # update prev col (i, j-1) -> (i, j)
            prev_row[:, n_y - 1] = prev_col
        loss = torch.log(prev_row[:, n_y - 1])  # n_T-1, n_y-1 (loss for y, T)

        return loss.mean()  # batch_size
