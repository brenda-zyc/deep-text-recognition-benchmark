import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F


class EPLoss(nn.Module):
    def __init__(self):
        super(EPLoss, self).init()

    def forward(self, pred, R, I, target):
        # pred: batch_size, num_steps, num_classes
        # target: batch_size, max_length+1
        # R: batch_size, num_steps, 3  (C, I, D)
        # I: batch_size, num_steps, num_classes
        batch_size = target.size[0]
        num_classes = pred.size[2]
        n_y = pred.size[1]  # todo
        n_T = target.size[1]  # todo  target中存的是每个position中的index， index在[0, num_classes-1]中
        p_I = torch.zeros(n_T, n_y)  # compute from R[:, :, 1]
        p_D = torch.zeros(n_T, n_y)      # compute from R[:, :, 2]
        p_C = torch.zeros(batch_size, n_T, n_y)  # compute from R[:, :, 0]
        # compute p_C
        for j in range(n_y):
            R_C = R[:, j, 0]  # assume only the first batch
            for i in range(n_T):
                target_onehot = F.one_hot(target[:, i], num_classes=num_classes)  # batch_size, num_classes
                pd_T, _ = torch.max(pred[:, j, :] * target_onehot, dim=1)  # batch_size
                p_C[:, i, j] = R_C * pd_T  # batch_size, n_T, n_y

        # compute p_I
        for j in range(n_y):
            R_I = R[:, j, 1]
            for i in range(n_T):
                target_onehot = F.one_hot(target[:, i], num_classes=num_classes)  # batch_size, num_classes
                I_T, _ = torch.max(I[:, j, :] * target_onehot, dim=1)  # batch_size
                if j == n_y-1:
                    p_I[:, i, j] = I_T  # batch_size, n_T, n_y
                else:
                    p_I[:, i, j] = R_I * I_T  # batch_size, n_T, n_y

        # compute p_D
        EOS = torch.tensor(1, dtype=torch.float)  # todo: float32 or 64?
        for j in range(n_y):
            R_D = R[:, j, 2].unsqueeze(1).repeat(1, num_classes)
            p_D[:, :, j] = torch.where(target.float() == EOS, EOS, R_D)  # num_batch, num_steps

        prev_row = torch.zeros(n_y)
        ep_matrix[0, 0] = 1
        # todo: below does not consider the batch, how to convert to batch operations??
        for j in range(1, n_y):  # i=0
            prev_row[j] = prev_row[j - 1] * p_D[0, j]  # p_D is the matrix for p(D_ij)
        for i in range(1, n_T):
            prev_col = prev_row[0] * p_I[i - 1, 1]
            for j in range(1, n_y):
                curr_col = prev_row[j - 1] * p_C[i - 1, j - 1] + prev_row[j] * p_I[i - 1, j] + prev_col * p_D[
                    i, j - 1]  # right shift prev_col
                prev_row[j - 1] = prev_col  # no longer need i-1, j-1, so update prev_row[j-1] to i, j-1
                prev_col = curr_col  # update prev col (i, j-1) -> (i, j)
            prev_row[n_y - 1] = prev_col
        loss = prev_row[n_y - 1]  # n_T-1, n_y-1 (loss for y, T)

        return loss
