import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, eploss=False):
        # todo: add a parameter indicating whether it's ep or attention, only include R, I in the EP case
        super(Attention, self).__init__()
        self.eploss = eploss
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes, eploss=eploss)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)  # compute probability from hidden layers (pd)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        """input_char: batch_size,
        """
        input_char = input_char.unsqueeze(1)  # batch_size, 1
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)  # onehot_dim == num_classes?
        one_hot = one_hot.scatter_(1, input_char, 1)  # batch size, onehot_dim  (192 * 38)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input: sequence modeling output
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.  26 here

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),  # hidden state
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))  # cell state
        if self.eploss:
            R = torch.FloatTensor(batch_size, num_steps, 3).fill_(0).to(device)  # initialize with zeros
            I = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)  # todo： num_steps 对应encoder和decoder的长度？？

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                if self.eploss:
                    hidden, alpha, r, pd_chars = self.attention_cell(hidden, batch_H, char_onehots)
                    R[:, i, :] = r  # b, num_steps, 3
                    I[:, i, :] = pd_chars  # b, num_steps, num_classes
                else:
                    hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)

            probs = self.generator(output_hiddens)  # num_steps, hidden -> num_steps, num_classes

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                if self.eploss:
                    hidden, alpha, r, pd_chars = self.attention_cell(hidden, batch_H, char_onehots)
                    R[:, i, :] = r  # b, num_steps, 3
                    I[:, i, :] = pd_chars  # b, num_steps, num_classes
                else:
                    hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)

                probs_step = self.generator(hidden[0])  # b, hidden size -> b, num classes  y_{t} one hot
                probs[:, i, :] = probs_step  # b, num_steps, num_classes
                _, next_input = probs_step.max(1)  # [b]  (optimal class for each batch) convert back from one hot
                targets = next_input  # y_{t}
        # todo: return R, I
        # print("Prediction: {}".format(probs.size()))
        if self.eploss:
            return probs, R, I
        else:
            return probs # batch_size x num_steps x num_classes [pd for y]


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings, eploss=False):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)  # 256+25
        self.hidden_size = hidden_size
        self.eploss = eploss
        if eploss:
            # parameters for edit probability
            self.edit = nn.Linear(hidden_size, 3, bias=False)  # pd of edit operations
            self.insert = nn.Linear(hidden_size, num_embeddings, bias=False)  # pd of inserted characters

    def forward(self, prev_hidden, batch_H, char_onehots):
        # batch_H: batch_size, num_encoder, hidden_size
        # prev_hidden: batch_size, hidden_size  s_{t-1}
        # char_onehots: batch_size, num_classes (number of alphabets?)  y_{t-1}

        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)  # batch_size, num_steps, hidden_size
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)  # batch_size, 1, hidden_size
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size, num_encoder_step, 1 (aggregate information from encoder and previous hidden state)
        alpha = F.softmax(e, dim=1)  # weight for each encoder step     batch_size, num_encoder_steps, 1
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel <- (batch_size, 1, num_channels)
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)  # prev_hidden 为(h,c), concat_context为input, (batch_size, hidden_size)
        if self.eploss:
            r = F.softmax(self.edit(cur_hidden[0]), dim=1)  # batch_size, 3
            i = F.softmax(self.insert(cur_hidden[0]), dim=1)  # batch_size, num_classes
            return cur_hidden, alpha, r, i
        else:
            return cur_hidden, alpha