import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from model.lstm_block import ConvLSTMCell
from torch.autograd import Variable

class PredNet(nn.Module):
    def __init__(self, R_channels, E_channels, R_up_channels, stack_sizes, R_stack_sizes, A_filter_sizes, Ahat_filter_sizes, R_filter_sizes, output_mode='error', start_eval=None):
        super(PredNet, self).__init__()
        self.r_channels = R_channels
        self.e_channels = E_channels
        self.r_up_channels = R_up_channels + (0, )
        self.n_layers = len(R_channels)
        self.output_mode = output_mode

        self.stack_sizes = stack_sizes
        self.num_layers = len(stack_sizes)
        assert len(R_stack_sizes) == self.num_layers
        self.R_stack_sizes = R_stack_sizes
        assert len(A_filter_sizes) == self.num_layers - 1
        self.A_filter_sizes = A_filter_sizes
        assert len(Ahat_filter_sizes) == self.num_layers
        self.Ahat_filter_sizes = Ahat_filter_sizes
        assert len(R_filter_sizes) == self.num_layers
        self.R_filter_sizes = R_filter_sizes

        default_output_modes = ['prediction', 'error']
        assert output_mode in default_output_modes, 'Invalid output_mode: ' + str(output_mode)

        self.start_eval = start_eval

        for i in range(self.n_layers):
            cell = ConvLSTMCell(self.e_channels[i] + self.r_channels[i] + self.r_up_channels[i], self.r_channels[i],
                                (3, 3))  # 6
            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.r_channels[i], 3, padding=1), nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)


        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            if l == 0:
                update_A = nn.Sequential(nn.Conv2d(6, 48, (3, 3), padding=1),
                                     self.maxpool)
            elif l == 1:
                update_A = nn.Sequential(nn.Conv2d(96, 96, (3, 3), padding=1),
                                     self.maxpool)

            elif l == 2:
                update_A = nn.Sequential(nn.Conv2d(192, 192, (3, 3), padding=1),
                                     self.maxpool)

            # if l == 0:
            else:
                update_A = nn.Sequential(nn.Conv2d(self.r_channels[l], self.r_channels[l + 1], (3, 3), padding=1), self.maxpool)

            setattr(self, 'update_A{}'.format(l), update_A)

        # self.reset_parameters()

    def get_initial_states(self, input_shape):
        '''
        input_shape is like: (batch_size, timeSteps, 1, x, y, z)
                         or: (batch_size, timeSteps, 1, x, y, z)
        '''

        init_x = input_shape[3]
        init_y = input_shape[4]
        base_initial_state = np.zeros((input_shape[0], 3))


        initial_states = []
        states_to_pass = ['R', 'c', 'E']
        layerNum_to_pass = {sta: self.n_layers for sta in states_to_pass}

        for sta in states_to_pass:
            for lay in range(layerNum_to_pass[sta]):
                downSample_factor = 2 ** lay
                x = init_x // downSample_factor
                y = init_y // downSample_factor
                # z = init_z // downSample_factor
                if sta in ['R', 'c']:
                    stack_size = self.R_stack_sizes[lay]
                elif sta == 'E':
                    stack_size = self.stack_sizes[lay] * 2
                elif sta == 'Ahat':
                    stack_size = self.stack_sizes[lay]
                output_size = stack_size * x * y  # flattened size
                reducer = np.zeros((input_shape[2], output_size))  # (3, output_size)
                initial_state = np.dot(base_initial_state, reducer)  # (batch_size, output_size)
                output_shape = (-1, stack_size, x, y)
                initial_state = Variable(torch.from_numpy(np.reshape(initial_state, output_shape)).float().cuda(),
                                         requires_grad=True)
                initial_states += [initial_state]

        return initial_states

    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()

    def predict(self, As):
        states = self.get_initial_states(As.shape)
        n = self.num_layers

        R_seq = states[:(n)]  # 1
        E_seq = states[(2 * n):(3 * n)]  # 1
        c_seq = states[(n):(2 * n)]  # 1
        batch_size, time_steps, c, height, width = As.shape
        frame_predictions = torch.zeros((self.start_eval, batch_size, height, width, c))

        for t in range(0, time_steps):
            A = As[:, t]

            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))  # ConvLSTM
                inputs = [R_seq[l], E_seq[l]]
                if l != self.n_layers - 1:
                    inputs.append(R_up)

                inputs = torch.cat(inputs, dim=1)
                R_next, c_next = cell(inputs, c_seq[l])

                R_seq[l] = R_next  # b, 3, 128, 160
                c_seq[l] = c_next  # (b, 3, 128, 160)x2

                if l > 0:
                    R_up = self.upsample(R_next)

            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])

                if l == 0:
                    frame_prediction = A_hat

                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E_current = torch.cat([pos, neg], 1)
                E_seq[l] = E_current
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    A = update_A(E_current)

        for t in range(0, self.start_eval):
            A = frame_prediction

            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))  # ConvLSTM
                inputs = [R_seq[l], E_seq[l]]
                if l != self.n_layers - 1:
                    inputs.append(R_up)

                inputs = torch.cat(inputs, dim=1)
                R_next, c_next = cell(inputs, c_seq[l])

                R_seq[l] = R_next
                c_seq[l] = c_next

                if l > 0:
                    R_up = self.upsample(R_next)

            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])

                if l == 0:
                    frame_prediction = A_hat

                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E_current = torch.cat([pos, neg], 1)
                E_seq[l] = E_current
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    A = update_A(E_current)

            frame_predictions[t] = frame_prediction.permute(0, 2, 3, 1)
        return frame_predictions

    def forward(self, As):
        states = self.get_initial_states(As.shape)
        n = self.num_layers

        R_seq = states[:(n)]  # 1
        E_seq = states[(2 * n):(3 * n)]  # 1
        c_seq = states[(n):(2 * n)]  # 1
        batch_size, time_steps, c, height, width = As.shape

        total_error = []
        for t in range(0, time_steps):
            A = As[:, t]

            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))  # ConvLSTM
                inputs = [R_seq[l], E_seq[l]]
                if l != self.n_layers - 1:
                    inputs.append(R_up)

                inputs = torch.cat(inputs, dim=1)
                R_next, c_next = cell(inputs, c_seq[l])

                R_seq[l] = R_next
                c_seq[l] = c_next

                if l > 0:
                    R_up = self.upsample(R_next)

            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])

                if l == 0:
                    frame_prediction = A_hat

                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E_current = torch.cat([pos, neg],1)
                E_seq[l] = E_current
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    A = update_A(E_current)

            if self.output_mode == 'error':
                if t == 0:
                    continue
                mean_error = torch.cat([torch.mean(e.view(e.size(0), -1), 1, keepdim=True) for e in E_seq], 1)
                total_error.append(mean_error)

        if self.output_mode == 'error':
            return torch.stack(total_error, 2)
        elif self.output_mode == 'prediction':
            for t in range(0, self.start_eval):
                A = frame_prediction

                for l in reversed(range(self.n_layers)):
                    cell = getattr(self, 'cell{}'.format(l))  # ConvLSTM
                    inputs = [R_seq[l], E_seq[l]]
                    if l != self.n_layers - 1:
                        inputs.append(R_up)

                    inputs = torch.cat(inputs, dim=1)
                    R_next, c_next = cell(inputs, c_seq[l])

                    R_seq[l] = R_next
                    c_seq[l] = c_next

                    if l > 0:
                        R_up = self.upsample(R_next)

                for l in range(self.n_layers):
                    conv = getattr(self, 'conv{}'.format(l))
                    A_hat = conv(R_seq[l])

                    if l == 0:
                        frame_prediction = A_hat

                    pos = F.relu(A_hat - A)
                    neg = F.relu(A - A_hat)
                    E_current = torch.cat([pos, neg], 1)
                    E_seq[l] = E_current
                    if l < self.n_layers - 1:
                        update_A = getattr(self, 'update_A{}'.format(l))
                        A = update_A(E_current)

            return frame_prediction

class SatLU(nn.Module):

    def __init__(self, lower=0, upper=255, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)


    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' ('\
            + 'min_val=' + str(self.lower) \
	        + ', max_val=' + str(self.upper) \
	        + inplace_str + ')'

if __name__ == '__main__':

    E_channels = (6, 96, 192, 384)
    R_channels = (3, 48, 96, 192)
    R_up_channels = (48, 96, 192)

    stack_sizes = (3, 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filter_sizes = (3, 3, 3)
    Ahat_filter_sizes = (3, 3, 3, 3)
    R_filter_sizes = (3, 3, 3, 3)

    inputs = torch.ones((2, 1, 3, 128, 128)).cuda()
    model = PredNet(R_channels, E_channels, R_up_channels, stack_sizes, R_stack_sizes, A_filter_sizes, Ahat_filter_sizes, R_filter_sizes, output_mode='prediction').cuda()
    states = model.get_initial_states((2, 5, 3, 128, 128))
    # print(model)

    x, a = model(inputs)

