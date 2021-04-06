import torch.nn as nn
import torch.nn.functional as F


class Base(nn.Module):
    def __init__(self, state_size, num_frame=2):
        super(Base, self).__init__()
        out_channels = [16, 32]
        in_channels = [num_frame, out_channels[0]]
        padding_sizes = [2, 0]
        kernel_sizes = [8, 4]
        stride_sizes = [4, 2]
        self.conv = nn.ModuleList(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride_size,
                      padding=padding_size) for in_channel, out_channel, kernel_size, stride_size, padding_size in
            zip(in_channels, out_channels, kernel_sizes, stride_sizes, padding_sizes))
        size0, size1 = state_size

        self.feature = nn.Sequential()
        for i in range(len(self.conv)):
            self.feature.add_module("conv{}".format(i), self.conv[i])
            bn = nn.BatchNorm2d(num_features=out_channels[i])
            self.feature.add_module("bn{}".format(i), bn)

        for i in range(len(self.conv)):
            size0 = (size0 + 2 * padding_sizes[i] - kernel_sizes[i]) // stride_sizes[i] + 1
            size1 = (size1 + 2 * padding_sizes[i] - kernel_sizes[i]) // stride_sizes[i] + 1

        self.fc_size = out_channels[-1] * size0 * size1
        self.fc = nn.Linear(self.fc_size, 256, bias=False)

    def _init_params(self):
        for param in self.parameters():
            if param.dim() < 2:
                param.data.zero_()
            else:
                nn.init.xavier_uniform_(param)


class Actor(Base):

    def __init__(self, state_size, action_size, num_frame=2):
        super(Actor, self).__init__(state_size, num_frame)
        self.output = nn.Linear(256, action_size, bias=False)
        self._init_params()

    def forward(self, state):
        x = state
        x = self.feature(x)
        x = F.relu(self.fc(x.view(-1, self.fc_size)))
        log_probs = F.log_softmax(self.output(x), dim=1)
        return log_probs


class Critic(Base):

    def __init__(self, state_size, num_frame=2):
        super(Critic, self).__init__(state_size, num_frame)
        self.output = nn.Linear(256, 1, bias=False)
        self._init_params()

    def forward(self, state):
        x = state
        x = self.feature(x)
        x = F.relu(self.fc(x.view(-1, self.fc_size)))
        values = self.output(x)
        return values


class Actor_Critic(Base):

    def __init__(self, state_size, action_size, num_frame):
        super(Actor_Critic, self).__init__(state_size, num_frame)
        self.actor = nn.Linear(256, action_size, bias=False)
        self.critic = nn.Linear(256, 1, bias=False)
        self._init_params()

    def forward(self, state):
        x = state
        # for layer in self.conv:
        #     x = F.relu(layer(x))
        x = self.feature(x)
        x = F.relu(self.fc(x.view(-1, self.fc_size)))
        log_probs = F.log_softmax(self.actor(x), dim=1)
        values = self.critic(x)
        return log_probs, values
