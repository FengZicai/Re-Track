import torch
import torch.nn as nn
import torch.nn.functional as F


class DPCEncoderDecoder(nn.Module):

    def __init__(self, n_filters=[64, 256], layer_sizes=[1024, 2048 * 3], filter_sizes=[1], strides=[1],
                 bneck_size=128, input_size=2048):
        super(DPCEncoderDecoder, self).__init__()

        n_filters.append(bneck_size)
        n_layers = len(n_filters)
        self.input_size = input_size
        self.bneck_size = bneck_size

        self.conv_1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=filter_sizes[0], stride=strides[0])
        self.conv_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=filter_sizes[0], stride=strides[0])
        self.conv_3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=filter_sizes[0], stride=strides[0])

        self.conv1_1 = nn.Conv1d(in_channels=144, out_channels=1024, kernel_size=filter_sizes[0], stride=strides[0])

        layer_sizes.insert(0, self.bneck_size)
        self.out_size = int(layer_sizes[n_layers - 1] / 3)

        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=6144)

        self.globalAvgPool2048 = nn.AvgPool1d(2048, stride=1)
        self.globalAvgPool8 = nn.AvgPool1d(8, stride=1)

        self.fc1_0_1 = nn.Linear(in_features=128, out_features=8)
        self.fc1_0_2 = nn.Linear(in_features=8, out_features=128)

        self.fc2_0_1 = nn.Linear(in_features=128, out_features=8)
        self.fc2_0_2 = nn.Linear(in_features=8, out_features=128)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, non_linearity=F.relu, regularizer=None, weight_decay=0.001,
                dropout_prob=None, pool=F.avg_pool1d, pool_sizes=None, padding='same',
                verbose=False, closing=None):

        x0_0 = self.conv_1(x)
        non_x0_0 = non_linearity(x0_0)
        c0_0, _ = torch.max(non_x0_0, dim=2)

        x1_0 = self.conv_2(non_x0_0)
        x1_0_original_out = x1_0
        x1_0 = self.globalAvgPool2048(x1_0)
        x1_0 = x1_0.view(x1_0.size(0), -1)
        x1_0 = self.fc1_0_1(x1_0)
        x1_0 = non_linearity(x1_0)
        x1_0 = self.fc1_0_2(x1_0)
        x1_0 = self.sigmoid(x1_0)
        x1_0 = x1_0.view(x1_0.size(0), x1_0.size(1), 1)
        x1_0 = x1_0 * x1_0_original_out
        non_x1_0 = non_linearity(x1_0)
        c1_0, _ = torch.max(non_x1_0, dim=2)

        x2_0 = self.conv_3(non_x1_0)
        x2_0_original_out = x2_0
        x2_0 = self.globalAvgPool2048(x2_0)
        x2_0 = x2_0.view(x2_0.size(0), -1)
        x2_0 = self.fc2_0_1(x2_0)
        x2_0 = non_linearity(x2_0)
        x2_0 = self.fc2_0_2(x2_0)
        x2_0 = self.sigmoid(x2_0)
        x2_0 = x2_0.view(x2_0.size(0), x2_0.size(1), 1)
        x2_0 = x2_0 * x2_0_original_out
        non_x2_0 = non_linearity(x2_0)
        c2_0, _ = torch.max(non_x2_0, dim=2)

        c1_1_part = self.fc1(c2_0)
        non_c1_1 = non_linearity(c1_1_part)
        x1_1 = self.conv1_1(torch.cat((c1_0, non_c1_1), 1).view(c1_0.shape[0], -1, 8))
        non_x1_1 = non_linearity(x1_1)
        c1_1, _ = torch.max(non_x1_1, dim=2)

        c0_2 = self.fc2(c1_1)
        output = c0_2.view(c0_2.shape[0], 3, self.out_size)
        encoded = c2_0
        return output, encoded
