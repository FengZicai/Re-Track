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

        self.conv0_0 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=filter_sizes[0], stride=strides[0])
        setattr(self, 'conv_{}_{}'.format(0, 0), self.conv0_0)
        self.conv1_0 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=filter_sizes[0], stride=strides[0])
        setattr(self, 'conv_{}_{}'.format(1, 0), self.conv1_0)
        self.conv2_0 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=filter_sizes[0], stride=strides[0])
        setattr(self, 'conv_{}_{}'.format(2, 0), self.conv2_0)
        self.conv1_1 = nn.Conv1d(in_channels=256, out_channels=1024, kernel_size=filter_sizes[0], stride=strides[0])
        setattr(self, 'conv_{}_{}'.format(1, 1), self.conv1_1)
        self.conv0_1 = nn.Conv1d(in_channels=464, out_channels=2048, kernel_size=filter_sizes[0], stride=strides[0])
        setattr(self, 'conv_{}_{}'.format(0, 1), self.conv0_1)
        self.conv0_2 = nn.Conv1d(in_channels=2320, out_channels=6144, kernel_size=filter_sizes[0], stride=strides[0])
        setattr(self, 'conv_{}_{}'.format(0, 2), self.conv0_2)

        layer_sizes.insert(0, self.bneck_size)
        self.out_size = int(layer_sizes[n_layers - 1] / 3)

        self.fc0_1 = nn.Linear(in_features=256, out_features=1792)
        setattr(self, 'fc_{}_{}'.format(0, 1), self.fc0_1)
        self.fc1_1 = nn.Linear(in_features=128, out_features=1792)
        setattr(self, 'fc_{}_{}'.format(1, 1), self.fc1_1)
        self.fc0_2 = nn.Linear(in_features=1024, out_features=7168)
        setattr(self, 'fc_{}_{}'.format(0, 2), self.fc0_2)

        self.globalAvgPool2048 = nn.AvgPool1d(2048, stride=1)
        setattr(self, 'globalAvgPool_{}'.format(2048), self.globalAvgPool2048)
        self.globalAvgPool4 = nn.AvgPool1d(4, stride=1)
        setattr(self, 'globalAvgPool_{}'.format(4), self.globalAvgPool4)
        self.globalAvgPool8 = nn.AvgPool1d(8, stride=1)
        setattr(self, 'globalAvgPool_{}'.format(8), self.globalAvgPool8)

        self.fc0_0_1 = nn.Linear(in_features=64, out_features=4)
        setattr(self, 'fc_{}_{}_{}'.format(0, 0, 1), self.fc0_0_1)
        self.fc0_0_2 = nn.Linear(in_features=4, out_features=64)
        setattr(self, 'fc_{}_{}_{}'.format(0, 0, 2), self.fc0_0_2)

        self.fc1_0_1 = nn.Linear(in_features=256, out_features=16)
        setattr(self, 'fc_{}_{}_{}'.format(1, 0, 1), self.fc1_0_1)
        self.fc1_0_2 = nn.Linear(in_features=16, out_features=256)
        setattr(self, 'fc_{}_{}_{}'.format(1, 0, 2), self.fc1_0_2)

        self.fc2_0_1 = nn.Linear(in_features=128, out_features=8)
        setattr(self, 'fc_{}_{}_{}'.format(2, 0, 1), self.fc2_0_1)
        self.fc2_0_2 = nn.Linear(in_features=8, out_features=128)
        setattr(self, 'fc_{}_{}_{}'.format(2, 0, 2), self.fc2_0_2)

        self.fc0_1_1 = nn.Linear(in_features=2048, out_features=128)
        setattr(self, 'fc_{}_{}_{}'.format(0, 1, 1), self.fc0_1_1)
        self.fc0_1_2 = nn.Linear(in_features=128, out_features=2048)
        setattr(self, 'fc_{}_{}_{}'.format(0, 1, 2), self.fc0_1_2)

        self.fc1_1_1 = nn.Linear(in_features=1024, out_features=64)
        setattr(self, 'fc_{}_{}_{}'.format(1, 1, 1), self.fc1_1_1)
        self.fc1_1_2 = nn.Linear(in_features=64, out_features=1024)
        setattr(self, 'fc_{}_{}_{}'.format(1, 1, 2), self.fc1_1_2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, non_linearity=F.relu, regularizer=None, weight_decay=0.001,
                dropout_prob=None, pool=F.avg_pool1d, pool_sizes=None, padding='same',
                verbose=False, closing=None):

        x0_0 = self.conv0_0(x)
        x0_0_original_out = x0_0
        x0_0 = self.globalAvgPool2048(x0_0)
        x0_0 = x0_0.view(x0_0.size(0), -1)
        x0_0 = self.fc0_0_1(x0_0)
        x0_0 = non_linearity(x0_0)
        x0_0 = self.fc0_0_2(x0_0)
        x0_0 = self.sigmoid(x0_0)
        x0_0 = x0_0.view(x0_0.size(0), x0_0.size(1), 1)
        x0_0 = x0_0 * x0_0_original_out
        non_x0_0 = non_linearity(x0_0)
        c0_0, _ = torch.max(non_x0_0, dim=2)

        x1_0 = self.conv1_0(non_x0_0)
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

        c0_1_part = self.fc0_1(c1_0)
        non_c0_1 = non_linearity(c0_1_part)

        x0_1 = self.conv0_1(torch.cat((c0_0, non_c0_1), 1).view(c1_0.shape[0], -1, 4))
        x0_1_original_out = x0_1
        x0_1 = self.globalAvgPool4(x0_1)
        x0_1 = x0_1.view(x0_1.size(0), -1)
        x0_1 = self.fc0_1_1(x0_1)
        x0_1 = non_linearity(x0_1)
        x0_1 = self.fc0_1_2(x0_1)
        x0_1 = self.sigmoid(x0_1)
        x0_1 = x0_1.view(x0_1.size(0), x0_1.size(1), 1)
        x0_1 = x0_1 * x0_1_original_out
        non_x0_1 = non_linearity(x0_1)
        c0_1, _ = torch.max(non_x0_1, dim=2)

        x2_0 = self.conv2_0(non_x1_0)
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

        c1_1_part = self.fc1_1(c2_0)
        non_c1_1 = non_linearity(c1_1_part)
        x1_1 = self.conv1_1(torch.cat((c1_0, non_c1_1), 1).view(c1_0.shape[0], -1, 8))
        x1_1_original_out = x1_1
        x1_1 = self.globalAvgPool8(x1_1)
        x1_1 = x1_1.view(x1_1.size(0), -1)
        x1_1 = self.fc1_1_1(x1_1)
        x1_1 = non_linearity(x1_1)
        x1_1 = self.fc1_1_2(x1_1)
        x1_1 = self.sigmoid(x1_1)
        x1_1 = x1_1.view(x1_1.size(0), x1_1.size(1), 1)
        x1_1 = x1_1 * x1_1_original_out
        non_x1_1 = non_linearity(x1_1)
        c1_1, _ = torch.max(non_x1_1, dim=2)

        c0_2_part = self.fc0_2(c1_1)
        non_c0_2 = non_linearity(c0_2_part)
        x0_2 = self.conv0_2(torch.cat((c0_0, c0_1, non_c0_2), 1).view(c0_0.shape[0], -1, 4))
        non_x0_2 = non_linearity(x0_2)
        c0_2, _ = torch.max(non_x0_2, dim=2)

        output = c0_2.view(c0_2.shape[0], 3, self.out_size)

        encoded = torch.cat((c2_0, c0_1), 1).view(c2_0.shape[0], -1)
        return output, encoded
