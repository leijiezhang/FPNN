import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN_c(torch.nn.Module):
    """
    This is the FPN based on BP, I reform the FPN net referring to the graph attention network.
    """

    def __init__(self, prototypes: torch.Tensor, n_cls, device):
        """

        :param prototypes:
        :param n_fea:
        """
        super(FPN_c, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]
        self.n_cls = n_cls

        # parameters in network
        # self.proto = torch.autograd.Variable(prototypes, requires_grad=False)
        self.proto = nn.Parameter(prototypes, requires_grad=True).to(device)

        # self.proto_reform = torch.autograd.Variable(prototypes, requires_grad=True)
        # self.data_reform = torch.autograd.Variable(prototypes, requires_grad=True)
        self.fire_strength_ini = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True).to(device)
        self.fire_strength = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True).to(device)
        # if torch.cuda.is_available():
        #     self.proto = self.proto.cuda()
        #     self.proto_reform = self.proto_reform.cuda()
        self.fs_layers = torch.nn.Sequential(
            torch.nn.Linear(self.n_fea, 2 * self.n_fea),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.n_fea, self.n_fea),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_fea, 1),
            # torch.nn.Tanh()
        ).to(device)
        self.w_layer = nn.Linear(self.n_fea, self.n_fea).to(device)

        # parameters in consequent layer

        # self.fire_strength_active = nn.LeakyReLU(0.005)
        self.relu_active = nn.ReLU().to(device)
        self.leak_relu_active = nn.functional.leaky_relu
        self.batch_norm = torch.nn.BatchNorm1d(self.n_rules).to(device)
        # parameters in consequent layer
        self.consq_layers = [nn.Linear(self.n_fea, n_cls).to(device) for _ in range(self.n_rules)]
        for i, consq_layers_item in enumerate(self.consq_layers):
            self.add_module('para_consq_{}'.format(i), consq_layers_item)

    def forward(self, data: torch.Tensor, is_train):
        n_batch = data.shape[0]
        # activate prototypes
        # self.proto_reform = torch.tanh(self.w_layer(self.proto))
        # self.data_reform = torch.tanh(self.w_layer(data))

        # data_expands = self.data_reform.repeat_interleave(self.n_rules, dim=0)
        # proto_expands = self.proto_reform.repeat(n_batch, 1)

        # data_expands = self.data_reform.repeat(self.n_rules, 1)
        # proto_expands = self.proto_reform.repeat_interleave(n_batch, dim=0)
        data_expands = data.repeat(self.n_rules, 1)
        proto_expands = self.proto.repeat_interleave(n_batch, dim=0)
        data_diff = (data_expands - proto_expands).view(self.n_rules, n_batch, self.n_fea)
        self.fire_strength_ini = torch.cat([self.fs_layers(data_diff_item) for data_diff_item in data_diff], dim=1)
        # self.fire_strength_ini = torch.tanh(self.fire_strength_ini)
        # self.fire_strength_ini = nn.functional.dropout(self.fire_strength_ini, 0.6)
        # self.fire_strength_ini = self.batch_norm(self.fire_strength_ini)
        self.fire_strength = F.softmax(self.fire_strength_ini, dim=1)

        # self.fire_strength = nn.functional.dropout(self.fire_strength, 0.2, training=is_train)
        # print(fire_strength)

        # produce consequent layer
        fire_strength_processed = self.fire_strength.t().unsqueeze(2).repeat(1, 1, self.n_cls)
        data_processed = torch.cat([consq_layers_item(data) for consq_layers_item in self.consq_layers], dim=0)
        data_processed = data_processed.view(self.n_rules, n_batch, self.n_cls)
        # data_processed = nn.functional.dropout(data_processed, 0.2)
        outputs = torch.mul(fire_strength_processed, data_processed).sum(0)
        # outputs = F.softmax(outputs, dim=1)

        return outputs


class FPN(torch.nn.Module):
    """
    This is the FPN based on BP, I reform the FPN net referring to the graph attention network.
    """

    def __init__(self, prototypes: torch.Tensor):
        """

        :param prototypes:
        :param n_fea:
        """
        super(FPN, self).__init__()
        self.n_rules = prototypes.shape[0]
        self.n_fea = prototypes.shape[1]

        # parameters in network
        self.proto = torch.autograd.Variable(prototypes, requires_grad=False)
        # self.proto = nn.Parameter(prototypes, requires_grad=True)
        self.proto_reform = torch.autograd.Variable(prototypes, requires_grad=True)
        self.data_reform = torch.autograd.Variable(prototypes, requires_grad=True)
        self.fire_strength_ini = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True)
        self.fire_strength = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True)
        # if torch.cuda.is_available():
        #     self.proto = self.proto.cuda()
        #     self.proto_reform = self.proto_reform.cuda()
        self.fs_layers = torch.nn.Sequential(
            torch.nn.Linear(self.n_fea, 2 * self.n_fea),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.n_fea, self.n_fea),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_fea, 1),
            torch.nn.Tanh()
        )
        self.w_layer = nn.Linear(self.n_fea, self.n_fea)

        # parameters in consequent layer

        # self.fire_strength_active = nn.LeakyReLU(0.005)
        self.relu_active = nn.ReLU()
        self.leak_relu_active = nn.functional.leaky_relu
        self.batch_norm = torch.nn.BatchNorm1d(self.n_rules)
        # parameters in consequent layer
        self.consq_layers = [nn.Linear(self.n_fea, 1) for _ in range(self.n_rules)]
        for i, consq_layers_item in enumerate(self.consq_layers):
            self.add_module('para_consq_{}'.format(i), consq_layers_item)

    def forward(self, data: torch.Tensor, is_train):
        n_batch = data.shape[0]
        # activate prototypes
        self.proto_reform = torch.tanh(self.w_layer(self.proto))
        self.data_reform = torch.tanh(self.w_layer(data))

        # data_expands = self.data_reform.repeat_interleave(self.n_rules, dim=0)
        # proto_expands = self.proto_reform.repeat(n_batch, 1)

        data_expands = self.data_reform.repeat(self.n_rules, 1)
        proto_expands = self.proto_reform.repeat_interleave(n_batch, dim=0)
        data_diff = (data_expands - proto_expands).view(self.n_rules, n_batch, self.n_fea)
        self.fire_strength_ini = torch.cat([self.fs_layers(data_diff_item) for data_diff_item in data_diff], dim=1)
        # self.fire_strength_ini = torch.tanh(self.fire_strength_ini)
        # self.fire_strength_ini = nn.functional.dropout(self.fire_strength_ini, 0.6)
        # self.fire_strength_ini = self.batch_norm(self.fire_strength_ini)
        self.fire_strength = nn.functional.softmax(self.fire_strength_ini, dim=1)

        # self.fire_strength = nn.functional.dropout(self.fire_strength, 0.2, training=is_train)
        # print(fire_strength)

        # produce consequent layer
        data_processed = torch.cat([consq_layers_item(data) for consq_layers_item in self.consq_layers], dim=1)
        # data_processed = nn.functional.dropout(data_processed, 0.2)
        outputs = torch.mul(self.fire_strength, data_processed).sum(1).unsqueeze(1)

        return outputs


# class FPN(torch.nn.Module):
#     """
#     This is the FPN based on BP, I reform the FPN net referring to the graph attention network.
#     """
#
#     def __init__(self, prototypes: torch.Tensor):
#         """
#
#         :param prototypes:
#         :param n_fea:
#         """
#         super(FPN, self).__init__()
#         self.n_rules = prototypes.shape[0]
#         self.n_fea = prototypes.shape[1]
#
#         # parameters in network
#         # self.proto = torch.autograd.Variable(prototypes, requires_grad=False)
#         self.proto = nn.Parameter(prototypes, requires_grad=True)
#         # self.proto_reform = torch.autograd.Variable(prototypes, requires_grad=True)
#         # self.data_reform = torch.autograd.Variable(prototypes, requires_grad=True)
#         self.fire_strength_ini = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True)
#         self.fire_strength = torch.autograd.Variable(torch.empty(self.n_rules, self.n_fea), requires_grad=True)
#         # if torch.cuda.is_available():
#         #     self.proto = self.proto.cuda()
#         #     self.proto_reform = self.proto_reform.cuda()
#         # self.fs_layers = torch.nn.Sequential(
#         #     torch.nn.Linear(self.n_fea, 2 * self.n_fea),
#         #     torch.nn.ReLU(),
#         #     torch.nn.Linear(2 * self.n_fea, self.n_fea),
#         #     torch.nn.ReLU(),
#         #     torch.nn.Linear(self.n_fea, 1),
#         #     torch.nn.Tanh()
#         # )
#         n_channel = 4
#         n_padding = 1
#         kernel_size = 3
#         pooling_size = 2
#         att_size = n_channel * torch.ceil((torch.ceil(torch.tensor(self.n_fea + 2*n_padding + 1 - kernel_size).float()/pooling_size) +
#                                    2*n_padding + 1 - kernel_size)/pooling_size)
#         self.fs_layers = RelationNetwork(int(att_size), int(att_size/2), n_channel, n_padding, kernel_size, pooling_size)
#         # self.w_layer = nn.Linear(self.n_fea, self.n_fea)
#
#         # parameters in consequent layer
#
#         # self.fire_strength_active = nn.LeakyReLU(0.005)
#         self.relu_active = nn.ReLU()
#         self.leak_relu_active = nn.functional.leaky_relu
#         self.batch_norm = torch.nn.BatchNorm1d(self.n_rules)
#         # parameters in consequent layer
#         self.consq_layers = [nn.Linear(self.n_fea, 1) for _ in range(self.n_rules)]
#         for i, consq_layers_item in enumerate(self.consq_layers):
#             self.add_module('para_consq_{}'.format(i), consq_layers_item)
#
#     def forward(self, data: torch.Tensor, is_train):
#         n_batch = data.shape[0]
#         # activate prototypes
#         # self.proto_reform = torch.tanh(self.w_layer(self.proto))
#         # self.data_reform = torch.tanh(self.w_layer(data))
#
#         # data_expands = self.data_reform.repeat_interleave(self.n_rules, dim=0)
#         # proto_expands = self.proto_reform.repeat(n_batch, 1)
#
#         # data_expands = self.data_reform.repeat(self.n_rules, 1)
#         # proto_expands = self.proto_reform.repeat_interleave(n_batch, dim=0)
#
#         # data_expands = self.data_reform.repeat(self.n_rules, 1)
#         # proto_expands = self.proto_reform.repeat_interleave(n_batch, dim=0)
#         data_expands = data.repeat(self.n_rules, 1)
#         proto_expands = self.proto.repeat_interleave(n_batch, dim=0)
#
#         data_diff = (data_expands - proto_expands).view(self.n_rules, n_batch, self.n_fea)
#         self.fire_strength_ini = torch.cat([self.fs_layers(data_diff_item) for data_diff_item in data_diff], dim=1)
#         # self.fire_strength_ini = torch.tanh(self.fire_strength_ini)
#         # self.fire_strength_ini = nn.functional.dropout(self.fire_strength_ini, 0.6)
#         # self.fire_strength_ini = self.batch_norm(self.fire_strength_ini)
#         self.fire_strength = nn.functional.softmax(self.fire_strength_ini, dim=1)
#
#         # self.fire_strength = nn.functional.dropout(self.fire_strength, 0.2, training=is_train)
#         # print(fire_strength)
#
#         # produce consequent layer
#         data_processed = torch.cat([consq_layers_item(data) for consq_layers_item in self.consq_layers], dim=1)
#         # data_processed = nn.functional.dropout(data_processed, 0.2)
#         outputs = torch.mul(self.fire_strength, data_processed).sum(1).unsqueeze(1)
#
#         return outputs


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, input_size, hidden_size, n_channel, n_padding, kernel_size, pooling_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv1d(1, n_channel, kernel_size=kernel_size, padding=n_padding),
                        nn.BatchNorm1d(n_channel, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(pooling_size))
        self.layer2 = nn.Sequential(
                        nn.Conv1d(n_channel, n_channel, kernel_size=kernel_size, padding=n_padding),
                        nn.BatchNorm1d(n_channel, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(pooling_size))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x.unsqueeze(1))
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        return out

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         m.weight.data.normal_(0, math.sqrt(2. / n))
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()
#     elif classname.find('Linear') != -1:
#         n = m.weight.size(1)
#         m.weight.data.normal_(0, 0.01)
#         m.bias.data = torch.ones(m.bias.data.size())