import torch.nn as nn
import numpy as np
np.random.seed(42)
import tensorflow as tf
import torch as pt
import torch.nn.functional as F
import torchvision as ptv
import numpy as np

from keras import regularizers
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense

import numpy as np

MAX_INT = np.iinfo(np.int32).max
data_format = 0


class Dnn(pt.nn.Module):
    def __init__(self, input_shape, n_cls, device):
        super(Dnn, self).__init__()
        self.input_shape = input_shape
        self.n_cls = n_cls
        self.device = device

    def forward(self, data):
        return data


class MlpCls421(Dnn):
    def __init__(self, input_shape, n_cls, device):
        super(MlpCls421, self).__init__(input_shape, n_cls, device)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape/2)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape/2), int(input_shape/4)).to(device)
        self.fc3 = pt.nn.Linear(int(input_shape/4), n_cls).to(device)

    def forward(self, data):
        dout = nn.functional.relu(F.relu(self.fc1(data)))
        dout = nn.functional.relu(F.relu(self.fc2(dout)))
        return self.fc3(dout)


class MlpCls21(Dnn):
    def __init__(self, input_shape, n_cls, device):
        super(MlpCls21, self).__init__(input_shape, n_cls, device)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape/2)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape/2), n_cls).to(device)

    def forward(self, data):
        dout = nn.functional.relu(F.relu(self.fc1(data)))
        return self.fc2(dout)


class MlpCls121(Dnn):
    def __init__(self, input_shape, n_cls, device):
        super(MlpCls121, self).__init__(input_shape, n_cls, device)
        self.fc1 = pt.nn.Linear(input_shape, 2*input_shape).to(device)

        self.fc2 = pt.nn.Linear(2*input_shape, input_shape).to(device)
        self.fc3 = pt.nn.Linear(input_shape, n_cls).to(device)

    def forward(self, data):
        dout = nn.functional.relu(F.relu(self.fc1(data)))
        dout = nn.functional.relu(F.relu(self.fc2(dout)))
        return self.fc3(dout)


class MlpCls212(Dnn):
    def __init__(self, input_shape, n_cls, device):
        super(MlpCls212, self).__init__(input_shape, n_cls, device)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape/2)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape/2), input_shape).to(device)
        self.fc3 = pt.nn.Linear(input_shape, n_cls).to(device)

    def forward(self, data):
        dout = nn.functional.relu(F.relu(self.fc1(data)))
        dout = nn.functional.relu(F.relu(self.fc2(dout)))
        return self.fc3(dout)


class MlpCls42124(Dnn):
    def __init__(self, input_shape, n_cls, device):
        super(MlpCls42124, self).__init__(input_shape, n_cls, device)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape/2)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape/2), int(input_shape/4)).to(device)
        self.fc3 = pt.nn.Linear(int(input_shape/4), int(input_shape/2)).to(device)
        self.fc4 = pt.nn.Linear(int(input_shape/2), input_shape).to(device)
        self.fc5 = pt.nn.Linear(input_shape, n_cls).to(device)

    def forward(self, data):
        dout = nn.functional.relu(F.relu(self.fc1(data)))
        dout = nn.functional.relu(F.relu(self.fc2(dout)))
        dout = nn.functional.relu(F.relu(self.fc3(dout)))
        dout = nn.functional.relu(F.relu(self.fc4(dout)))
        return self.fc5(dout)


class CnnCls22(Dnn):
    def __init__(self, input_shape, n_cls, device):
        super(CnnCls22, self).__init__(input_shape, n_cls, device)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=3, padding=1),
            nn.BatchNorm1d(3, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        self.layer2 = nn.Sequential(
            nn.Conv1d(3, 6, kernel_size=3, padding=1),
            nn.BatchNorm1d(6, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int(((self.input_shape + 1 * 2 - 3) / 1 + 1) / 2)
        input_size = int(((input_size + 1 * 2 - 3) / 1 + 1) / 2)
        self.fc1 = nn.Linear(input_size*6, int(input_size*6/2)).to(device)
        self.fc2 = nn.Linear(int(input_size*6/2), self.n_cls).to(device)

    def forward(self, data):
        out = self.layer1(data.unsqueeze(1))
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class CnnCls21(Dnn):
    def __init__(self, input_shape, n_cls, device):
        super(CnnCls21, self).__init__(input_shape, n_cls, device)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=3, padding=1),
            nn.BatchNorm1d(3, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        self.layer2 = nn.Sequential(
            nn.Conv1d(3, 6, kernel_size=3, padding=1),
            nn.BatchNorm1d(6, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int(((self.input_shape+1*2-3)/1 + 1)/2)
        input_size = int(((input_size+1*2-3)/1 + 1)/2)
        self.fc1 = nn.Linear(input_size*6, self.n_cls).to(device)

    def forward(self, data):
        out = self.layer1(data.unsqueeze(1))
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class CnnCls11(Dnn):
    def __init__(self, input_shape, n_cls, device):
        super(CnnCls11, self).__init__(input_shape, n_cls, device)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=3, padding=1),
            nn.BatchNorm1d(3, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int(((self.input_shape+1*2-3)/1 + 1)/2)
        self.fc1 = nn.Linear(input_size*3, self.n_cls).to(device)

    def forward(self, data):
        out = self.layer1(data.unsqueeze(1))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class CnnCls12(Dnn):
    def __init__(self, input_shape, n_cls, device):
        super(CnnCls12, self).__init__(input_shape, n_cls, device)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=3, padding=1),
            nn.BatchNorm1d(3, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int(((self.input_shape+1*2-3)/1 + 1)/2)
        self.fc1 = nn.Linear(input_size * 3, int(input_size* 3 / 2)).to(device)
        self.fc2 = nn.Linear(int(input_size* 3 / 2), self.n_cls).to(device)

    def forward(self, data):
        out = self.layer1(data.unsqueeze(1))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class MlpCls12421(Dnn):
    def __init__(self, input_shape, n_cls, device):
        super(MlpCls12421, self).__init__(input_shape, n_cls, device)
        self.fc1 = pt.nn.Linear(input_shape, 2*input_shape).to(device)

        self.fc2 = pt.nn.Linear(2*input_shape, 4*input_shape).to(device)
        self.fc3 = pt.nn.Linear(4 * input_shape, 2*input_shape).to(device)
        self.fc4 = pt.nn.Linear(2 * input_shape, input_shape).to(device)
        self.fc5 = pt.nn.Linear(input_shape, n_cls).to(device)

    def forward(self, data):
        dout = nn.functional.relu(F.relu(self.fc1(data)))
        dout = nn.functional.relu(F.relu(self.fc2(dout)))
        dout = nn.functional.relu(F.relu(self.fc3(dout)))
        dout = nn.functional.relu(F.relu(self.fc4(dout)))
        return self.fc5(dout)


class MlpReg(pt.nn.Module):
    def __init__(self, input_shape, device):
        super(MlpReg, self).__init__()
        self.fc1 = pt.nn.Linear(input_shape, 2*input_shape).to(device)
        self.fc2 = pt.nn.Linear(2*input_shape, input_shape).to(device)
        self.fc3 = pt.nn.Linear(input_shape, 1).to(device)

    def forward(self, data):
        dout = F.relu(self.fc1(data))
        dout = F.relu(self.fc2(dout))
        return self.fc3(dout)


def dev_network_d(input_shape):
    '''
    deeper network architecture with three hidden layers
    '''
    x_input = Input(shape=(input_shape,))
    intermediate = Dense(1000, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(250, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl2')(intermediate)
    intermediate = Dense(20, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl3')(intermediate)
    intermediate = Dense(2, activation='softmax',  name='score')(intermediate)
    return Model(x_input, intermediate)


def dev_network_s(input_shape, output_shape):
    '''
    network architecture with one hidden layer
    '''
    x_input = Input(shape=(input_shape,))
    # intermediate = Dense(20, activation='relu',
    #             kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(2*input_shape, activation='relu', name='hl1')(x_input)
    intermediate = Dense(input_shape, activation='relu', name='hl2')(intermediate)
    intermediate = Dense(output_shape, activation='softmax',  name='score')(intermediate)
    return Model(x_input, intermediate)


def dev_network_s_r(input_shape):
    '''
    network architecture with one hidden layer
    '''
    x_input = Input(shape=(input_shape,))
    # intermediate = Dense(20, activation='relu',
    #             kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(2*input_shape, activation='relu', name='hl1')(x_input)
    intermediate = Dense(input_shape, activation='relu', name='hl2')(intermediate)
    intermediate = Dense(1, name='score')(intermediate)
    return Model(x_input, intermediate)


def dev_network_sr(input_shape, output_shape):
    '''
    network architecture with one hidden layer
    '''
    x_input = Input(shape=(input_shape,))
    # intermediate = Dense(20, activation='relu',
    #             kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(2*input_shape, activation='relu', name='hl1')(x_input)
    intermediate = Dense(input_shape, activation='relu', name='hl2')(intermediate)
    intermediate = Dense(1, name='score')(intermediate)
    return Model(x_input, intermediate)


def dev_network_linear(input_shape):
    '''
    network architecture with no hidden layer, equivalent to linear mapping from
    raw inputs to anomaly scores
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(1, activation='linear',  name = 'score')(x_input)
    return Model(x_input, intermediate)
