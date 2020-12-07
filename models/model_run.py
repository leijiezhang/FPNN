from param_config import ParamConfig
from dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.svm import SVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict
import torch.nn as nn
import os
import torch
from kmeans_pytorch import kmeans
from keras.utils import to_categorical
from sklearn.cluster import KMeans
from loss_utils import NRMSELoss, LikelyLoss, LossFunc
from fpn_models import FpnMlpFsCls, FpnCov1dFSCls, FpnMlpFsReg, FpnCov1dFSReg

from dataset import DatasetTorch
from mlp_model import MlpReg, dev_network_s, dev_network_sr, dev_network_s_r, MlpCls
import scipy.io as io
from h_utils import HNormal
from rules import RuleKmeans
from fnn_solver import FnnSolveReg


def svc(train_fea: torch.Tensor, test_fea: torch.Tensor, train_gnd: torch.Tensor,
        test_gnd: torch.Tensor, loss_fun: LossFunc, paras: Dict):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param train_fea: training data
    :param test_fea: test data
    :param train_gnd: training label
    :param test_gnd: test label
    :param loss_fun: the loss function that used to calculate the loss of regression or accuracy of classification task
    :param paras: parameters that used for training SVC model

    :return:
    """
    """ codes for parameters 
        paras = dict()
        paras['kernel'] = 'rbf'
        paras['gamma'] = gamma
        paras['C'] = C
    """
    print("training the one-class SVM")
    train_gnd = train_gnd.squeeze()
    test_gnd = test_gnd.squeeze()
    if 'kernel' in paras:
        svm_kernel = paras['kernel']
    else:
        svm_kernel = 'rbf'
    if 'gamma' in paras:
        svm_gamma = paras['gamma']
    else:
        svm_gamma = 'scale'
    if 'C' in paras:
        svm_c = paras['C']
    else:
        svm_c = 1
    svm_train = SVC(kernel=svm_kernel, gamma=svm_gamma, C=svm_c)
    clf = make_pipeline(StandardScaler(), svm_train)
    clf.fit(train_fea.numpy(), train_gnd.numpy())
    train_gnd_hat = clf.predict(train_fea.numpy())
    test_gnd_hat = clf.predict(test_fea.numpy())

    train_acc = loss_fun.forward(train_gnd.squeeze(), torch.tensor(train_gnd_hat))
    test_acc = loss_fun.forward(test_gnd.squeeze(), torch.tensor(test_gnd_hat))

    """ following code is designed for those functions that need to output the svm results
    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data using SVM: {train_acc:.2f}%")
        param_config.log.info(f"Accuracy of test data using SVM: {test_acc:.2f}%")
    else:
        param_config.log.info(f"loss of training data using SVM: {train_acc:.4f}")
        param_config.log.info(f"loss of test data using SVM: {test_acc:.4f}")
    """

    return train_acc, test_acc


def svr(train_fea: torch.Tensor, test_fea: torch.Tensor, train_gnd: torch.Tensor,
        test_gnd: torch.Tensor, loss_fun: LossFunc, paras: Dict):
    """
    todo: this is the method for sSVR
    :param train_fea: training data
    :param test_fea: test data
    :param train_gnd: training label
    :param test_gnd: test label
    :param loss_fun: the loss function that used to calculate the loss of regression or accuracy of classification task
    :param paras: parameters that used for training SVC model

    :return:
    """
    """ codes for parameters 
        paras = dict()
        paras['kernel'] = 'rbf'
        paras['gamma'] = gamma
        paras['C'] = C
    """
    print("training the one-class SVR")
    train_gnd = train_gnd.squeeze()
    test_gnd = test_gnd.squeeze()
    if 'kernel' in paras:
        svm_kernel = paras['kernel']
    else:
        svm_kernel = 'rbf'
    if 'gamma' in paras:
        svm_gamma = paras['gamma']
    else:
        svm_gamma = 'scale'
    if 'C' in paras:
        svm_c = paras['C']
    else:
        svm_c = 1
    svm_train = SVR(kernel=svm_kernel, gamma=svm_gamma, C=svm_c)
    clf = make_pipeline(StandardScaler(), svm_train)
    clf.fit(train_fea.numpy(), train_gnd.numpy())
    train_gnd_hat = clf.predict(train_fea.numpy())
    test_gnd_hat = clf.predict(test_fea.numpy())

    train_loss = loss_fun.forward(train_gnd.squeeze(), torch.tensor(train_gnd_hat))
    test_loss = loss_fun.forward(test_gnd.squeeze(), torch.tensor(test_gnd_hat))

    """ following code is designed for those functions that need to output the svr results
    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data using SVR: {train_acc:.2f}%")
        param_config.log.info(f"Accuracy of test data using SVE: {test_acc:.2f}%")
    else:
        param_config.log.info(f"loss of training data using SVE: {train_loss:.4f}")
        param_config.log.info(f"loss of test data using SVE: {test_loss:.4f}")
    """

    return train_loss, test_loss


def fpn_cls(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
        todo: this is the method for fuzzy Neuron network using kmeans
        :param param_config:
        :param train_data: training dataset
        :param test_data: test dataset
        :param current_k: current k
        :return:
    """
    h_computer = HNormal()
    rules = RuleKmeans()
    rules.fit(train_data.fea, param_config.n_rules)
    h_train, _ = h_computer.comute_h(train_data.fea, rules)
    # run FNN solver for given rule number
    fnn_solver = FnnSolveReg()
    fnn_solver.h = h_train
    fnn_solver.y = train_data.gnd
    fnn_solver.para_mu = 0.1
    w_optimal = fnn_solver.solve().squeeze()

    rules.consequent_list = w_optimal

    n_rule_train = h_train.shape[0]
    n_smpl_train = h_train.shape[1]
    n_fea_train = h_train.shape[2]
    h_cal_train = h_train.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_train = h_cal_train.reshape(n_smpl_train, n_rule_train * n_fea_train)
    y_train_hat = h_cal_train.mm(rules.consequent_list.reshape(1, n_rule_train * n_fea_train).t())

    fnn_train_acc = LikelyLoss().forward(train_data.gnd, y_train_hat)

    h_test, _ = h_computer.comute_h(test_data.fea, rules)
    n_rule_test = h_test.shape[0]
    n_smpl_test = h_test.shape[1]
    n_fea_test = h_test.shape[2]
    h_cal_test = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_test = h_cal_test.reshape(n_smpl_test, n_rule_test * n_fea_test)
    y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(1, n_rule_test * n_fea_test).t())

    fnn_test_acc = LikelyLoss().forward(test_data.gnd, y_test_hat)

    param_config.log.info(f"Training acc of traditional FNN: {fnn_train_acc}")
    param_config.log.info(f"Test acc of test traditional FNN: {fnn_test_acc}")
    return fnn_train_acc, fnn_test_acc


def fpn_reg(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
        todo: this is the method for fuzzy Neuron network using kmeans
        :param param_config:
        :param train_data: training dataset
        :param test_data: test dataset
        :param current_k: current k
        :return:
    """
    h_computer = HNormal()
    rules = RuleKmeans()
    rules.fit(train_data.fea, param_config.n_rules)
    h_train, _ = h_computer.comute_h(train_data.fea, rules)
    # run FNN solver for given rule number
    fnn_solver = FnnSolveReg()
    fnn_solver.h = h_train
    fnn_solver.y = train_data.gnd
    fnn_solver.para_mu = 0.1
    w_optimal = fnn_solver.solve().squeeze()

    rules.consequent_list = w_optimal

    n_rule_train = h_train.shape[0]
    n_smpl_train = h_train.shape[1]
    n_fea_train = h_train.shape[2]
    h_cal_train = h_train.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_train = h_cal_train.reshape(n_smpl_train, n_rule_train * n_fea_train)
    y_train_hat = h_cal_train.mm(rules.consequent_list.reshape(1, n_rule_train * n_fea_train).t())

    fnn_train_mse = NRMSELoss().forward(train_data.gnd, y_train_hat)

    h_test, _ = h_computer.comute_h(test_data.fea, rules)
    n_rule_test = h_test.shape[0]
    n_smpl_test = h_test.shape[1]
    n_fea_test = h_test.shape[2]
    h_cal_test = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_test = h_cal_test.reshape(n_smpl_test, n_rule_test * n_fea_test)
    y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(1, n_rule_test * n_fea_test).t())

    fnn_test_mse = NRMSELoss().forward(test_data.gnd, y_test_hat)

    param_config.log.info(f"Training me of traditional FNN: {fnn_train_mse}")
    param_config.log.info(f"Test mse of test traditional FNN: {fnn_test_mse}")
    return fnn_train_mse, fnn_test_mse


def fpn_run_reg(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    h_computer = HNormal()
    rules = RuleKmeans()
    rules.fit(train_data.fea, param_config.n_rules)
    h_train, _ = h_computer.comute_h(train_data.fea, rules)
    # run FNN solver for given rule number
    fnn_solver = FnnSolveReg()
    fnn_solver.h = h_train
    fnn_solver.y = train_data.gnd
    fnn_solver.para_mu = 0.1
    w_optimal = fnn_solver.solve().squeeze()

    rules.consequent_list = w_optimal

    n_rule_train = h_train.shape[0]
    n_smpl_train = h_train.shape[1]
    n_fea_train = h_train.shape[2]
    h_cal_train = h_train.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_train = h_cal_train.reshape(n_smpl_train, n_rule_train * n_fea_train)
    y_train_hat = h_cal_train.mm(rules.consequent_list.reshape(1, n_rule_train * n_fea_train).t())

    fnn_train_mse = NRMSELoss().forward(train_data.gnd, y_train_hat)

    h_test, _ = h_computer.comute_h(test_data.fea, rules)
    n_rule_test = h_test.shape[0]
    n_smpl_test = h_test.shape[1]
    n_fea_test = h_test.shape[2]
    h_cal_test = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_test = h_cal_test.reshape(n_smpl_test, n_rule_test * n_fea_test)
    y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(1, n_rule_test * n_fea_test).t())

    fnn_test_mse = NRMSELoss().forward(test_data.gnd, y_test_hat)

    param_config.log.info(f"Training me of traditional FNN: {fnn_train_mse}")
    param_config.log.info(f"Test mse of test traditional FNN: {fnn_test_mse}")

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    # model: nn.Module = MLP(train_data.fea.shape[1])
    # rules = RuleKmeans()
    # rules.fit(train_data.fea, param_config.n_rules)
    prototype_list = rules.center_list
    # prototype_list = torch.ones(param_config.n_rules, train_data.n_fea)
    # prototype_list = train_data.fea[torch.randperm(train_data.n_smpl)[0:param_config.n_rules], :]
    fpn_model: nn.Module = FpnMlpFsReg(prototype_list, param_config.device)
    # initiate model parameter
    # fpn_model.proto_reform_w.data = torch.eye(train_data.fea.shape[1])
    # model.proto_reform_layer.bias.data = torch.zeros(train_data.fea.shape[1])

    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    loss_fn = nn.MSELoss()
    epochs = param_config.n_epoch

    fpn_train_losses = []
    fpn_valid_losses = []

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
                      f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
                      f"k_{current_k}.pkl"
    # #load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            outputs_temp = fpn_model(data, True)
            loss = loss_fn(outputs_temp, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, 1).to(param_config.device)
        outputs_val = torch.empty(0, 1).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                outputs_temp = fpn_model(data, False)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            loss_train = NRMSELoss().forward(outputs_train, gnd_train)
            fpn_train_losses.append(loss_train.item())
            for i, (data, labels) in enumerate(valid_loader):
                outputs_temp = fpn_model(data, False)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            loss_val = NRMSELoss().forward(outputs_val, gnd_val)
            fpn_valid_losses.append(loss_val.item())
        param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        if best_test_rslt < loss_train:
            torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train loss : {fpn_train_losses[-1]}, test loss : {fpn_valid_losses[-1]}")

    # mlp model
    mlp_model: nn.Module = MlpReg(train_data.fea.shape[1], param_config.device)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=param_config.lr)
    loss_fn = nn.MSELoss()
    epochs = param_config.n_epoch

    mlp_train_losses = []
    mlp_valid_losses = []

    for epoch in range(epochs):
        mlp_model.train()

        for i, (data, labels) in enumerate(train_loader):
            outputs = mlp_model(data)
            # loss = loss_fn(outputs.double(), labels.double().squeeze(1))
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mlp_model.eval()
        outputs_train = torch.empty(0, 1).to(param_config.device)
        outputs_val = torch.empty(0, 1).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                outputs_temp = mlp_model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            loss_train = loss_fn(outputs_train, gnd_train)
            mlp_train_losses.append(loss_train.item())
            for i, (data, labels) in enumerate(valid_loader):
                outputs_temp = mlp_model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            loss_val = NRMSELoss().forward(outputs_val, gnd_val)
            mlp_valid_losses.append(loss_val.item())

        param_config.log.info(
            f"mlp epoch : {epoch + 1}, train loss : {mlp_train_losses[-1]}, test loss : {mlp_valid_losses[-1]}")

    param_config.log.info("finished")
    plt.figure(0)
    title = f"FPN MSE of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    # plt.plot(torch.arange(len(mlp_train_losses)), torch.tensor(mlp_train_losses), 'b--', linewidth=2, markersize=5)
    # plt.plot(torch.arange(len(mlp_valid_losses)), torch.tensor(mlp_valid_losses), 'r--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_losses)), fnn_train_mse.cpu().expand_as(torch.tensor(fpn_valid_losses)),
             'b--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_losses)), fnn_test_mse.cpu().expand_as(torch.tensor(fpn_valid_losses)),
             'r--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_losses)), torch.tensor(fpn_train_losses).cpu(), 'b:', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(fpn_valid_losses)), torch.tensor(fpn_valid_losses).cpu(), 'r:', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(mlp_valid_losses)), torch.tensor(mlp_train_losses).cpu(), 'b-.', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(mlp_valid_losses)), torch.tensor(mlp_valid_losses).cpu(), 'r-.', linewidth=2,
             markersize=5)
    plt.legend(['fnn train', 'fnn test', 'fpn train', 'fpn test', 'mlp train', 'mlp test'])
    # plt.legend(['fnn train', 'fnn test', 'fpn train', 'fpn test'])
    # plt.legend(['mlp train', 'mlp test', 'fpn train', 'fpn test'])
    # plt.legend(['fpn train', 'fpn test'])
    plt.show()

    # save all the results
    save_dict = dict()
    save_dict["fpn_train_losses"] = torch.tensor(fpn_train_losses).numpy()
    save_dict["fpn_valid_losses"] = torch.tensor(fpn_valid_losses).numpy()
    save_dict["mlp_train_losses"] = torch.tensor(mlp_train_losses).numpy()
    save_dict["mlp_valid_losses"] = torch.tensor(mlp_valid_losses).numpy()
    save_dict["fnn_train_mse"] = fnn_train_mse.numpy()
    save_dict["fnn_test_mse"] = fnn_test_mse.numpy()
    data_save_file = f"{data_save_dir}/mse_bpfnn_{param_config.dataset_folder}_rule" \
                     f"_{param_config.n_rules}_lr_{param_config.lr:.6f}" \
                     f"_k_{current_k}.mat"
    io.savemat(data_save_file, save_dict)
    return fpn_train_losses, fpn_valid_losses


def fpn_run_cls_cov(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    h_computer = HNormal()
    rules = RuleKmeans()
    rules.fit(train_data.fea, param_config.n_rules)
    h_train, _ = h_computer.comute_h(train_data.fea, rules)
    # run FNN solver for given rule number
    fnn_solver = FnnSolveReg()
    fnn_solver.h = h_train
    fnn_solver.y = train_data.gnd
    fnn_solver.para_mu = 0.1
    w_optimal = fnn_solver.solve().squeeze()

    rules.consequent_list = w_optimal

    n_rule_train = h_train.shape[0]
    n_smpl_train = h_train.shape[1]
    n_fea_train = h_train.shape[2]
    h_cal_train = h_train.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_train = h_cal_train.reshape(n_smpl_train, n_rule_train * n_fea_train)
    y_train_hat = h_cal_train.mm(rules.consequent_list.reshape(1, n_rule_train * n_fea_train).t())

    fnn_train_mse = LikelyLoss().forward(train_data.gnd, y_train_hat)

    h_test, _ = h_computer.comute_h(test_data.fea, rules)
    n_rule_test = h_test.shape[0]
    n_smpl_test = h_test.shape[1]
    n_fea_test = h_test.shape[2]
    h_cal_test = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_test = h_cal_test.reshape(n_smpl_test, n_rule_test * n_fea_test)
    y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(1, n_rule_test * n_fea_test).t())

    fnn_test_mse = LikelyLoss().forward(test_data.gnd, y_test_hat)

    param_config.log.info(f"Training acc of traditional FNN: {fnn_train_mse}")
    param_config.log.info(f"Test acc of test traditional FNN: {fnn_test_mse}")

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    # model: nn.Module = MLP(train_data.fea.shape[1])
    # rules = RuleKmeans()
    # rules.fit(train_data.fea, param_config.n_rules)
    prototype_list = rules.center_list
    # prototype_list = torch.ones(param_config.n_rules, train_data.n_fea)
    # prototype_list = train_data.fea[torch.randperm(train_data.n_smpl)[0:param_config.n_rules], :]
    n_cls = train_data.gnd.unique().shape[0]
    fpn_model: nn.Module = FpnCov1dFSCls(prototype_list, n_cls, param_config.device)
    # fpn_model = fpn_model.cuda()
    # initiate model parameter
    # fpn_model.proto_reform_w.data = torch.eye(train_data.fea.shape[1])
    # model.proto_reform_layer.bias.data = torch.zeros(train_data.fea.shape[1])

    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    fpn_train_acc = []
    fpn_valid_acc = []

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
                      f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
                      f"k_{current_k}.pkl"
    # #load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs_temp = fpn_model(data, True)
            loss = loss_fn(outputs_temp, labels.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = fpn_model(data, False)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            _, predicted_train = torch.max(outputs_train, 1)
            correct_train_num = (predicted_train == gnd_train.squeeze()).squeeze().sum()
            acc_train = correct_train_num/gnd_train.shape[0]
            fpn_train_acc.append(acc_train)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = fpn_model(data, False)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            _, predicted_val = torch.max(outputs_val, 1)
            correct_val_num = (predicted_val == gnd_val.squeeze()).squeeze().sum()
            acc_val = correct_val_num/gnd_val.shape[0]
            fpn_valid_acc.append(acc_val)
        param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        if best_test_rslt < acc_train:
            best_test_rslt = acc_train
            torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train acc : {fpn_train_acc[-1]}, test acc : {fpn_valid_acc[-1]}")

    # mlp model
    mlp_model: nn.Module = MlpCls(train_data.fea.shape[1], n_cls, param_config.device)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=param_config.lr)
    loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    mlp_train_acc = []
    mlp_valid_acc = []

    for epoch in range(epochs):
        mlp_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs = mlp_model(data)
            loss = loss_fn(outputs, labels.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mlp_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = mlp_model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            _, predicted_train = torch.max(outputs_train, 1)
            correct_train_num = (predicted_train == gnd_train.squeeze()).squeeze().sum()
            acc_train = correct_train_num / gnd_train.shape[0]
            mlp_train_acc.append(acc_train)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = mlp_model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            _, predicted_val = torch.max(outputs_val, 1)
            correct_val_num = (predicted_val == gnd_val.squeeze()).squeeze().sum()
            acc_val = correct_val_num / gnd_val.shape[0]
            mlp_valid_acc.append(acc_val)

        param_config.log.info(
            f"mlp epoch : {epoch + 1}, train acc : {mlp_train_acc[-1]}, test acc : {mlp_valid_acc[-1]}")

    param_config.log.info("finished")
    plt.figure(0)
    title = f"FPN Acc of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    # plt.plot(torch.arange(len(mlp_train_acc)), torch.tensor(mlp_train_acc), 'b--', linewidth=2, markersize=5)
    # plt.plot(torch.arange(len(mlp_valid_acc)), torch.tensor(mlp_valid_acc), 'r--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_acc)), fnn_train_mse.cpu().expand_as(torch.tensor(fpn_valid_acc)),
             'b--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_acc)), fnn_test_mse.cpu().expand_as(torch.tensor(fpn_valid_acc)),
             'r--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_acc)), torch.tensor(fpn_train_acc).cpu(), 'b:', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(fpn_valid_acc)), torch.tensor(fpn_valid_acc).cpu(), 'r:', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(mlp_valid_acc)), torch.tensor(mlp_train_acc).cpu(), 'b-.', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(mlp_valid_acc)), torch.tensor(mlp_valid_acc).cpu(), 'r-.', linewidth=2,
             markersize=5)
    plt.legend(['fnn train', 'fnn test', 'fpn train', 'fpn test', 'mlp train', 'mlp test'])
    # plt.legend(['fnn train', 'fnn test', 'fpn train', 'fpn test'])
    # plt.legend(['mlp train', 'mlp test', 'fpn train', 'fpn test'])
    # plt.legend(['fpn train', 'fpn test'])
    plt.show()

    # save all the results
    save_dict = dict()
    save_dict["fpn_train_acc"] = torch.tensor(fpn_train_acc).numpy()
    save_dict["fpn_valid_acc"] = torch.tensor(fpn_valid_acc).numpy()
    save_dict["mlp_train_acc"] = torch.tensor(mlp_train_acc).numpy()
    save_dict["mlp_valid_acc"] = torch.tensor(mlp_valid_acc).numpy()
    save_dict["fnn_train_mse"] = fnn_train_mse.cpu().numpy()
    save_dict["fnn_test_mse"] = fnn_test_mse.cpu().numpy()
    data_save_file = f"{data_save_dir}/mse_bpfnn_{param_config.dataset_folder}_rule" \
                     f"_{param_config.n_rules}_lr_{param_config.lr:.6f}" \
                     f"_k_{current_k}.mat"
    io.savemat(data_save_file, save_dict)
    return fpn_train_acc, fpn_valid_acc


def fpn_run_cls_mlp(param_config: ParamConfig, train_data: Dataset, test_data: Dataset, current_k):
    """
    todo: this is the method for fuzzy Neuron network using back propagation
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :param current_k: current k
    :return:
    """
    paras = dict()
    paras['kernel'] = 'rbf'
    svm_train_acc, svm_test_acc = svc(train_data.fea.cpu(), test_data.fea.cpu(), train_data.gnd.cpu(),
                                      test_data.gnd.cpu(), LikelyLoss(), paras)
    param_config.log.info(f"Accuracy of training data using SVM: {svm_train_acc}")
    param_config.log.info(f"Accuracy of test data using SVM: {svm_test_acc}")

    train_dataset = DatasetTorch(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetTorch(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=param_config.n_batch, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=param_config.n_batch, shuffle=False)
    # model: nn.Module = MLP(train_data.fea.shape[1])
    # rules = RuleKmeans()
    # rules.fit(train_data.fea, param_config.n_rules)
    _, prototype_list = kmeans(
        X=train_data.fea, num_clusters=param_config.n_rules, distance='euclidean', device=torch.device(train_data.fea.device)
    )
    prototype_list = prototype_list.to(param_config.device)
    # prototype_list = torch.ones(param_config.n_rules, train_data.n_fea)
    # prototype_list = train_data.fea[torch.randperm(train_data.n_smpl)[0:param_config.n_rules], :]
    n_cls = train_data.gnd.unique().shape[0]
    fpn_model: nn.Module = FpnMlpFsCls(prototype_list, n_cls, param_config.device)
    # fpn_model = fpn_model.cuda()
    # initiate model parameter
    # fpn_model.proto_reform_w.data = torch.eye(train_data.fea.shape[1])
    # model.proto_reform_layer.bias.data = torch.zeros(train_data.fea.shape[1])

    optimizer = torch.optim.Adam(fpn_model.parameters(), lr=param_config.lr, weight_decay=0.003)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    fpn_train_acc = []
    fpn_valid_acc = []

    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    model_save_file = f"{data_save_dir}/bpfnn_{param_config.dataset_folder}_" \
                      f"rule_{param_config.n_rules}_lr_{param_config.lr:.6f}_" \
                      f"k_{current_k}.pkl"
    # load the exist model
    # if os.path.exists(model_save_file):
    #     fpn_model.load_state_dict(torch.load(model_save_file))
    best_test_rslt = 0
    for epoch in range(epochs):
        fpn_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs_temp = fpn_model(data, True)
            loss = loss_fn(outputs_temp, labels.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fpn_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = fpn_model(data, False)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            _, predicted_train = torch.max(outputs_train, 1)
            correct_train_num = (predicted_train == gnd_train.squeeze()).squeeze().sum()
            acc_train = correct_train_num.float()/gnd_train.shape[0]
            fpn_train_acc.append(acc_train)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = fpn_model(data, False)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            _, predicted_val = torch.max(outputs_val, 1)
            correct_val_num = (predicted_val == gnd_val.squeeze()).squeeze().sum()
            acc_val = correct_val_num/gnd_val.shape[0]
            fpn_valid_acc.append(acc_val)
        # param_config.log.info(f"{fpn_model.fire_strength[0:5, :]}")
        if best_test_rslt < acc_train:
            best_test_rslt = acc_train
            torch.save(fpn_model.state_dict(), model_save_file)
        param_config.log.info(
            f"fpn epoch : {epoch + 1}, train acc : {fpn_train_acc[-1]}, test acc : {fpn_valid_acc[-1]}")

    # mlp model
    mlp_model: nn.Module = MlpCls(train_data.fea.shape[1], n_cls, param_config.device)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=param_config.lr)
    loss_fn = nn.CrossEntropyLoss()
    epochs = param_config.n_epoch

    mlp_train_acc = []
    mlp_valid_acc = []

    for epoch in range(epochs):
        mlp_model.train()

        for i, (data, labels) in enumerate(train_loader):
            # data = data.cuda()
            # labels = labels.cuda()
            outputs = mlp_model(data)
            loss = loss_fn(outputs, labels.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mlp_model.eval()
        outputs_train = torch.empty(0, n_cls).to(param_config.device)
        outputs_val = torch.empty(0, n_cls).to(param_config.device)

        gnd_train = torch.empty(0, 1).to(param_config.device)
        gnd_val = torch.empty(0, 1).to(param_config.device)
        with torch.no_grad():
            for i, (data, labels) in enumerate(train_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = mlp_model(data)
                outputs_train = torch.cat((outputs_train, outputs_temp), 0)
                gnd_train = torch.cat((gnd_train, labels), 0)
            _, predicted_train = torch.max(outputs_train, 1)
            correct_train_num = (predicted_train == gnd_train.squeeze()).squeeze().sum()
            acc_train = correct_train_num / gnd_train.shape[0]
            mlp_train_acc.append(acc_train)
            for i, (data, labels) in enumerate(valid_loader):
                # data = data.cuda()
                # labels = labels.cuda()
                outputs_temp = mlp_model(data)
                outputs_val = torch.cat((outputs_val, outputs_temp), 0)
                gnd_val = torch.cat((gnd_val, labels), 0)
            _, predicted_val = torch.max(outputs_val, 1)
            correct_val_num = (predicted_val == gnd_val.squeeze()).squeeze().sum()
            acc_val = correct_val_num / gnd_val.shape[0]
            mlp_valid_acc.append(acc_val)

        param_config.log.info(
            f"mlp epoch : {epoch + 1}, train acc : {mlp_train_acc[-1]}, test acc : {mlp_valid_acc[-1]}")

    param_config.log.info("finished")
    plt.figure(0)
    title = f"FPN Acc of {param_config.dataset_folder}, prototypes:{param_config.n_rules}"
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    # plt.plot(torch.arange(len(mlp_train_acc)), torch.tensor(mlp_train_acc), 'b--', linewidth=2, markersize=5)
    # plt.plot(torch.arange(len(mlp_valid_acc)), torch.tensor(mlp_valid_acc), 'r--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_acc)), svm_train_acc.cpu().expand_as(torch.tensor(fpn_valid_acc)),
             'k-', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_acc)), svm_test_acc.cpu().expand_as(torch.tensor(fpn_valid_acc)),
             'k--', linewidth=2, markersize=5)
    plt.plot(torch.arange(len(fpn_valid_acc)), torch.tensor(fpn_train_acc).cpu(), 'r-', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(fpn_valid_acc)), torch.tensor(fpn_valid_acc).cpu(), 'r--', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(mlp_valid_acc)), torch.tensor(mlp_train_acc).cpu(), 'b-', linewidth=2,
             markersize=5)
    plt.plot(torch.arange(len(mlp_valid_acc)), torch.tensor(mlp_valid_acc).cpu(), 'b--', linewidth=2,
             markersize=5)
    plt.legend(['svm train', 'svm test', 'fpn train', 'fpn test', 'mlp train', 'mlp test'])
    # plt.legend(['fnn train', 'fnn test', 'fpn train', 'fpn test'])
    # plt.legend(['mlp train', 'mlp test', 'fpn train', 'fpn test'])
    # plt.legend(['fpn train', 'fpn test'])
    plt.show()

    # save all the results
    save_dict = dict()
    save_dict["fpn_train_acc"] = torch.tensor(fpn_train_acc).numpy()
    save_dict["fpn_valid_acc"] = torch.tensor(fpn_valid_acc).numpy()
    save_dict["mlp_train_acc"] = torch.tensor(mlp_train_acc).numpy()
    save_dict["mlp_valid_acc"] = torch.tensor(mlp_valid_acc).numpy()
    save_dict["svm_train_acc"] = svm_train_acc.cpu().numpy()
    save_dict["svm_test_acc"] = svm_test_acc.cpu().numpy()
    data_save_file = f"{data_save_dir}/mse_bpfnn_{param_config.dataset_folder}_rule" \
                     f"_{param_config.n_rules}_lr_{param_config.lr:.6f}" \
                     f"_k_{current_k}.mat"
    io.savemat(data_save_file, save_dict)
    return fpn_train_acc, fpn_valid_acc

