from utils.param_config import ParamConfig
from utils.loss_utils import RMSELoss, LikelyLoss
from models.model_run_gpu import fpn_run, mlp_run_r, fpn_run_c
import torch
import os
import scipy.io as io


# Dataset configuration
# init the parameters statlib_calhousing_config
param_config = ParamConfig()
param_config.config_parse('sdd_config')

param_config.log.info(f"dataset : {param_config.dataset_folder}")
param_config.log.info(f"prototype number : {param_config.n_rules}")
param_config.log.info(f"batch_size : {param_config.n_batch}")
param_config.log.info(f"epoch_size : {param_config.n_epoch}")


for i in torch.arange(len(param_config.dataset_list)):
    # load dataset
    dataset = param_config.get_dataset_mat(int(i))

    param_config.log.debug(f"=====starting on {dataset.name}=======")
    loss_fun = None
    if dataset.task == 'C':
        param_config.log.war(f"=====Mission: Classification=======")
        param_config.loss_fun = LikelyLoss()
    else:
        param_config.log.war(f"=====Mispara_consq_bias_rsion: Regression=======")
        param_config.loss_fun = RMSELoss()

    fpn_train_loss_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
    fpn_test_loss_tsr = torch.zeros(param_config.n_epoch, 0).to(param_config.device)
    mlp_train_loss_tsr = torch.empty(param_config.n_epoch, 0).to(param_config.device)
    mlp_test_loss_tsr = torch.zeros(param_config.n_epoch, 0).to(param_config.device)
    fnn_train_loss_tsr = torch.empty(1, 0).to(param_config.device)
    fnn_test_loss_tsr = torch.zeros(1, 0).to(param_config.device)

    for kfold_idx in torch.arange(param_config.n_kfolds):
        param_config.log.war(f"=====k_fold: {kfold_idx + 1}=======")
        train_data, test_data = dataset.get_kfold_data(kfold_idx)

        fpn_train_loss, fpn_test_loss, mlp_train_loss, mlp_test_loss, fnn_train_loss, fnn_test_loss = \
            fpn_run_c(param_config, train_data, test_data, kfold_idx + 1)

        fpn_test_loss_tsr = torch.cat([fpn_test_loss_tsr, torch.tensor(fpn_test_loss).unsqueeze(1)], 1)
        fpn_train_loss_tsr = torch.cat([fpn_train_loss_tsr, torch.tensor(fpn_train_loss).unsqueeze(1)], 1)
        mlp_test_loss_tsr = torch.cat([mlp_test_loss_tsr, torch.tensor(mlp_test_loss).unsqueeze(1)], 1)
        mlp_train_loss_tsr = torch.cat([mlp_train_loss_tsr, torch.tensor(mlp_train_loss).unsqueeze(1)], 1)
        fnn_test_loss_tsr = torch.cat([fnn_test_loss_tsr, torch.tensor(fnn_test_loss).unsqueeze(1)], 1)
        fnn_train_loss_tsr = torch.cat([fnn_train_loss_tsr, torch.tensor(fnn_train_loss).unsqueeze(1)], 1)
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    save_dict = dict()
    save_dict["fpn_test_loss_tsr"] = fpn_test_loss_tsr.numpy()
    save_dict["fpn_train_loss_tsr"] = fpn_train_loss_tsr.numpy()
    save_dict["mlp_test_loss_tsr"] = mlp_test_loss_tsr.numpy()
    save_dict["mlp_train_loss_tsr"] = mlp_train_loss_tsr.numpy()
    save_dict["fnn_test_loss_tsr"] = fnn_test_loss_tsr.numpy()
    save_dict["fnn_train_loss_tsr"] = fnn_train_loss_tsr.numpy()

    data_save_file = f"{data_save_dir}/mse_fpn_{param_config.dataset_folder}" \
                     f"_rule{param_config.n_rules}_epoch_{param_config.n_epoch}.mat"
    io.savemat(data_save_file, save_dict)
