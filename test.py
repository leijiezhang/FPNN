import pandas as pd
import numpy as np
from pandas import DataFrame
import seaborn as sns
from matplotlib import gridspec
sns.set(style='darkgrid')
alg = ["RC", 'CC', "GA","NN","Free","PC","Forward"]
colors = ["#e74c3c", "#3498db","#9b59b6", "#34495e","#2ecc71","#ffff99", "#95a5a6"]
models = ['linear', 'mlp']
dataset = ['mnist', 'fashion', 'kmnist', 'cifar10']
data_lists = []
count = 0
for model in models:
    for data in dataset:
        if data == 'cifar10':
            if model == 'linear':
                model = 'resnet'
            else:
                model = 'densenet'
        # print(model)
        # print(data)
        GA = []
        NN = []
        Free = []
        PC = []
        Forward = []
        RC = []
        CC = []
        trial_list = [1, 2, 3, 4, 5]

        for i in trial_list:
            i = str(i)
            RC.append(
                pd.read_csv("results/trial_" + i + "/" + data + "_" + model + "/backward" + model + data + "1.txt",
                            sep=" ", index_col=None))
            CC.append(pd.read_csv("results/trial_" + i + "/" + data + "_" + model + "/myfor" + model + data + "1.txt",
                                  sep=" ", index_col=None))
            GA.append(
                pd.read_csv("results/trial_" + i + "/" + data + "_" + model + "/ga" + model + data + "1.txt", sep=" ",
                            index_col=None))
            NN.append(
                pd.read_csv("results/trial_" + i + "/" + data + "_" + model + "/nn" + model + data + "1.txt", sep=" ",
                            index_col=None))
            Free.append(
                pd.read_csv("results/trial_" + i + "/" + data + "_" + model + "/free" + model + data + "1.txt", sep=" ",
                            index_col=None))
            PC.append(
                pd.read_csv("results/trial_" + i + "/" + data + "_" + model + "/pc" + model + data + "1.txt", sep=" ",
                            index_col=None))
            Forward.append(
                pd.read_csv("results/trial_" + i + "/" + data + "_" + model + "/forward" + model + data + "1.txt",
                            sep=" ", index_col=None))

        data_list = [RC, CC, GA, NN, Free, PC, Forward]
        data_lists.append(data_list)
df_lists = []
for j in range(8):
    df_list = []
    for i in range(len(trial_list)):
        df = pd.DataFrame(columns=alg)
        for name, data in zip(alg, data_lists[j]):
            df[name] = data[i].iloc[:,2]
            df['epoch'] = np.arange(len(df))
        df_list.append(df)
    df_lists.append(df_list)

df_all1 = pd.concat(df_lists[0],axis=0)
df_all2 = pd.concat(df_lists[1],axis=0)
df_all3 = pd.concat(df_lists[2],axis=0)
df_all4 = pd.concat(df_lists[3],axis=0)
df_all5 = pd.concat(df_lists[4],axis=0)
df_all6 = pd.concat(df_lists[5],axis=0)
df_all7 = pd.concat(df_lists[6],axis=0)
df_all8 = pd.concat(df_lists[7],axis=0)

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=[16, 8]
plt.subplots_adjust(wspace=0.3, hspace=0.3)
ax1 = plt.subplot(241)
for i in range(len(alg)):
    a = alg[i]
    c = colors[i]
    sns.lineplot(x="epoch", y=a, data=df_all1, color=c, ci='sd', label=a)
ax1.set_xlim((0, 250))
ax1.set_ylim((0.77, 0.95))
ax1.set_ylabel('Test Accuracy',fontsize=10)
ax1.set_xlabel('Epoch',fontsize=10)
ax1.locator_params('y',nbins=5)
ax1.locator_params('x',nbins=5)
ax1.legend_.remove()
ax1.set_title('MNIST, Linear')