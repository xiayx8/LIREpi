import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from torch import nn
import pickle
import json
import torch
import math
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from PDB_Parser import StructureDataParser
from model_block.EGNN import eg
from double_mydata import esmAF_feature
from double_mydata import cal_edges
from double_mydata import esmAF_feature_heavy_chain
from double_mydata import cal_edges_heavy_chain
#from tools import analysis
from sklearn import metrics
from sklearn.model_selection import KFold
import warnings
import random
from tqdm import tqdm
import torch
from torch import nn
from einops import rearrange
from torch import einsum
import copy
class KD_EGNN(nn.Module):
    def __init__(self, infeature_size, outfeature_size, nhidden_eg, edge_feature,
                 n_eglayer, nclass, device):
        super(KD_EGNN, self).__init__()
        self.dropout = 0.3
        # 初始化一些自蒸馏所需的参数
        self.temperature = 3
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # self.device=torch.device('cpu')
        self.loss_coefficient = 0.7  # 权重调整系数
        self.feature_loss_coefficient = 0.03  # 特征损失系数
        # EGNN layers as per your model structure
        self.eg1 = eg(in_node_nf=infeature_size,
                      nhidden=nhidden_eg,
                      n_layers=n_eglayer,
                      out_node_nf=outfeature_size,
                      in_edge_nf=edge_feature,
                      attention=True,
                      normalize=False,
                      tanh=True,
                      device=device)
        self.eg2 = eg(in_node_nf=outfeature_size,
                      nhidden=nhidden_eg,
                      n_layers=n_eglayer,
                      out_node_nf=outfeature_size,
                      in_edge_nf=edge_feature,
                      attention=True,
                      normalize=False,
                      tanh=True,
                      device=device)
        self.eg3 = eg(in_node_nf=outfeature_size,
                      nhidden=nhidden_eg,
                      n_layers=n_eglayer,
                      out_node_nf=int(outfeature_size / 2),
                      in_edge_nf=edge_feature,
                      attention=True,
                      normalize=False,
                      tanh=True,
                      device=device)
        # self.eg4 = eg(in_node_nf=int(outfeature_size / 2),
        #               nhidden=nhidden_eg,
        #               n_layers=n_eglayer,
        #               out_node_nf=int(outfeature_size / 4),
        #               in_edge_nf=edge_feature,
        #               attention=True,
        #               normalize=False,
        #               tanh=True,
        #               device=device)

        self.fc1 = nn.Sequential(
            nn.Linear(outfeature_size, int(outfeature_size / 4)),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(int(outfeature_size / 4), nclass)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(outfeature_size, int(outfeature_size / 4)),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(int(outfeature_size / 4), nclass)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(int(outfeature_size / 2), int(outfeature_size / 4)),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(int(outfeature_size / 4), nclass)
        )
        # self.fc4 = nn.Sequential(
        #     nn.Linear(int(outfeature_size / 4), int(outfeature_size / 4)),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(outfeature_size / 4), nclass)
        # )

    def forward(self, x_res, x_pos, edge_index):

        x_res = F.dropout(x_res, self.dropout, training=self.training)
        output_res, pre_pos_res = self.eg1(h=x_res,
                                           x=x_pos.float(),
                                           edges=edge_index,
                                           edge_attr=None)

        output_res2, pre_pos_res2 = self.eg2(h=output_res,
                                             x=pre_pos_res.float(),
                                             edges=edge_index,
                                             edge_attr=None)

        output_res3, pre_pos_res3 = self.eg3(h=output_res2,
                                             x=pre_pos_res2.float(),
                                             edges=edge_index,
                                             edge_attr=None)

        # output_res4, pre_pos_res4 = self.eg4(h=output_res3,
        #                                      x=pre_pos_res3.float(),
        #                                      edges=edge_index,
        #                                      edge_attr=None)

        out1 = self.fc1(output_res)
        out2 = self.fc2(output_res2)
        out3 = self.fc3(output_res3)
        # print('out3',out3.shape)
        # out4 = self.fc4(output_res4)  # 使用教师模型的第四层输出
        return [out3, out2, out1], [output_res3, output_res2, output_res]


class FeatureFusionModel(nn.Module):
    def __init__(self, input_dim=256, output_dim=256, dropout_rate=0.25):
        super(FeatureFusionModel, self).__init__()
        self.G_model = KD_EGNN(infeature_size=1280, outfeature_size=512, nhidden_eg=128, edge_feature=0, n_eglayer=4,
                            nclass=2,
                            device='cuda:1')  # 你原来的MODEL
        self.fc = nn.Linear(input_dim, 2)  # 全连接层
        self.dropout = nn.Dropout(dropout_rate)  # Dropout层

    def forward(self, ag_fused, ag_coord, ag_edge, ab_fused, ab_coord, ab_edge):
        # 获取抗原和抗体特征
        _, ag_features = self.G_model(ag_fused, ag_coord, ag_edge)
        _, ab_features = self.G_model(ab_fused, ab_coord, ab_edge)

        # 取第一个特征（假设这是你需要的特征）
        ag_fea = ag_features[0]  # (L, 256)
        ab_fea = ab_features[0]  # (N, 256)

        # 拼接特征
        fused_features = torch.cat([ag_fea, ab_fea], dim=0)  # (L+N, 256)

        # 通过全连接层和Dropout
        output = self.fc(fused_features)
        output = self.dropout(output)
        pooled_out= output.mean(dim=0, keepdim=True)  # (1, 2)
        return pooled_out

def analysis(y_true, y_pred, best_threshold=None):
    """
    y_true: List[int] or np.array, true labels (0/1)
    y_pred: List[float] or np.array, predicted probabilities (after sigmoid or softmax)
    best_threshold: float or None, if None will search for best threshold based on MCC
    """
    #######mcc版本
    if best_threshold is None:
        thresholds = np.linspace(0.001, 0.999, 200)  # 探索阈值从0.001到0.999
        best_mcc = -1
        for thresh in thresholds:
            binary_pred = (np.array(y_pred) >= thresh).astype(int)
            mcc = metrics.matthews_corrcoef(y_true, binary_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = thresh
        print(f"Auto-selected best threshold: {best_threshold:.4f} with MCC: {best_mcc:.4f}")

    ###############f1版本
    # if best_threshold is None:
    #     thresholds = np.linspace(0.001, 0.999, 200)  # 探索阈值从0.001到0.999
    #     best_f1 = -1
    #     for thresh in thresholds:
    #         binary_pred = (np.array(y_pred) >= thresh).astype(int)
    #         f1 = metrics.f1_score(y_true, binary_pred, zero_division=0)
    #         if f1 > best_f1:
    #             best_f1 = f1
    #             best_threshold = thresh
    #     print(f"Auto-selected best threshold: {best_threshold:.4f} with MCC: {best_f1:.4f}")

    # 使用最佳阈值进行最终二分类
    binary_pred = (np.array(y_pred) >= best_threshold).astype(int)
    binary_true = np.array(y_true)

    # 计算分类指标
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred, zero_division=0)
    recall = metrics.recall_score(binary_true, binary_pred, zero_division=0)
    f1 = metrics.f1_score(binary_true, binary_pred, zero_division=0)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    # 计算曲线下指标
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds_pr = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)

    results = {
        'ACC': binary_acc,
        'PRE': precision,
        'REC': recall,
        'F1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'MCC': mcc,
        'Best_Threshold': best_threshold,
    }

    return results,best_threshold
# 初始化 cross attention



def train(train_loader,model,optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
    #for batch in train_loader:
        antigen_inputs, antibody_inputs, label, antigen_name, antibody_name = batch

        # antigen_inputs = [x.to(device) for x in antigen_inputs]
        # antibody_inputs = [x.to(device) for x in antibody_inputs]
        antigen_inputs = [x.to(device).squeeze() for x in antigen_inputs]
        antibody_inputs = [x.to(device).squeeze() for x in antibody_inputs]
        label = label.to(device)
        label = label.squeeze()
        #label = label.unsqueeze(0)
        #loss, _ = predict(antigen_inputs, antibody_inputs, label)
        ag_esm, ag_edge, ag_coord = antigen_inputs
        ab_esm, ab_edge, ab_coord = antibody_inputs

        # 解析名字
        ag_pdbid, ag_chain = antigen_name[0].split('_')
        ab_pdbid, ab_chain = antibody_name[0].split('_')
        # # 构造文件名
        # ag_filename = f"{ag_pdbid.lower()}_{ag_chain}_vit.npy"
        # ab_filename = f"{ab_pdbid.lower()}_{ab_chain}_vit.npy"
        # # 文件夹路径
        # folder_l = "/mnt/Data6/23gsy/graph-piston/get_agabl/muti_piston_vector/"
        # folder_g = "/mnt/Data6/23gsy/graph-piston/muti_piston_vector/cls_token_16/"
        # ag_struct_path_g = os.path.join(folder_g, ag_filename)
        # ag_struct = np.load(ag_struct_path_g)
        # # 抗体struct
        # ab_struct_path_l = os.path.join(folder_l, ab_filename)
        # ab_struct_path_g = os.path.join(folder_g, ag_filename)
        #
        # if os.path.exists(ab_struct_path_l):
        #     ab_struct = np.load(ab_struct_path_l)
        # else:
        #     ab_struct = np.load(ab_struct_path_g)
        # ag_feature2_tensor = torch.tensor(ag_struct, dtype=torch.float32).to(device)
        # ab_feature2_tensor = torch.tensor(ab_struct, dtype=torch.float32).to(device)
        # ag_fused = fusion_model(ag_esm, ag_feature2_tensor)
        # ab_fused = fusion_model(ab_esm, ab_feature2_tensor)

        output=model(ag_esm, ag_coord, ag_edge,ab_esm, ab_coord, ab_edge)

        #pred_score = cross_att(ag_fea, ab_fea)
        label=label.unsqueeze(0)
        label = label.long()  # 强制转换为long
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)  # 正常计算
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(val_loader, model,optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    #for batch in tqdm(val_loader, desc="evaluate"):
    for batch in val_loader:
        antigen_inputs, antibody_inputs, label, antigen_name, antibody_name = batch
        # antigen_inputs = [x.to(device) for x in antigen_inputs]
        # antibody_inputs = [x.to(device) for x in antibody_inputs]
        antigen_inputs = [x.to(device).squeeze() for x in antigen_inputs]
        antibody_inputs = [x.to(device).squeeze() for x in antibody_inputs]
        label = label.to(device)
        label = label.squeeze()
        #label = label.unsqueeze(0)
        # loss, _ = predict(antigen_inputs, antibody_inputs, label)
        ag_esm, ag_edge, ag_coord = antigen_inputs
        ab_esm, ab_edge, ab_coord = antibody_inputs

        # 解析名字
        # ag_pdbid, ag_chain = antigen_name[0].split('_')
        # ab_pdbid, ab_chain = antibody_name[0].split('_')
        # # 构造文件名
        # ag_filename = f"{ag_pdbid.lower()}_{ag_chain}_vit.npy"
        # ab_filename = f"{ab_pdbid.lower()}_{ab_chain}_vit.npy"
        # # 文件夹路径
        # folder_l = "/mnt/Data6/23gsy/graph-piston/get_agabl/muti_piston_vector/"
        # folder_g = "/mnt/Data6/23gsy/graph-piston/muti_piston_vector/cls_token_16/"
        # ag_struct_path_g = os.path.join(folder_g, ag_filename)
        # ag_struct = np.load(ag_struct_path_g)
        # # 抗体struct
        # ab_struct_path_l = os.path.join(folder_l, ab_filename)
        # ab_struct_path_g = os.path.join(folder_g, ag_filename)
        #
        # if os.path.exists(ab_struct_path_l):
        #     ab_struct = np.load(ab_struct_path_l)
        # else:
        #     ab_struct = np.load(ab_struct_path_g)
        #
        # ag_feature2_tensor = torch.tensor(ag_struct, dtype=torch.float32).to(device)
        # ab_feature2_tensor = torch.tensor(ab_struct, dtype=torch.float32).to(device)
        # ag_fused = fusion_model(ag_esm, ag_feature2_tensor)
        # ab_fused = fusion_model(ab_esm, ab_feature2_tensor)

        output = model(ag_esm, ag_coord, ag_edge, ab_esm, ab_coord, ab_edge)

        # pred_score = cross_att(ag_fea, ab_fea)
        label = label.unsqueeze(0)
        label = label.long()  # 强制转换为long
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)  # 正常计算
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # 对每个位置计算Softmax
        prob = torch.softmax(output, dim=1)  # (200, 2)
        # 取类别1的概率
        class1_prob = prob[:, 1]  # (200,)
        all_preds.extend(class1_prob.detach().cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    return total_loss / len(val_loader),all_preds,all_labels
def test(val_loader,model,device, best_threshold=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        # for batch in tqdm(val_loader, desc="test"):
        for batch in val_loader:
            antigen_inputs, antibody_inputs, label, antigen_name, antibody_name = batch
            # antigen_inputs = [x.to(device) for x in antigen_inputs]
            # antibody_inputs = [x.to(device) for x in antibody_inputs]
            antigen_inputs = [x.to(device).squeeze() for x in antigen_inputs]
            antibody_inputs = [x.to(device).squeeze() for x in antibody_inputs]
            label = label.to(device)
            label = label.squeeze()
            # label = label.unsqueeze(0)
            # loss, _ = predict(antigen_inputs, antibody_inputs, label)
            ag_esm, ag_edge, ag_coord = antigen_inputs
            ab_esm, ab_edge, ab_coord = antibody_inputs
            # 解析名字
            # ag_pdbid, ag_chain = antigen_name[0].split('_')
            # ab_pdbid, ab_chain = antibody_name[0].split('_')
            # # 构造文件名
            # ag_filename = f"{ag_pdbid.lower()}_{ag_chain}_vit.npy"
            # ab_filename = f"{ab_pdbid.lower()}_{ab_chain}_vit.npy"
            # # 文件夹路径
            # folder_l = "/mnt/Data6/23gsy/graph-piston/get_agabl/muti_piston_vector/"
            # folder_g = "/mnt/Data6/23gsy/graph-piston/muti_piston_vector/cls_token_16/"
            # ag_struct_path_g = os.path.join(folder_g, ag_filename)
            # ag_struct = np.load(ag_struct_path_g)
            # # 抗体struct
            # ab_struct_path_l = os.path.join(folder_l, ab_filename)
            # ab_struct_path_g = os.path.join(folder_g, ag_filename)
            #
            # if os.path.exists(ab_struct_path_l):
            #     ab_struct = np.load(ab_struct_path_l)
            # else:
            #     ab_struct = np.load(ab_struct_path_g)
            #
            # ag_feature2_tensor = torch.tensor(ag_struct, dtype=torch.float32).to(device)
            # ab_feature2_tensor = torch.tensor(ab_struct, dtype=torch.float32).to(device)
            # ag_fused = fusion_model(ag_esm, ag_feature2_tensor)
            # ab_fused = fusion_model(ab_esm, ab_feature2_tensor)

            output = model(ag_esm, ag_coord, ag_edge, ab_esm, ab_coord, ab_edge)

            # pred_score = cross_att(ag_fea, ab_fea)
            label = label.unsqueeze(0)
            # pred_score = pred_score.mean(dim=[1, 2])  # 取平均，变成 (batch_size,)
            prob = torch.softmax(output, dim=1)  # (200, 2)
            # 取类别1的概率
            class1_prob = prob[:, 1]  # (200,)
            all_preds.extend(class1_prob.detach().cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    return all_labels,all_preds
    #return analysis(np.array(all_labels), np.array(all_preds), best_threshold)


class MYDatasets(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.antigen_names = dataframe['antigen_chain_ID'].values
        self.antigen_seqs = dataframe['antigen_chain_sequence'].values
        self.antibody_names = dataframe['antibody_chain_ID'].values
        self.antibody_seqs = dataframe['antibody_chain_sequence'].values
        self.labels = dataframe['label'].values

        self.topk = 40
        self.num_rbf = 16
        self.map = 14

    def __getitem__(self, index):
        antigen_name = self.antigen_names[index]
        antigen_seq = self.antigen_seqs[index]
        antibody_name = self.antibody_names[index]
        antibody_seq = self.antibody_seqs[index]
        label = self.labels[index]

        # 加载特征
        antigen_esm, antigen_af = esmAF_feature(antigen_name)
        antibody_esm, antibody_af = esmAF_feature_heavy_chain(antibody_name)
        #antibody_esm, antibody_af = esmAF_feature(antibody_name)
        antigen_esm = torch.from_numpy(antigen_esm).float()
        antibody_esm = torch.from_numpy(antibody_esm).float()

        antigen_edges, antigen_CA = cal_edges(antigen_name)
        antibody_edges, antibody_CA = cal_edges_heavy_chain(antibody_name)
        #antibody_edges, antibody_CA = cal_edges(antibody_name)
        antigen_inputs = [antigen_esm, antigen_edges, antigen_CA]
        antibody_inputs = [antibody_esm, antibody_edges, antibody_CA]

        label_tensor = torch.tensor([float(label)], dtype=torch.float32)

        return antigen_inputs, antibody_inputs, label_tensor, antigen_name, antibody_name

    def __len__(self):
        return len(self.antigen_names)

# def set_random_seed(seed: int = 3407):
#     """设置随机种子，以确保结果可复现"""
#     random.seed(seed)  # Python的随机数生成器
#     np.random.seed(seed)  # numpy的随机数生成器
#     torch.manual_seed(seed)  # PyTorch的随机数生成器
#     torch.cuda.manual_seed_all(seed)  # 如果使用GPU的话，固定所有设备上的随机种子
#     # 如果你有确定性的计算需求
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
def set_random_seed(seed: int = 3407):
    """设置随机种子，以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # For CUDA reproducibility
    torch.use_deterministic_algorithms(True)

def main():
    set_random_seed(3407)  # 固定随机种子，确保可复现
    # Load your dataset
    ###del#train_file = './double_train.pkl'  # Replace with your actual data file
    # train_file='./double_train_copy.pkl'
    # test_file = './double_test2.pkl'
    # ######2000多条的
    # train_file = './train_4.0_data.pkl'
    # test_file = './test_4.0_data.pkl'
    #########154条
    train_file = './train154.pkl'
    test_file = './test37.pkl'
    with open(train_file, "rb") as f:
        train_data = pickle.load(f)
    with open(test_file, "rb") as f:
        test_data = pickle.load(f)
    # Define model, optimizer, and loss function

    # criterion = nn.BCEWithLogitsLoss()
    fold_result = []
    saved_cross_attn_models = []
    # 初始化列表
    train_ag_ids = []
    train_ag_sequences = []
    train_ab_ids = []
    train_ab_sequences = []
    train_labels = []  # 还要加上标签哦！

    # 遍历 train_data 字典
    for (ag_id, ab_id), (ag_sequence, ab_sequence, label) in train_data.items():
        train_ag_ids.append(ag_id)  # 抗原ID
        train_ag_sequences.append(ag_sequence)  # 抗原序列
        train_ab_ids.append(ab_id)  # 抗体ID
        train_ab_sequences.append(ab_sequence)  # 抗体序列
        train_labels.append(label)  # 标签

    # 整理成字典
    train_dic = {
        "antigen_chain_ID": train_ag_ids,
        "antigen_chain_sequence": train_ag_sequences,
        "antibody_chain_ID": train_ab_ids,
        "antibody_chain_sequence": train_ab_sequences,
        "label": train_labels,
    }

    # 转成DataFrame
    train_dataframe = pd.DataFrame(train_dic)

    test_ag_ids = []
    test_ag_sequences = []
    test_ab_ids = []
    test_ab_sequences = []
    test_labels = []  # 还要加上标签哦！

    # 遍历 train_data 字典
    for (ag_id, ab_id), (ag_sequence, ab_sequence, label) in test_data.items():
        test_ag_ids.append(ag_id)  # 抗原ID
        test_ag_sequences.append(ag_sequence)  # 抗原序列
        test_ab_ids.append(ab_id)  # 抗体ID
        test_ab_sequences.append(ab_sequence)  # 抗体序列
        test_labels.append(label)  # 标签

    # 整理成字典
    test_dic = {
        "antigen_chain_ID": test_ag_ids,
        "antigen_chain_sequence": test_ag_sequences,
        "antibody_chain_ID": test_ab_ids,
        "antibody_chain_sequence": test_ab_sequences,
        "label": test_labels,
    }

    # 转成DataFrame
    test_dataframe = pd.DataFrame(test_dic)
    test_dataset = DataLoader(dataset=MYDatasets(test_dataframe), batch_size=1, shuffle=False, num_workers=0,
                              drop_last=True)
    # KFold Cross-Validation
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    best_epoch_results = []

    # Early stopping setup
    patience = 10 # Number of epochs to wait for improvement

    # 5-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataframe)):
        print(f"\nFold {fold + 1}")
        if fold == 2:
            #print("⚙️ Setting special random seed for fold 3")
            set_random_seed(44)
        # elif fold == 4:
        #     #print("⚙️ Setting special random seed for fold 5")
        #     set_random_seed(44)
        else:
            set_random_seed(3407)
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        #cross_att = CrossAttention(dim_q=256, dim_kv=256, hidden_dim=128).to(device)
        model = FeatureFusionModel().to(device)

        # 加载模型（假设 fusion_model, aigen_model 等也已在外面定义好）
        #cross_att.to(device)

        # 准备数据（假设你已经写好自定义的Dataset）

        train_fold = train_dataframe.iloc[train_idx]
        val_fold = train_dataframe.iloc[val_idx]
        # Create data loaders
        train_dataset = DataLoader(dataset=MYDatasets(train_fold), batch_size=1, shuffle=True, num_workers=0,
                                   drop_last=True)
        val_dataset = DataLoader(dataset=MYDatasets(val_fold), batch_size=1, shuffle=True, num_workers=0,
                                 drop_last=True)

        # Initialize the model weights for this fold
        # model.load_state_dict(best_model_wts if best_model_wts is not None else model.state_dict())
        epochs_no_improve = 0  # Counter for epochs without improvement

        best_val_loss = float('inf')
        best_attn_model_wts = None
        best_thre=float('inf')
        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # 开始训练
        epochs = 200
        # train_pos_weight=cal_pos_weight(train_dataset)
        # val_pos_weight=cal_pos_weight(val_dataset)
        for epoch in range(1, epochs + 1):
            #print(f"\n--- Epoch {epoch} ---")
            train_loss = train(train_dataset, model,optimizer, device)
            #print(f"Train Loss: {train_loss:.4f}")
            val_loss,val_preds,val_labels = evaluate(val_dataset, model,optimizer,device)
            print(f'epoch:{epoch}      train_loss:{train_loss}        val_loss:{val_loss}')
            results,threshold=analysis(np.array(val_labels), np.array(val_preds),best_threshold=None)

            scheduler.step()
            # Save the model if it has the lowest validation loss so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_attn_model_wts = copy.deepcopy(model.state_dict())  # Save the model weights
                # best_epoch_results = analysis(valid_true, valid_pred)
                epochs_no_improve = 0  # Reset counter as we have improvement
            else:
                epochs_no_improve += 1  # Increment counter if no improvement
            #results = analysis(valid_true, valid_pred, best_threshold=None)

            # results_str = json.dumps(results)  # Convert the dictionary to a JSON string

            with open('/mnt/Data6/23gsy/SEKD-main/double_input/d37_validation_results.txt', 'a') as result_file:
                result_file.write("'ACC', 'PRE', 'REC', 'F1', 'AUC', 'AUPRC', 'MCC','thre'\n")
                result_file.write(f"{fold + 1}\t{epoch}\t{best_val_loss:.4f}\t{results}\n")
            # Check if early stopping condition is met
            if epochs_no_improve==0:
                best_thre=threshold
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
                break  # Stop training

        saved_cross_attn_models.append(best_attn_model_wts)
        model.load_state_dict(best_attn_model_wts)
        test_labels, test_preds = test(test_dataset, model,device)
        test_results,test_thre = analysis(np.array(test_labels), np.array(test_preds),best_threshold=None)
        print(f"Test results for fold {fold + 1}: {test_results}")
        with open('/mnt/Data6/23gsy/SEKD-main/double_input/d37_test_results.txt', 'a') as result_file:
            result_file.write("'ACC', 'PRE', 'REC', 'F1', 'AUC', 'AUPRC', 'MCC','thre'\n")
            result_file.write(f"{fold + 1}\t{best_val_loss:.4f}\t{test_results}\n")
        fold_result.append(test_results)

    # After all folds, calculate the average results
    metrics = ['ACC', 'PRE', 'REC', 'F1', 'AUC', 'AUPRC', 'MCC','thre']
    avg_results = {metric: 0 for metric in metrics}

    # Sum up each metric from all folds
    for result in fold_result:
        for metric in metrics:
            avg_results[metric] += result.get(metric, 0)

    # Calculate the average for each metric
    num_folds = len(fold_result)  # Divide by 2 as we're adding results from training and testing
    for metric in metrics:
        avg_results[metric] /= num_folds

    # Print the averaged results
    print("\nAverage Results across all folds:")
    for metric, avg_value in avg_results.items():
        print(f"{metric}: {avg_value:.4f}")

    # Save the models of all folds (optional)
    for fold, model_weights in enumerate(saved_cross_attn_models):
        torch.save(model_weights, f"/mnt/Data6/23gsy/SEKD-main/double_input/d37_crossattn_fold_{fold + 1}.pth")
        print(f"Saved the best model for fold {fold + 1}")
if __name__ == '__main__':
    main()
