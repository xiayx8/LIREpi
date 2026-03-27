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
from MY_data import esmAF_feature
from MY_data import cal_edges
from tools import analysis
from sklearn.model_selection import KFold
import warnings
import random
warnings.filterwarnings("ignore")
import time  # 导入time模块


# Assuming the KD_EGNN class is defined as shown in your earlier messages
class KD_EGNN(nn.Module):
    def __init__(self, infeature_size, outfeature_size, nhidden_eg, edge_feature,
                 n_eglayer, nclass, device):
        super(KD_EGNN, self).__init__()
        self.dropout = 0.3
        # 初始化一些自蒸馏所需的参数
        self.temperature = 3
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

        # return [out4, out3, out2, out1], [output_res4, output_res3, output_res2, output_res]

#######特征融合
class FiLMFusion(nn.Module):
    def __init__(self, seq_dim=1280, struct_dim=16, out_dim=1280):
        """
        Args:
            seq_dim: 序列特征维度（例如1280）
            struct_dim: 结构特征维度（例如16）
            out_dim: 融合后希望得到的特征维度（这里设置为1280）
        """
        super(FiLMFusion, self).__init__()
        # 利用结构特征生成 scale缩放 和 shift平移 参数，总共2*seq_dim个输出
        self.film_generator = nn.Linear(struct_dim, 2 * seq_dim)
        # 保留 out_proj 层，但设置输出为1280
        self.out_proj = nn.Linear(seq_dim, out_dim)

    def forward(self, seq_feat, struct_feat):
        """
        Args:
            seq_feat: (L, seq_dim) 序列特征
            struct_feat: (1, struct_dim) 结构特征
        返回:
            融合后的特征: (L, out_dim)
        """
        L = seq_feat.size(0)
        # 扩展结构特征到每个序列位置
        struct_expanded = struct_feat.expand(L, -1)#做一个（序列长度，16）的扩展结构特征
        film_params = self.film_generator(struct_expanded)  # (L, 2*seq_dim)
        scale, shift = film_params.chunk(2, dim=-1)  #将线性层输出的 (L, 2*seq_dim) 切成两个 (L, seq_dim)，一个是 scale，一个是 shift。
        scale = 1 + scale#在缩放系数上加 1，这样不会破坏原特征的基准，而是相对地进行缩放。
        modulated = scale * seq_feat + shift  # 对序列特征进行逐元素的缩放和平移操作，实现特征“调制”。
        fused_feat = self.out_proj(modulated)  # (L, out_dim) == (L, 1280)
        return fused_feat

def distillation_loss(outputs, features, label, T=3, alpha=0.7, beta=0.03, N=3):
    """
    计算蒸馏损失，包括交叉熵损失、KL散度损失和L2损失。
    student_outputs: 学生模型的输出（四层）
    teacher_outputs: 教师模型的输出
    student_features: 学生模型的特征（四层）
    teacher_features: 教师模型的特征
    label: 真实标签
    T: 温度参数
    alpha: 学生模型和教师模型损失的权重（KL损失的系数）
    beta: L2损失的系数
    N: 总层数，学生模型输出的层数
    """
    student_outputs = outputs[1:]  # 这将是 [out3, out2, out1]
    teacher_output=outputs[0]
    student_features=features[1:]
    teacher_feature=features[0]
    loss_ce = 0
    loss_kl = 0
    loss_l2 = 0
    # 定义线性变换，将不同维度的特征调整为128维
    #linear_2 = nn.Linear(256, 128).to(device)  # 第二层：256 -> 128
    linear_3 = nn.Linear(512, 256).to(device)  # 第三层：512 -> 128
    linear_4 = nn.Linear(512, 256).to(device)  # 第四层：512 -> 128
    # 1. 计算交叉熵损失（学生模型与真实标签的分类损失）
    for i, output in enumerate(outputs):  # 对前三层计算交叉熵损失
        loss_ce += (1 - alpha) * F.cross_entropy(output, label)  # Cross-entropy for each layer

    # 2. 计算KL散度损失（学生模型与教师模型输出的差异）
    for student_output in student_outputs:  # 对前三层计算KL散度损失
        loss_kl += alpha * kd_loss(student_output, teacher_output, alpha, T)  # KL divergence with teacher
    # for feature in features:
    #     print(feature.shape)
    i=0
    for student_feature in student_features:  # 从第1层开始遍历
        
        # 对于第二、第三、第四层特征，进行线性变换使其维度与第一层相同
        if i == 0:
            feature_aligned = linear_3(student_feature)  # 第二层特征变换为128维
        elif i == 1:
            feature_aligned = linear_4(student_feature)  # 第三层特征变换为128维
        # elif i == 2:
        #     feature_aligned = linear_4(student_feature)  # 第四层特征变换为128维
        # 计算与第一层特征之间的L2损失
        loss_l2 += beta * F.mse_loss(feature_aligned, teacher_feature)  # 计算 L2 损失
        i+=1
    # 3. 计算L2损失（学生模型每层特征与教师模型最深层特征的差异）
    # loss_l2 += beta * F.mse_loss(teacher_features[0], student_features[-1])  # 最浅层与最深层特征图之间的L2损失

    # 总损失是加权的交叉熵损失、KL损失和L2损失的加和
    total_loss = loss_ce + loss_kl + loss_l2
    return total_loss


def kd_loss(student_output, teacher_output, alpha=0.7, T=3):
    """
    计算KL散度损失（用于学生模型与教师模型之间的知识蒸馏）。
    student_output: 学生模型的输出
    teacher_output: 教师模型的输出
    alpha: 学生模型和教师模型损失的权重
    T: 温度参数
    """
    student_probs = F.log_softmax(student_output / T, dim=-1)
    teacher_probs = F.softmax(teacher_output / T, dim=-1)
    loss_kl = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (T ** 2)
    return loss_kl


# Custom dataset class to load data
class myDatasets(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.topk = 40
        self.num_rbf = 16
        self.map = 14

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])

        # Load features using esmAF_feature and cal_edges
        esm_fs, af_fs = esmAF_feature(sequence_name)
        esm_fs = torch.from_numpy(esm_fs).float()
        edge_index, CA_coords = cal_edges(sequence_name)

        return sequence_name, sequence, label, esm_fs, af_fs, edge_index, CA_coords

    def __len__(self):
        return len(self.labels)


# Function to train KD_EGNN
def train_KD_EGNN(model, fusion_model,train_loader, optimizer, device):
    model.train()  # Set student model to training mode
    fusion_model.train()
    total_loss = 0
    train_pred = []
    train_true = []
    # print(train_loader)
    # for batch_idx, (sequence_name, sequence, label, esm_features, af_features, edge_index, CA_coords) in enumerate(
    #         train_loader):
    for train_data in train_loader:
        sequence_name, sequence, label, esm_fs, af_fs, edge_index, CA_coords = train_data

        antigen_id, chain = sequence_name[0].split('_')

        # esm_features = esm_features.to(device)
        esm_fs = esm_fs.to(device)
        label = label.to(device)
        edge_index = edge_index.to(device)
        CA_coords = CA_coords.to(device)
        edge_index = edge_index.squeeze()
        esm_fs = esm_fs.squeeze()
        CA_coords = CA_coords.squeeze()
        label = label.squeeze()
        #拼接vit
        # 获取L的大小
        L = esm_fs.size(0)
        # 从npy文件加载feature2，假设文件名为'feature2.npy'
        vit_feature = np.load(
            '/mnt/Data6/23gsy/graph-piston/muti_piston_vector/cls_token_16/' + antigen_id.lower() + '_' + chain + '_vit.npy')
        # 将feature2从numpy数组转换为PyTorch张量
        feature2_tensor = torch.tensor(vit_feature, dtype=torch.float32).to(device)  # shape (1, 16)
        # 扩展feature2的维度，使其与feature1在L维度上对齐
        fused_feature = fusion_model(esm_fs,feature2_tensor)
        # 拼接两个特征向量
        #esm_vit_fs = torch.cat((esm_fs, feature2_expanded), dim=1)  # shape (L, 1296)

        #print(sequence_name)
        # print(sequence)
        # print(label, label.shape)
        # print(esm_fs, esm_fs.shape)
        # print(abc)
        # # print(af_fs,af_fs.shape)
        # print(edge_index, edge_index.shape)
        # print(CA_coords, CA_coords.shape)
        outputs, features = model(fused_feature, CA_coords, edge_index)

        # Forward pass through the teacher model (only for distillation, not used in final predictions)
        # with torch.no_grad():
        # teacher_outputs, teacher_features = teacher_model(esm_fs, CA_coords, edge_index)

        # Compute distillation loss between the student and teacher outputs
        distillation_loss_value = distillation_loss(outputs, features, label, T=model.temperature, alpha=0.7, beta=0.03,
                                                    N=3)

        # Compute any additional loss (e.g., cross-entropy loss for classification)
        # Assuming labels are for node classification (e.g., binary cross-entropy for each node)
        loss = distillation_loss_value  # You can add additional losses if needed
        output = sum(outputs[:-1]) / len(outputs[:-1])
        softmax = torch.nn.Softmax(dim=1)
        y_pred = softmax(output)
        y_pred = y_pred.cpu().detach().numpy()
        y_true = label.cpu().detach().numpy()
        train_pred += [pred[1] for pred in y_pred]
        train_true += list(y_true)
        #pred_dict[sequence_name[0]] = [pred[1] for pred in y_pred]

        total_loss += loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # torch.cuda.empty_cache()
    return total_loss / len(train_loader),train_true,train_pred


# Function to evaluate KD_EGNN
def evaluate(model,fusion_model,optimizer, data_loader,device):
    model.train()
    fusion_model.train()
    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        sequence_name, sequence, label, esm_fs, af_fs, edge_index, CA_coords = data
        # print('验证：\n')
        # edge_index = edge_index.squeeze(0)
        # esm_fs = esm_fs.squeeze(0)
        # CA_coords = CA_coords.squeeze(0)
        # label = label.squeeze()
        y_true, esm_fs, edge_index, CA_coords = label.to(device), esm_fs.to(device), edge_index.to(device), CA_coords.to(device)
        y_true, esm_fs, edge_index, CA_coords = torch.squeeze(y_true).long(), torch.squeeze(
                esm_fs), torch.squeeze(edge_index), torch.squeeze(CA_coords)
        # 拼接vit
        # 获取L的大小
        antigen_id, chain = sequence_name[0].split('_')
        L = esm_fs.size(0)
        # 从npy文件加载feature2，假设文件名为'feature2.npy'
        vit_feature = np.load(
            '/mnt/Data6/23gsy/graph-piston/muti_piston_vector/cls_token_16/' + antigen_id.lower() + '_' + chain + '_vit.npy')
        # 将feature2从numpy数组转换为PyTorch张量
        feature2_tensor = torch.tensor(vit_feature, dtype=torch.float32).to(device)  # shape (1, 16)
        # 扩展feature2的维度，使其与feature1在L维度上对齐
        fused_feature = fusion_model(esm_fs,feature2_tensor)
        # 拼接两个特征向量
        #esm_vit_fs = torch.cat((esm_fs, feature2_expanded), dim=1)  # shape (L, 1296)
        outputs, outputs_feature = model(fused_feature, CA_coords, edge_index)
        # print([o.shape for o in outputs])  # 如果 outputs 是张量列表
        # print(sum(outputs[:-1]))
        # print(len(outputs[:-1]))
        output = sum(outputs[:-1]) / len(outputs[:-1])
        # output=outputs[0]
        # loss = criterion(output, y_true)
        # criterion = nn.CrossEntropyLoss()
        # loss = criterion(output, y_true)
        distillation_loss_value = distillation_loss(outputs, outputs_feature, y_true, T=model.temperature, alpha=0.7, beta=0.03,
                                                    N=3)
        loss=distillation_loss_value
        softmax = torch.nn.Softmax(dim=1)
        y_pred = softmax(output)
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        valid_pred += [pred[1] for pred in y_pred]
        valid_true += list(y_true)
        pred_dict[sequence_name[0]] = [pred[1] for pred in y_pred]

        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n += 1
    epoch_loss_avg = epoch_loss / n

    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def test(model, fusion_model,data_loader,device):
    model.eval()
    fusion_model.eval()
    # epoch_loss = 0.0
    # n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            sequence_name, sequence, label, esm_fs, af_fs, edge_index, CA_coords = data

            y_true, esm_fs, edge_index, CA_coords = label.to(device), esm_fs.to(device), edge_index.to(device), CA_coords.to(device)
            y_true, esm_fs, edge_index, CA_coords = torch.squeeze(y_true).long(), torch.squeeze(
                esm_fs), torch.squeeze(edge_index), torch.squeeze(CA_coords)
            # 拼接vit
            # 获取L的大小
            antigen_id, chain = sequence_name[0].split('_')
            L = esm_fs.size(0)
            # 从npy文件加载feature2，假设文件名为'feature2.npy'
            vit_feature = np.load(
                '/mnt/Data6/23gsy/graph-piston/muti_piston_vector/cls_token_16/' + antigen_id.lower() + '_' + chain + '_vit.npy')
            # 将feature2从numpy数组转换为PyTorch张量
            feature2_tensor = torch.tensor(vit_feature, dtype=torch.float32).to(device)  # shape (1, 16)
            # 扩展feature2的维度，使其与feature1在L维度上对齐
            fused_feature = fusion_model(esm_fs,feature2_tensor)
            # 拼接两个特征向量
            #esm_vit_fs = torch.cat((esm_fs, feature2_expanded), dim=1)  # shape (L, 1296)
            outputs, outputs_feature = model(fused_feature, CA_coords, edge_index)
            #outputs, outputs_feature = model(esm_fs, CA_coords, edge_index)

            output = sum(outputs[:-1]) / len(outputs[:-1])
            # output=outputs[0]
            # loss = criterion(output, y_true)
            # loss=F.cross_entropy(output,y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(output)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_name[0]] = [pred[1] for pred in y_pred]

            # epoch_loss += loss.item()
            # n += 1
    # epoch_loss_avg = epoch_loss / n

    return valid_true, valid_pred, pred_dict

def set_random_seed(seed: int = 42):
    """设置随机种子，以确保结果可复现"""
    random.seed(seed)  # Python的随机数生成器
    np.random.seed(seed)  # numpy的随机数生成器
    torch.manual_seed(seed)  # PyTorch的随机数生成器
    torch.cuda.manual_seed_all(seed)  # 如果使用GPU的话，固定所有设备上的随机种子
    # 如果你有确定性的计算需求
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# Main script
# Cross-validation loop
if __name__ == '__main__':
    set_random_seed(3407)  # 固定随机种子，确保可复现
    # Load your dataset
    start_time = time.time()  # 记录开始时间
    print("start_time ", start_time)
    train_file = './sema_train_wash.pkl'  # Replace with your actual data file
    test_file = './sema_test_final.pkl'
    with open(train_file, "rb") as f:
        train_data = pickle.load(f)
    with open(test_file, "rb") as f:
        test_data = pickle.load(f)
    # Define model, optimizer, and loss function

    #criterion = nn.BCEWithLogitsLoss()
    fold_result = []
    saved_models = []
    saved_fusion_models = []  # 保存每一折的best fusion model权重
    # Prepare the dataset
    train_IDs, train_sequences, train_labels = [], [], []
    for ID in train_data:
        train_IDs.append(ID)
        item = train_data[ID]
        train_sequences.append(item[0])
        train_labels.append(item[1])
    train_dic = {"ID": train_IDs, "sequence": train_sequences, "label": train_labels}
    train_dataframe = pd.DataFrame(train_dic)
    test_IDs, test_sequences, test_labels = [], [], []
    for ID in test_data:
        test_IDs.append(ID)
        item = test_data[ID]
        test_sequences.append(item[0])
        test_labels.append(item[1])
    test_dic = {"ID": test_IDs, "sequence": test_sequences, "label": test_labels}
    test_dataframe = pd.DataFrame(test_dic)
    test_dataset = DataLoader(dataset=myDatasets(test_dataframe), batch_size=1, shuffle=True, num_workers=0,
                              drop_last=True)
    # KFold Cross-Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    best_epoch_results = []

    # Early stopping setup
    patience = 5  # Number of epochs to wait for improvement

    # 5-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataframe)):

        print(f"\nFold {fold + 1}")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        model = KD_EGNN(infeature_size=1280, outfeature_size=512, nhidden_eg=128, edge_feature=0, n_eglayer=4, nclass=2,
                        device=device)
        model.to(device)
        fusion_model = FiLMFusion(seq_dim=1280, struct_dim=16, out_dim=1280)
        fusion_model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        # Split data into train and validation sets
        train_fold = train_dataframe.iloc[train_idx]
        val_fold = train_dataframe.iloc[val_idx]

        # Create data loaders
        train_dataset = DataLoader(dataset=myDatasets(train_fold), batch_size=1, shuffle=True, num_workers=0,
                                   drop_last=True)
        val_dataset = DataLoader(dataset=myDatasets(val_fold), batch_size=1, shuffle=True, num_workers=0,
                                 drop_last=True)

        # Initialize the model weights for this fold
        #model.load_state_dict(best_model_wts if best_model_wts is not None else model.state_dict())
        epochs_no_improve = 0  # Counter for epochs without improvement

        best_val_loss = float('inf')
        best_model_wts = None
        best_fusion_model_wts = None
        # Train the model for this fold
        best_fold_val_loss = float('inf')
        for epoch in range(200):
            train_loss,train_true,train_pred = train_KD_EGNN(model, fusion_model, train_dataset, optimizer, device)
            print(f"Epoch [{epoch + 1}/200], Loss: {train_loss:.4f}")
            # Evaluate after each epoch
            val_loss, valid_true, valid_pred, pred_dict = evaluate(model,fusion_model, optimizer,val_dataset,device)
            print(f"Epoch [{epoch + 1}/200], Validation Loss: {val_loss:.4f}")
            scheduler.step()
            # Save the model if it has the lowest validation loss so far
            if val_loss < best_fold_val_loss:
                best_fold_val_loss = val_loss
                best_model_wts = model.state_dict()  # Save the model weights
                best_fusion_model_wts=fusion_model.state_dict()
                # best_epoch_results = analysis(valid_true, valid_pred)
                epochs_no_improve = 0  # Reset counter as we have improvement
            else:
                epochs_no_improve += 1  # Increment counter if no improvement

            # Check if early stopping condition is met
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
                break  # Stop training

            results = analysis(valid_true, valid_pred)

            # results_str = json.dumps(results)  # Convert the dictionary to a JSON string

            with open('/mnt/Data6/23gsy/SEKD-main/seed/mutipiston+film/3g_validation_results.txt', 'a') as result_file:
                result_file.write("'ACC', 'PRE', 'REC', 'F1', 'AUC', 'AUPRC', 'MCC'\n")
                result_file.write(f"{fold + 1}\t{epoch + 1}\t{best_fold_val_loss:.4f}\t{results}\n")
            # fold_result.append(results)

        # Save the best model for this fold
        saved_models.append(best_model_wts)  # Store the model weights of the best model for this fold
        saved_fusion_models.append(best_fusion_model_wts)
        # Evaluate the best model on the test set (use the model saved after each fold)
        # Assuming you have a separate test dataset (or if you are using validation set as the test in CV)
        print("Evaluating on the test set using the best model of this fold...")
        model.load_state_dict(best_model_wts)
        fusion_model.load_state_dict(best_fusion_model_wts)
        test_true, test_pred, test_pred_dict = test(model, fusion_model,test_dataset,device)  # Assuming test_dataset exists
        test_results = analysis(test_true, test_pred)
        print(f"Test results for fold {fold + 1}: {test_results}")
        fold_result.append(test_results)

    end_time = time.time()  # 记录结束时间
    total_duration = end_time - start_time  # 计算总时长

    # 打印开始时间、结束时间和总时长
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total duration: {time.strftime('%H:%M:%S', time.gmtime(total_duration))}")

    # After all folds, calculate the average results
    metrics = ['ACC', 'PRE', 'REC', 'F1', 'AUC', 'AUPRC', 'MCC']
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
    for fold, model_weights in enumerate(saved_models):
        torch.save(model_weights, f"/mnt/Data6/23gsy/SEKD-main/seed/mutipiston+film/3g_second_model_fold_{fold + 1}.pth")
        print(f"Saved the best model for fold {fold + 1}")
    # Save the fusion models of all folds (optional)
    for fold, fusion_model_weights in enumerate(saved_fusion_models):
        torch.save(fusion_model_weights,
                   f"/mnt/Data6/23gsy/SEKD-main/seed/mutipiston+film/3g_second_film_fusion_model_fold_{fold + 1}.pth")
        print(f"Saved the best fusion model for fold {fold + 1}")