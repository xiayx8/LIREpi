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
from predict_MY_data import esmAF_feature
from predict_MY_data import cal_edges
from tools import analysis
from sklearn.model_selection import KFold
import warnings
import random
warnings.filterwarnings("ignore")


# Assuming the KD_EGNN class is defined as shown in your earlier messages
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
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.topk = 40
        self.num_rbf = 16
        self.map = 14

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]

        esm_fs, af_fs = esmAF_feature(sequence_name)
        esm_fs = torch.from_numpy(esm_fs).float()
        edge_index, CA_coords = cal_edges(sequence_name)

        return sequence_name, sequence, esm_fs, af_fs, edge_index, CA_coords

    def __len__(self):
        return len(self.names)

def inference(model, data_loader, device):
    model.eval()
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            sequence_name, sequence, esm_fs, af_fs, edge_index, CA_coords = data
            esm_fs, edge_index, CA_coords = esm_fs.to(device), edge_index.to(device), CA_coords.to(device)
            esm_fs, edge_index, CA_coords = esm_fs.squeeze(), edge_index.squeeze(), CA_coords.squeeze()

            antigen_id, chain = sequence_name[0].split('_')
            outputs, _ = model(esm_fs, CA_coords, edge_index)

            output = sum(outputs[:-1]) / len(outputs[:-1])  # shape: [L, 2]
            probs = torch.softmax(output, dim=-1)  # shape: [L, 2]
            residue_probs = probs[:, 1].cpu().detach().numpy().tolist()  # 取表位概率

            pred_dict[sequence_name[0]] = residue_probs

    return pred_dict


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
    set_random_seed(42)  # 固定随机种子，确保可复现
    # Load your dataset
    test_file = '/mnt/Data6/xyx/piston-main1/spike/output.pkl'
    with open(test_file, "rb") as f:
        test_data = pickle.load(f)
    # Define model, optimizer, and loss function

    #criterion = nn.BCEWithLogitsLoss()

    test_IDs, test_sequences = [], []
    for ID in test_data:
        test_IDs.append(ID)
        item = test_data[ID]
        test_sequences.append(item[0])
        #test_labels.append(item[1])
    test_dic = {"ID": test_IDs, "sequence": test_sequences}
    test_dataframe = pd.DataFrame(test_dic)
    inference_dataset = DataLoader(dataset=InferenceDataset(test_dataframe), batch_size=1, shuffle=False, num_workers=0)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = KD_EGNN(infeature_size=1280, outfeature_size=512, nhidden_eg=128, edge_feature=0, n_eglayer=4,
                            nclass=2,
                            device=device)
    model.to(device)
    # 加载训练好的模型参数（替换路径）
    model.load_state_dict(
        torch.load("/mnt/Data6/23gsy/SEKD-main/seed/mutipiston+film/3g_second_model_fold_5.pth", map_location='cpu'))

    pred_residue_probs = inference(model, inference_dataset, device)
    with open("/mnt/Data6/xyx/piston-main1/spike/predict_out111.txt", "w") as f:
        for seq_id, probs in pred_residue_probs.items():
            prob_str = '\t'.join(f"{p:.4f}" for p in probs)
            f.write(f"{seq_id}\t{prob_str}\n")

    #