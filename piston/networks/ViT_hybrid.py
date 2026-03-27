"""
Objective:
   The PIsToN-Hybrid component.
   The method combines empirically computed energies with the interface maps.
    The $Q$ energy terms are projected on to a latent vector using a fully connected network (FC),
                                        which is then concatenated to the vector obtained from ViT

Author:
    Vitalii Stebliankin (vsteb002@fiu.edu)
    Bioinformatics Research Group (BioRG)
    Florida International University

"""

import torch
from torch import nn
from .ViT_pytorch import Transformer

class ViT_Hybrid(nn.Module):
    def __init__(self, config, n_individual, img_size=24, num_classes=2, zero_head=False, vis=False, channels=13):
        super(ViT_Hybrid, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, channels, vis)
        self.individual_nn = nn.Linear(n_individual, n_individual)

        self.combine_nn = nn.Linear(config.hidden_size + n_individual, config.hidden_size)

        self.classifier_nn = nn.Linear(config.hidden_size, num_classes)

        self.af_ind = nn.GELU()
        self.af_combine = nn.GELU()


    def forward(self, x, individual_feat):
        x, attn_weights = self.transformer(x)
        x = x[:, 0] # classification token

        individual_x = self.individual_nn(individual_feat)
        individual_x = self.af_ind(individual_x)

        x = torch.cat([x, individual_x], dim=1)

        x = self.combine_nn(x)
        x = self.af_combine(x)

        logits = self.classifier_nn(x)

        return logits, attn_weights

class ViT_Hybrid_encoder(nn.Module):
    def __init__(self, config, n_individual, img_size=24, num_classes=2, zero_head=False, vis=False, channels=13):
        super(ViT_Hybrid_encoder, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, channels, vis)
        self.individual_nn = nn.Linear(n_individual, n_individual)

        self.combine_nn = nn.Linear(config.hidden_size + n_individual, config.hidden_size)

        self.af_ind = nn.GELU()
        self.af_combine = nn.GELU()

    def forward(self, x, individual_feat):
        x, attn_weights = self.transformer(x)
        x = x[:, 0]  # classification token

        individual_x = self.individual_nn(individual_feat)
        individual_x = self.af_ind(individual_x)

        x = torch.cat([x, individual_x], dim=1)

        x = self.combine_nn(x)
        x = self.af_combine(x)

        # logits = self.head(x[:, 0])

        return x, attn_weights
    # def forward(self, x, individual_feat):
    #     x, attn_weights = self.transformer(x)
    #     #print('VIT_hybrid_forward',x.shape)
    #     #后续是只对cls token层进行操作了
    #     # # print('atten_weights',len(attn_weights))
    #     # print('atten_weights', attn_weights[0].shape)
    #     # x = x[:, 0] # classification token，只使用分类标记
    #     # print('x = x[:, 0]',x.shape)
    #     # individual_x = self.individual_nn(individual_feat)
    #     # individual_x = self.af_ind(individual_x)#通过线性层再使用gelu激活
    #     #
    #     # x = torch.cat([x, individual_x], dim=1)
    #     # print('VIT_hybrid x = torch.cat([x, individual_x], dim=1)',x.shape)
    #     # x = self.combine_nn(x)
    #     # print('x = self.combine_nn(x)',x.shape)
    #     # x = self.af_combine(x)
    #     # print('x = self.af_combine(x)',x.shape)
    #     # #logits = self.head(x[:, 0])
    #
    #     return x, attn_weights

