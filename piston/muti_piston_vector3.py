import torch
import numpy as np
from networks.PIsToN_multiAttn import PIsToN_multiAttn
from networks.ViT_pytorch import get_ml_config
import os
import numpy as np
from torch.utils.data import Dataset
import random
import os

import torch
from Bio import PDB
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly
#
# from collections import defaultdict
import torch.nn.functional as F
# import torchvision.transforms as transforms
from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import ndimage
def read_energies(energies_dir, ppi):
    """
        (0) - indx
        (1) - Lrmsd     - ligand rmsd of the final position, after the rigid-body optimization.
        (2) -Irmsd     - interface rmsd of the final position, after the rigid-body optimization.
        (3) - st_Lrmsd  - initial ligand rmsd.
        (4) - st_Irmsd  - initial ligand rmsd.
    0 - (5) - glob      - global score of the candidate, which is linear combination of the terms described bellow. To rank the candidates, you should sort the rows by this column in ascending order.
    1 - (6) - aVdW      - attractive van der Waals
    2 - (7) - rVdW      - repulsive van der Waals
    3 - (8) - ACE       - Atomic Contact Energy | desolvation (10.1006/jmbi.1996.0859)
    4 - (9) - inside    - "Insideness" measure, which reflects the concavity of the interface.
    5 - (10) - aElec     - short-range attractive electrostatic term
    6 - (11) - rElec     - short-range repulsive electrostatic term
    7 - (12) - laElec    - long-range attractive electrostatic term
    8 - (13) - lrElec    - long-range repulsive electrostatic term
    9 - (14) - hb        - hydrogen and disulfide bonding
    10 - (15) - piS	     - pi-stacking interactions
    11 - (16) - catpiS	  - cation-pi interactions
    12 - (17) - aliph	  - aliphatic interactions
         (18) - prob      - rotamer probability
    :param ppi:
    :return:
    """
    energies_path = f"/mnt/Data6/23gsy/graph-piston/sema_refined/sema_train//refined-out-{ppi}.ref"

    to_read = False
    all_energies = None

    with open(energies_path, 'r') as f:
        for line in f.readlines():
            if to_read:
                all_energies = line.split('|')
                all_energies = [x.strip(' ') for x in all_energies]
                all_energies = all_energies[5:18]
                all_energies = [float(x) for x in all_energies]
                all_energies = np.array(all_energies)
                break
            if 'Sol # |' in line:
                to_read = True
    if all_energies is None:
        # energies couldn't be computed. Assign them to zero
        all_energies = np.zeros(13)

    all_energies = np.nan_to_num(all_energies)
    return all_energies

def learn_background_mask(grid):
    """
    Returns the mask with zero elements outside the patch
    :param grid: example of a grid image
    :return: mask
    """
    mask = np.zeros((grid.shape[0], grid.shape[1]))
    radius = grid.shape[0] / 2
    for row_i in range(grid.shape[0]):
        for column_i in range(grid.shape[1]):
            # Check if coordinates are within the radius
            x = column_i - radius
            y = radius - row_i
            if x ** 2 + y ** 2 <= radius ** 2:
                mask[row_i][column_i] = 1
    return mask
class PISToN_dataset(Dataset):
    def __init__(self, grid_dir, ppi_list, attn=None):

        ### Empirically learned mean and standard deviations:
        mean_array = [0.06383528408485302, 0.043833505848899605, -0.08456032982438057, 0.007828608135306595,
                      -0.06060602411612203, 0.06383528408485302, 0.043833505848899605, -0.08456032982438057,
                      0.007828608135306595, -0.06060602411612203, 11.390402735801011, 0.1496338245579665,
                      0.1496338245579665]
        std_array = [0.4507792893174703, 0.14148081793902434, 0.16581325050002976, 0.28599861830017204,
                     0.6102229371168204, 0.4507792893174703, 0.14148081793902434, 0.16581325050002976,
                     0.28599861830017204, 0.6102229371168204, 7.265311558033949, 0.18003612950610695,
                     0.18003612950610695]

        all_energies_mean = [-193.1392953586498, -101.97838818565408, 264.2099535864983, -17.27086075949363,
                             16.329959915611877, -102.78101054852341, 36.531006329113836, -27.1124789029536,
                             16.632626582278455, -8.784924050632918, -6.206329113924051, -1.8290084388185655,
                             -11.827215189873417]
        all_energies_std = [309.23521244706757, 66.75799437657368, 9792.783784373369, 25.384427268309658,
                            7.929941961525389, 94.05055841984323, 47.22518557457095, 24.392679889433445,
                            17.57399925906454, 7.041949880295568, 6.99554122803362, 2.557571754303165,
                            13.666329541281653]

        all_grids = []
        all_energies = []

        ppi_to_idx = {} # map ppi id to idx

        i=0
        for ppi in ppi_list:
            if os.path.exists(f"/mnt/Data6/23gsy/graph-piston/32sema_resized/sema_train/grid/{ppi}.npy"):
                ppi_to_idx[ppi] = i
                grid = np.load(f"/mnt/Data6/23gsy/graph-piston/32sema_resized/sema_train/grid/{ppi}.npy", allow_pickle=True)
                all_grids.append(grid)
                #energies_path = f"{grid_dir}/refined-out-{ppi}.ref"
                energies = read_energies(grid_dir, ppi)
                all_energies.append(energies)
                i+=1
        self.ppi_to_idx = ppi_to_idx
        #background_mask = learn_background_mask(grid)

        grid = np.stack(all_grids, axis=0)
        grid = np.swapaxes(grid, -1, 1).astype(np.float32)
        all_energies = np.stack(all_energies, axis=0)

        print(f"Interaction maps shape: {grid.shape}")
        print(f"All energies shape: {all_energies.shape}")

        ### Standard scaling
        # Interactino maps:
        for feature_i in range(grid.shape[1]):
            grid[:, feature_i, :, :] = (grid[:, feature_i, :, :] - mean_array[feature_i]) / std_array[feature_i]
            # Mask out values that are out of the radius:
            #grid = np.logical_and(grid, background_mask) * grid

        ## ENERGIES:
        for energy_i in range(all_energies.shape[1]):
            all_energies[:, energy_i] = (all_energies[:, energy_i] - all_energies_mean[energy_i]) / all_energies_std[
                energy_i]

        self.grid = grid
        self.all_energies = all_energies
        self.grid_dir = grid_dir
        self.ppi_list = ppi_list

    def __len__(self):
        return self.grid.shape[0]

    def read_scaled(self, ppi, device):
        idx = self.ppi_to_idx[ppi]
        grid = torch.from_numpy(np.expand_dims(self.grid[idx], 0))
        energies = torch.from_numpy(np.expand_dims(self.all_energies[idx], 0))
        ppi_id = self.ppi_list[idx]  # 获取对应的 PPI ID
        #print(ppi_id)
        return grid.to(device), energies.float().to(device),ppi_id


    def __getitem__(self, idx):
        ppi_id = self.ppi_list[idx]  # 获取对应的 PPI ID
        print(ppi_id)
        return self.grid[idx], self.all_energies[idx],ppi_id



MODEL_DIR='./saved_models/'
MODEL_NAME='PIsToN_multiAttn_contrast'

## Paremeters of the original training:
params = {'dim_head': 16,
          'hidden_size': 16,
          'dropout': 0,
          'attn_dropout': 0,
          'n_heads': 8,
          'patch_size': 4,
          'transformer_depth': 8}

#device=torch.device("cuda")
device=torch.device("cpu")

model_config=get_ml_config(params)
model = PIsToN_multiAttn(model_config, img_size=32,
                        num_classes=2).float().to(device)

model.load_state_dict(torch.load(MODEL_DIR + '/{}.pth'.format(MODEL_NAME), map_location=device))
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
n_params = sum([np.prod(p.size()) for p in model_parameters])

print(f"Loaded PiSToN model with {n_params} trainable parameters.")
GRID_DIR = '/mnt/Data6/23gsy/graph-piston/32sema_resized/sema_train/grid/'
ppi_list = os.listdir(GRID_DIR)

ppi_list = [x.split('.npy')[0] for x in ppi_list if 'resnames' not in x and '.ref' not in x]

#labels = [0 if 'neg' in x else 1 for x in ppi_list]


print(f"Extracted {len(ppi_list)} complexes.")
#print(f"{np.sum(labels)} acceptable and {len(labels) -np.sum(labels) } incorrect.")
masif_test_dataset = PISToN_dataset(GRID_DIR, ppi_list)
print('masif test dataset',masif_test_dataset)

start = time()
torch.set_num_threads(1)
device = torch.device("cpu")

masif_test_loader = DataLoader(masif_test_dataset, batch_size=1, shuffle=False, pin_memory=False)

all_outputs = []  # output score
all_attn = []  # output attention map
predicted_labels = []  # predicted label (0 for negative and 1 for positive)

with torch.no_grad():
    for instance in tqdm(masif_test_loader):
        grid, all_energies,ppi_id = instance
        # print('grid',grid.shape)
        # print('all_energies',all_energies.shape)
        print('ppi_id',ppi_id)
        #print(grid)


        ppi_id_str = ppi_id[0]  # 取出元组中的字符串部分
        antigen = ppi_id_str.split(',')[0]  # 用逗号分割并取第一个部分


        grid = grid.to(device)
        all_energies = all_energies.float().to(device)
        model = model.to(device)
        output, attn = model(grid, all_energies)
        


        # antibody_vector = antibody_vector.cpu().detach().numpy()
        # predict = predict.cpu().detach().numpy()
        np.save(f"/mnt/Data6/23gsy/graph-piston/muti_piston_vector/cls_token_16/{antigen}_vit.npy", output)
        # print(antigen)
        # print(new_features.shape)
        # print(abc)
        #print('attn',attn.shape)
        # 获取并打印中间层输出
        #print(stack_output)  # 打印捕获的 x = torch.stack(all_x, dim=1)
        #print(feature_transformer_output.shape)  # 打印捕获的 x
        #print(feature_attn_output.shape)  # 打印捕获的 feature_attn
        #print('打断',abc)

# 移除钩子
# output = torch.cat(all_outputs, axis=0)
print(f"Total inference time: {time() - start} sec")

# output