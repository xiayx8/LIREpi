#import pymesh #Importing pymesh here avoids library conflict (CXXABI_1.3.11)
from tqdm import tqdm
import numpy as np
import pdb
from sklearn.neighbors import KDTree
import os
from scipy import ndimage
from torch import nn
from networks.ViT_pytorch import Encoder
from networks.ViT_hybrid import ViT_Hybrid_encoder
import torch.nn.functional as F
from Bio.PDB import PDBParser, DSSP

from sklearn.decomposition import PCA
from data_prepare.map_patch_atom import map_patch_indices
import torch



def load_pdb_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()


def parse_pdb(pdb_data, chain_id):
    residue_dict = {}
    index = 0
    for line in pdb_data:
        if line.startswith("ATOM"):
            line_chain_id = line[21]  # 链ID在22列
            # residue_seq = line[22:26].strip()+line[26].strip()  # 残基序号在23-26列
            residue_seq = line[22:26].strip()  # 残基序号在23-26列
            if line_chain_id == chain_id:
                if residue_seq not in residue_dict:
                    residue_dict[residue_seq] = index
                    index += 1

    return residue_dict


def get_res_num_dict(protein_id, chain_id,pdb_file):
    pdb_data = load_pdb_file(pdb_file)
    residue_dict = parse_pdb(pdb_data, chain_id)
    return residue_dict
def parse_residue_info(atom_info):
    # 格式解析：链id:残基序号:残基名称-虚拟点序号:原子名称
    parts = atom_info.split(':')
    residue_id = parts[1]  # 获取残基序号

    return residue_id


def extract_esm_values(name_grid, esm_matrix,residue_index_dict):
    last_key = list(residue_index_dict.keys())[-1]
    last_value = residue_index_dict[last_key]
    last_value+=last_value
    # 初始化一个75x75x20的矩阵
    grid_size = name_grid.shape
    esm_grid = np.zeros((grid_size[0], grid_size[1], 480), dtype=np.float64)
    # 打印PSSM矩阵的大小
    print(f"esm matrix shape: {esm_matrix.shape}")
    # min_residue_number=min_residue_number+1
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            block = name_grid[i, j]
            #print("block",block)
            # if block is None:
            #     esm_grid[i, j] = [None] * 1280
            #     continue
            if block is None or len(block) == 0:
                esm_grid[i, j] = np.zeros((480,), dtype=np.float64)  # 用全零向量填充
                continue
            residues = set()  # 用于存储块中所有独特的残基序号

            for atom_info in block:
                residue_id = parse_residue_info(atom_info)
                residues.add(residue_id)
            #print(f"Block {i}, {j} residues: {residues}")  # 调试信息
            # 获取所有残基的PSSM值
            esm_values = []

            for residue_id in residues:
                row_id=residue_index_dict[residue_id]
                esm_column_values = esm_matrix[row_id]
                esm_values.append(esm_column_values)
                # pssm_column_values = []
                # for residue_id in residues:
                #     if isinstance(residue_id, int) and 0 <= residue_id - 1 < pssm_matrix.shape[1]:
                #         pssm_column_values.append(pssm_matrix[k, residue_id - 1])
                #     else:
                #         print(f"Invalid residue_id: {residue_id}")
                # pssm_values.append(pssm_column_values)
            esm_values = np.array(esm_values)
            if len(residues) > 0:
                esm_avg = np.mean(esm_values, axis=0)
            esm_grid[i, j] = esm_avg

    return esm_grid

def get_min_residue_number(pdb_file, chain):
    min_residue_number = None
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM") and line[21] == chain:
                residue_number = int(line[22:26].strip())
                if min_residue_number is None or residue_number < min_residue_number:
                    min_residue_number = residue_number
    return min_residue_number

def create_residue_dict(pdb_file, chain_id):
    residue_dict = {}
    index = 0
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                if line[21] == chain_id:
                    residue_number = int(line[22:26].strip())
                    if residue_number not in residue_dict:
                        residue_dict[residue_number] = index
                        index += 1
    return residue_dict
def extract_cdr_coordinates(pdb_file, antibody_chain):
    atom_coordinates = []
    cdr_regions = {
        'H': {
            'H1': (26, 32),
            'H2': (52, 56),
            'H3': (95, 102),
        },
        'L': {
            'L1': (24, 34),
            'L2': (50, 56),
            'L3': (89, 97)
        }
    }
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chain_name = line[21]
                if chain_name == antibody_chain:
                    residue_number = int(line[22:26])
                    for cdr_name, cdr_range in cdr_regions["H"].items():
                        cdr_start, cdr_end = cdr_range
                        if cdr_start <= residue_number <= cdr_end:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            atom_coordinates.append((x, y, z))
                            break  # Break once the atom is found in any CDR region
    return atom_coordinates

def extract_coordinates_from_structure(structure_dict, atom_info_array):
    atom_coordinates = []

    for atom_info in atom_info_array:
        atom_info = atom_info[0]
        parts = atom_info.split(':')
        chain = parts[0]
        resid = int(parts[1])
        res_name = parts[2].split("-")[0]
        atom_name = parts[-1]

        if chain in structure_dict and (resid, res_name) in structure_dict[chain] and atom_name in structure_dict[chain][(resid, res_name)]:
            coordinates = structure_dict[chain][(resid, res_name)][atom_name]
            atom_coordinates.append(coordinates)

    return atom_coordinates

def extract_filtered_coordinates(pdb_file, atom_info_array, cdr_coordinates):
    structure_dict = build_structure_dict(pdb_file)
    coordinates = extract_coordinates_from_structure(structure_dict, atom_info_array)

    filter_index = []
    for i, sub_list in enumerate(coordinates):
        if sub_list in cdr_coordinates:
            filter_index.append(i)

    return filter_index

def build_structure_dict(pdb_file):
    structure_dict = {}

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21]  # 获取链名
                resid = int(line[22:26])  # 获取残基ID
                res_name = line[17:20].strip()  # 获取残基名
                atom_name = line[12:16].strip()  # 获取原子名
                x = float(line[30:38])  # 获取x坐标
                y = float(line[38:46])  # 获取y坐标
                z = float(line[46:54])  # 获取z坐标

                # 添加到结构字典中
                if chain not in structure_dict:
                    structure_dict[chain] = {}
                residue_key = (resid, res_name)
                if residue_key not in structure_dict[chain]:
                    structure_dict[chain][residue_key] = {}
                structure_dict[chain][residue_key][atom_name] = (x, y, z)

    return structure_dict

def polar_to_cartesian(rho, theta, rotate_theta=0):
    # Interpolate the polar coordinates into a d x d square, where d is the diameter of the patch
    # rotate_theta - rotate all coordinates on a constant angle (used to search for matching patches).
    #cart_grid = np.zeros((rho.shape[0],radius*2,radius*2))
    cart_coord_x = np.zeros(rho.shape)
    cart_coord_y = np.zeros(rho.shape)

    for coord_i in range(0, rho.shape[0]):
        rho_coord = rho[coord_i]
        theta_coord = theta[coord_i]
        cart_coord_x[coord_i] = rho_coord*np.cos(theta_coord+rotate_theta)
        cart_coord_y[coord_i] = rho_coord*np.sin(theta_coord+rotate_theta)

    return cart_coord_x, cart_coord_y

def get_new_coord_patch(radius):
    new_patch_coord = []
    for i in range(0, radius*2):
        for j in range(0, radius*2):
            new_patch_coord.append((i-radius,j-radius))
    return np.array(new_patch_coord)


def compute_patch_grid(x, y, input_feat, radius, flag= False, interpolate=True, stringarray=False):
    old_coord = np.stack((x, y), axis=-1)
    #输入的特征是否是字符串数组
    if not stringarray:
        patch_grid = np.zeros((radius * 2, radius * 2, input_feat.shape[1]))  # shape = (24 x 24 x 5)
    else:
        patch_grid = np.full((radius * 2, radius * 2, 12), np.array(['x' for _ in range(12)], dtype=object))
        input_feat = np.array(input_feat,dtype=object)

    for feature_i in range(0, patch_grid.shape[-1]):
        all_indices = []

        old_coord_patch = old_coord
        new_coord_patch = get_new_coord_patch(radius)  # grid coordinates [-r, r]

        # map old coordinates to the new grid:
        kdt = KDTree(old_coord_patch)

        # both shapes are (num_points, 12)
        if interpolate:
            dist, indx_old = kdt.query(new_coord_patch, k=12)  # interpolate across 4 nearest neighbors
        else:
            dist, indx_old = kdt.query(new_coord_patch, k=12)#查询网格中每个新点对应的12个最近邻旧点，返回这些邻居的距离dist和索引indx_old

        # Square the distances (as in the original pyflann)
        dist = np.square(dist)
        #将新坐标 (x_new, y_new) 转换为网格坐标 (row_i, column_i)，注意这里进行了偏移和坐标的转换。对坐标进行平移（即增加 radius）以确保所有的坐标都是非负数，从而可以正确访问补丁网格 patch_grid。
        # go over each new coordinate
        for grid_point_i in range(0, dist.shape[0]):
            x_new, y_new = new_coord_patch[grid_point_i]  # coordinate in a new grid
            r_tmp = np.sqrt(x_new ** 2 + y_new ** 2)  # length of the radius from the center to the new point

            # Because our grid has negative coordinates, we will shift them to have only positive coordinates:
            column_i = x_new + radius  # row index of the final patch grid偏移值，移到正半轴
            row_i = -y_new + radius - 1  # column index of the final patch grid
            #如果距离为 0（表示当前点为最近邻），则直接将邻居的特征赋值给补丁网格。如果是中心点（x_new == 0 and y_new == 0），则使用第一个特征，否则使用最近邻的特征。
            if dist[grid_point_i][0] == 0:  # if the coordinate is for the neighbor that doesn't exist
                neigh_index_i = indx_old[grid_point_i][0]
                if x_new == 0 and y_new == 0:
                    patch_grid[row_i][column_i][feature_i] = input_feat[0][feature_i]  # if center point
                else:
                    patch_grid[row_i][column_i][feature_i] = input_feat[neigh_index_i][feature_i]
                all_indices.append(neigh_index_i)
                continue

            # get distance and index
            dist_grid_point = dist[grid_point_i]
            result_grid_points = indx_old[grid_point_i]  # points to interpolate

            dist_to_include = []
            result_to_include = []  # old index list

            # remove duplicated points, ensuring each old coordinate only maps to a new coordinate,确保每个旧坐标都被映射到了新坐标上
            for i, result_i in enumerate(result_grid_points):
                if result_i not in result_to_include:
                    result_to_include.append(result_i)
                    dist_to_include.append(dist_grid_point[i])
                    all_indices.append(result_i)
            #根据距离的倒数计算权重，计算插值后的特征值，并将插值结果赋值给补丁网格。
            if interpolate:
                total_dist = np.sum(1 / np.array(dist_to_include))
                interpolated_value = 0

                # compute weight using distance
                for i, result_old_i in enumerate(result_to_include):
                    interpolated_value += input_feat[result_old_i][feature_i] * (1 / dist_to_include[i]) / total_dist

                patch_grid[row_i][column_i][feature_i] = interpolated_value
            else:
                # for i ,index in enumerate(result_to_include):
                #     patch_grid[row_i][column_i][i] = input_feat[index]
                for i in range(12):
                    patch_grid[row_i][column_i][:] = input_feat[result_grid_points[:12]].flatten()

        #检查是否全部映射
        unique_indices = np.unique(all_indices)
        if flag == False:
            if len(unique_indices) != input_feat.shape[0]:
                print(len(unique_indices))
                print(input_feat.shape[0])
                m = input_feat.shape[0]
                all_indices_array = np.array(list(range(m)))
                result = list(set(all_indices_array) - set(unique_indices))
                print(result)
                raise ValueError("Cannot map all vertices")
    return patch_grid



def read_patch(pid, ch, config):
    patch_dir = config['dirs']['patches'] + pid + '/'

    rho = np.load(patch_dir + '{}_{}_rho_wrt_center.npy'.format(pid, ch), allow_pickle=True)
    theta = np.load(patch_dir + '{}_{}_theta_wrt_center.npy'.format(pid, ch), allow_pickle=True)
    input_feat = np.load(patch_dir + '{}_{}_input_feat.npy'.format(pid, ch), allow_pickle=True)
    resnames = np.load(patch_dir + '{}_{}_resnames.npy'.format(pid, ch), allow_pickle=True)
    resnames = np.expand_dims(resnames, axis=1)

    # ## read interaction features
    # all_interact_feat = []
    # if 'interact_feat' in config.keys():
    #     for interact_feat in config['interact_feat'].keys():
    #         if config['interact_feat'][interact_feat] == True:
    #             feat =  np.load(patch_dir + '{}_{}.npy'.format(pid, interact_feat), allow_pickle=True)
    #             all_interact_feat.append(feat)
    #             ## read interaction features
    #
    #     all_interact_feat = np.concatenate(all_interact_feat, axis=-1)

    # Read 3D coordinates
    coord_3d = np.load(patch_dir + '{}_{}_coordinates.npy'.format(pid, ch), allow_pickle=True)

    return rho, theta, input_feat, resnames, coord_3d

def remove_comments(pdb_path, pdb_tmp_path):
    """
    Standartize PDB file by adding white spaces and making each line exactly 80 characters
    :param pdb_path: input PDB
    :param pdb_tmp_path: output PDB with fixed format
    :return: None
    """
    with open(pdb_path, 'r') as in_pdb:
        with open(pdb_tmp_path, 'w') as out:
            for line in in_pdb.readlines():
                if "USER" not in line:
                    newline = []
                    for i in range(80):
                        if i<len(line.strip('\n')):
                            newline.append(line[i])
                        else:
                            newline.append(' ')
                    if line[:4]=="ATOM" or line[:6]=="HETATM":
                        newline[77]=newline[13]
                        #pdb.set_trace()
                    out.write(''.join(newline)+'\n')
                    #out.write(line)
    return None


def compute_dssp(ppi, config):
    # Compute DSSP values as described in https://biopython.org/docs/1.75/api/Bio.PDB.DSSP.html
    antigen = ppi.split(',')[0]
    antibody = ppi.split(',')[1]
    antigen_pid = antigen.split('_')[0]
    antigen_ch = antigen.split('_')[1]
    antibody_pid = antibody.split('_')[0]
    antibody_ch = antibody.split('_')[1]

    # raw_pdb_dir = config['dirs']['raw_pdb']
    tmp_dir = config['dirs']['tmp']

    # pdb_path = tmp_dir+'{}_{}_{}.pdb'.format(pid,ch1,ch2)
    # extractPDB(raw_pdb_dir+"pdb{}.ent".format(pid.lower()), pdb_path, chain_ids=ch1+ch2)
    antigen_pdb_path = config['dirs']['protonated_pdb'] + '{}.pdb'.format(antigen_pid)
    antigen_pdb_tmp_path = f"{tmp_dir}/{antigen_pid}.pdb"
    antibody_pdb_path = config['dirs']['protonated_pdb'] + '{}.pdb'.format(antibody_pid)
    antibody_pdb_tmp_path = f"{tmp_dir}/{antibody_pid}.pdb"

    # remove hydrogens
    remove_comments(antigen_pdb_path, antigen_pdb_tmp_path)#规范化pdb文件
    remove_comments(antibody_pdb_path, antibody_pdb_tmp_path)

    parser = PDBParser(QUIET=1)
    antigen_struct = parser.get_structure(antigen_pid, antigen_pdb_tmp_path)
    antibody_struct = parser.get_structure(antibody_pid, antibody_pdb_tmp_path)

    antigen_model = antigen_struct[0]
    antibody_model = antibody_struct[0]
    #读取 PDB 文件中的蛋白质结构信息，推断每个氨基酸的二级结构（例如，α 螺旋、β 片层、转角等）以及其溶剂暴露面积。
    antigen_dssp = DSSP(antigen_model, antigen_pdb_tmp_path, dssp='mkdssp')  # example of a key: ('A', (' ', 1147, ' '))链ID 残基序号
    antibody_dssp = DSSP(antibody_model, antibody_pdb_tmp_path, dssp='mkdssp')  # example of a key: ('A', (' ', 1147, ' '))

    # Remove temporary file
    os.remove(antigen_pdb_tmp_path)
    # os.remove(antibody_pdb_tmp_path)
    return antigen_dssp,antibody_dssp

def convert_dssp_to_feat(dssp, names_grid):
    """
    Convert DSSP object into grid of features

    Hydrogen bonds for each chain will be computed separate,
                    as residue from one side can form bonds with multiple residues from the other side.

    PHI PSI - IUPAC peptide backbone torsion angles
    :param dssp:
    :param names_grid:
    :return: numpy array dssp_features
    0 - Relative ASA;
    1 - NH–>O_1_relidx
    2 -

    """


    dssp_features = np.zeros((names_grid.shape[0], names_grid.shape[1], 1))

    for i in range(names_grid.shape[0]):
        for j in range(names_grid.shape[1]):
            curr_name = names_grid[i][j][0] # example A:107:HIS-1621:CD2
            if curr_name!=0:
                # Read the residue from the array of names of a patch pair
                fields = curr_name.split(':')
                chain, resid = fields[0], fields[1]

                # Construct a key based on the current residue from two proteins
                # key example: ('A', (' ', 219, ' '))
                for key_i in dssp.keys():
                    if key_i[0]==chain and key_i[1][1] == int(resid):
                        dssp_key =key_i

                try:
                    dssp_features_i = dssp[dssp_key]
                except:
                    dssp_features[i][j][0] = 0
                    continue

                # Relative ASA:
                try:
                    dssp_features[i][j][0] = dssp_features_i[3]
                except:
                    dssp_features[i][j][0] = 0


    return dssp_features

def convert_dssp_to_feat_all(dssp, names_array):
    """
    Convert DSSP object into grid of features

    Hydrogen bonds for each chain will be computed separate,
                    as residue from one side can form bonds with multiple residues from the other side.

    PHI PSI - IUPAC peptide backbone torsion angles
    :param dssp:
    :param names_grid:
    :return: numpy array dssp_features
    0 - Relative ASA;
    1 - NH–>O_1_relidx
    2 -

    """
    names_array = np.squeeze(names_array,axis=1)
    dssp_features = np.zeros(len(names_array))

    for i in range(len(names_array)):
        curr_name = names_array[i] # example A:107:HIS-1621:CD2
        if curr_name!=0:
        # Read the residue from the array of names of a patch pair
            fields = curr_name.split(':')
            chain, resid = fields[0], fields[1]

            # Construct a key based on the current residue from two proteins
            # key example: ('A', (' ', 219, ' '))
            for key_i in dssp.keys():
                if key_i[0]==chain and key_i[1][1] == int(resid):
                    dssp_key =key_i

            try:
                dssp_features_i = dssp[dssp_key]
            except:
                dssp_features[i] = 0
                continue

            # Relative ASA:
            try:
                dssp_features[i] = dssp_features_i[3]
            except:
                dssp_features[i] = 0

    dssp_features = np.expand_dims(dssp_features,axis=1)

    return dssp_features


def find_optimal_rotation(p1_rho, p1_theta, p2_rho, p2_theta, p1_coord_3d, p2_coord_3d, radius):
    # return (p1target_x, p1target_y), (p2_x, p2_y), where (p2_x, p2_y)

    optimal_angle = 0
    optimal_distance = np.inf
    angle_step = 6.28/100 # make 100 rotations
    curr_angle=0
    p1target_x, p1target_y = polar_to_cartesian(p1_rho, p1_theta)
    p1_coord_grid = compute_patch_grid(p1target_x, p1target_y, p1_coord_3d, radius)

    while curr_angle<6.28:
        p2_x, p2_y = polar_to_cartesian(p2_rho, p2_theta, curr_angle) # rotate only p2
        p2_coord_grid = compute_patch_grid(p2_x, p2_y, p2_coord_3d, radius)
        dist_grid = np.sqrt(np.sum(np.square(p1_coord_grid - p2_coord_grid), axis=-1))
        avg_dist = dist_grid.mean()
        if avg_dist < optimal_distance:
            optimal_angle=curr_angle
            optimal_distance = avg_dist

        curr_angle+=angle_step
    print(f"Optimal angle: {optimal_angle} radians.")
    # print(f"Average distance: {optimal_distance}")
    p2_x, p2_y = polar_to_cartesian(p2_rho, p2_theta, optimal_angle)  # rotate only p2
    return (p1target_x, p1target_y), (p2_x, p2_y)

def convert_antibody_patch(ppi, config,flag = False):
    antibody = ppi.split(',')[1]
    antibody_pid = antibody.split('_')[0]
    antibody_ch = antibody.split('_')[1]

    antibody_out_grid= config['dirs']['grid'] + '{}_{}.npy'.format(antibody_pid, antibody_ch)
    antibody_out_resnames = config['dirs']['grid'] + '{}_{}_resnames.npy'.format(antibody_pid, antibody_ch)
    radius = config['ppi_const']['patch_r']
    antibody_esm_feature=config['dirs']['esm']+'{}_esm2.npy'.format(antibody)
    p2_rho, p2_theta, p2_input_feat, p2_resnames, p2_coord_3d = read_patch(antibody_pid, antibody_ch, config)
    antibody_esm_three_file=config['dirs']['esm'] + '{}_esmpca3.npy'.format(antibody)
    pdb_file = config['dirs']['chains_pdb'] + antibody + ".pdb"
    antibody_min_residue_number = get_min_residue_number(pdb_file, antibody_ch)
    antibody_residue_dict = create_residue_dict(pdb_file, antibody_ch)
    antibody_residue_index_dict = get_res_num_dict(antibody_pid, antibody_ch,pdb_file)
    atom_info_array = p2_resnames
    cdr_pdb = config['dirs']['CDR'] + antibody_pid + ".pdb"
    cdr_coordinates = extract_cdr_coordinates(cdr_pdb, antibody_ch)
    filtered_index = extract_filtered_coordinates(pdb_file, atom_info_array, cdr_coordinates)
    p2_rho = p2_rho[filtered_index]
    p2_theta = p2_theta[filtered_index]
    p2_input_feat = p2_input_feat[filtered_index]
    p2_resnames = p2_resnames[filtered_index]
    p2_coord_3d = p2_coord_3d[filtered_index]
    print(p2_input_feat.shape)

    ##### 极坐标转化为直角坐标
    p2_x, p2_y = polar_to_cartesian(p2_rho, p2_theta)

    p2_patch_grid = compute_patch_grid(p2_x, p2_y, p2_input_feat, radius,flag)  # (r, r, n_feat)
    print("compute residues")
    p2name_grid = compute_patch_grid(p2_x, p2_y, p2_resnames, radius,flag, interpolate=False, stringarray=True)
    print("compute coordinates")
    antibody_single_grid = np.concatenate([p2_patch_grid], axis=-1)

    antibody_names_grid = p2name_grid
    print("compute dssp")
    # if config['interact_feat']['dssp']==True:
    antigen_dssp, antibody_dssp = compute_dssp(ppi, config)
    dssp_grid_2 = convert_dssp_to_feat(antibody_dssp, p2name_grid)
    #print("compute esm")
    #antibody_esm_feature_matrix = np.load(antibody_esm_feature)
    #antibody_esm_grid = np.load(antibody_esm_feature)
    #antibody_esm_grid=extract_esm_values(p2name_grid,antibody_esm_feature_matrix,antibody_min_residue_number,antibody_residue_dict,antibody_residue_index_dict)
    #print('antibody esm 大小',antibody_esm_grid.shape)
    # 获取图像的维度
    #image_height, image_width, num_features = antibody_esm_grid.shape
    # print('esm_demension', num_features)
    # antibody_esm_flat = antibody_esm_grid.reshape(-1, num_features)
    #
    # # 使用PCA将1280维降到128维
    # pca = PCA(n_components=128)
    # antibody_esm_reduced = pca.fit_transform(antibody_esm_flat)
    #
    # # 将降维后的数据重新 reshape 成 (image_height, image_width, 128)
    # antibody_esm_reduced_grid = antibody_esm_reduced.reshape(image_height, image_width, 128)
    # print('reduce:', antibody_esm_reduced_grid.shape)
    # dimension 1 +1
    antibody_single_grid = np.concatenate([antibody_single_grid, dssp_grid_2], axis=-1)
    #antibody_one_grid=np.concatenate([antibody_single_grid,antibody_esm_grid],axis=-1)
    print(antibody_single_grid.shape)
    m = antibody_single_grid.shape[0]

    # 将n的值以空格分隔的格式写入到 antigen_size.txt 中，采用追加的方式
    with open("/mnt/Data2/23gsy/sematrain6/antibody_size.txt", "a") as f:
        f.write(f"{m} ")
    np.save(antibody_out_grid, antibody_single_grid)
    np.save(antibody_out_resnames, antibody_names_grid)
    return None

def convert_antigen_patch(ppi, config,flag = False):
    antigen = ppi.split(',')[0]
    antigen_pid = antigen.split('_')[0]
    antigen_ch = antigen.split('_')[1]

    antigen_out_grid = config['dirs']['grid'] + '{}_{}.npy'.format(antigen_pid, antigen_ch)
    antigen_out_resnames = config['dirs']['grid'] + '{}_{}_resnames.npy'.format(antigen_pid, antigen_ch)
    radius = config['ppi_const']['patch_r']
    antigen_esm_feature = config['dirs']['esm'] + '{}_esm2.npy'.format(antigen)
    p1_rho, p1_theta, p1_input_feat, p1_resnames, p1_coord_3d = read_patch(antigen_pid, antigen_ch, config)#加载数据
    pdb_file = config['dirs']['chains_pdb'] + antigen + ".pdb"
    antigen_esm_three_file = config['dirs']['esm'] + '{}_esmpca3.npy'.format(antigen)
    antigen_min_residue_number = get_min_residue_number(pdb_file, antigen_ch)
    antigen_residue_dict = create_residue_dict(pdb_file, antigen_ch)
    #残基和个数索引的参数
    antigen_residue_index_dict=get_res_num_dict(antigen_pid,antigen_ch,pdb_file)
    ##### 极坐标转化为直角坐标
    p1target_x, p1target_y = polar_to_cartesian(p1_rho, p1_theta)

    p1target_patch_grid = compute_patch_grid(p1target_x, p1target_y, p1_input_feat, radius, flag)  # (r, r, n_feat)
    print("compute residues")
    p1name_grid = compute_patch_grid(p1target_x, p1target_y, p1_resnames, radius, flag, interpolate=False,
                                     stringarray=True)
    print("compute coordinates")
    # dimension 5 + 5 + 1
    antigen_single_grid = np.concatenate([p1target_patch_grid], axis=-1)

    antigen_names_grid = p1name_grid
    # print(antigen_names_grid)
    print("compute dssp")
    # if config['interact_feat']['dssp']==True:
    antigen_dssp, antibody_dssp = compute_dssp(ppi, config)
    dssp_grid_1 = convert_dssp_to_feat(antigen_dssp, p1name_grid)
    #print("compute esm2")
    #antigen_esm_feature_matrix = np.load(antigen_esm_feature)
    #antigen_esm_grid=np.load(antigen_esm_feature)
    #print("esm shape",antigen_esm_grid.shape)

    #antigen_esm_grid = extract_esm_values(p1name_grid, antigen_esm_feature_matrix,antigen_min_residue_number,antigen_residue_dict,antigen_residue_index_dict)
    #print('esm特征新网格大小：',antigen_esm_grid.shape)
    # 使用PCA将PSSM特征降维到1维
    # 获取图像的维度
    #image_height, image_width, num_features = antigen_esm_grid.shape
    # print('image_height',image_height)
    # print('image_width',image_width)
    # print('esm_demension',num_features)
    # antigen_esm_flat = antigen_esm_grid.reshape(-1, num_features)
    #
    # # 使用PCA将1280维降到128维
    # pca = PCA(n_components=128)
    # antigen_esm_reduced = pca.fit_transform(antigen_esm_flat)
    #
    # # 将降维后的数据重新 reshape 成 (image_height, image_width, 128)
    # antigen_esm_reduced_grid = antigen_esm_reduced.reshape(image_height, image_width, 128)
    # print('reduce:',antigen_esm_reduced_grid.shape)
    # np.save(antigen_esm_three_file,antigen_esm_grid)
    # dimension 1 +1
    antigen_single_grid = np.concatenate([antigen_single_grid, dssp_grid_1], axis=-1)
    #print('grid_shape',antigen_single_grid.shape)
    #antigen_one_grid = np.concatenate([antigen_single_grid, antigen_esm_grid], axis=-1)
    print(antigen_single_grid.shape)
    
    n = antigen_single_grid.shape[0]

    # 将n的值以空格分隔的格式写入到 antigen_size.txt 中，采用追加的方式
    with open("/mnt/Data2/23gsy/sematrain6/antigen_size.txt", "a") as f:
        f.write(f"{n} ")
    np.save(antigen_out_grid, antigen_single_grid)
    np.save(antigen_out_resnames, antigen_names_grid)
    return None

def convert_antibody_to_images(ppi_list, config,flag = False):
    for ppi in tqdm(ppi_list):
        antibody = ppi.split(',')[1]
        antibody_pid = antibody.split('_')[0]
        antibody_ch = antibody.split('_')[1]
        antibody_out_grid = config['dirs']['grid'] + '{}_{}.npy'.format(antibody_pid, antibody_ch)
        print("Computing image for {}".format(ppi))
        if  os.path.exists(antibody_out_grid):
            print("Image is already computed {}. Skipping".format(ppi))
            continue
        convert_antibody_patch(ppi, config, flag=flag)
    return None

def convert_antigen_to_images(ppi_list, config,flag = False):
    for ppi in tqdm(ppi_list):
        antigen = ppi.split(',')[0]
        antigen_pid = antigen.split('_')[0]
        antigen_ch = antigen.split('_')[1]
        antigen_out_grid = config['dirs']['grid'] + '{}_{}.npy'.format(antigen_pid, antigen_ch)

        print("Computing image for {}".format(ppi))
        if os.path.exists(antigen_out_grid):
            print("Image is already computed {}. Skipping".format(ppi))
            continue
        convert_antigen_patch(ppi,config,flag=flag)
    return None