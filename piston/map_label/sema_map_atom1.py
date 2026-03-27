"this code use to generate true label matrix after splitting patches"
import numpy as np

import os

import index
import resize
from index import *
from resize import *


def load_data(antigen, antibody, matrix_path):
    antigen = np.load(antigen, allow_pickle=True)
    antibody = np.load(antibody, allow_pickle=True)
    interaction_matrix = np.loadtxt(matrix_path)
    return antigen, antibody, interaction_matrix


def split_into_patches(data, patch):
    patch_size = (patch, patch)
    patches = []
    for i in range(0, data.shape[0], patch_size[0]):
        for j in range(0, data.shape[1], patch_size[1]):
            patch = data[i:i + patch_size[0], j:j + patch_size[1]]
            patches.append(patch)
    return np.array(patches)


def compute_map_label(antigen, antibody, matrix):
    antigen_flat = np.concatenate(antigen)
    antibody_flat = np.concatenate(antibody)
    row = np.array(np.where(np.isin(matrix[:, 0], antibody_flat))[0])
    col = np.array(np.where(np.isin(matrix[0, :], antigen_flat))[0])
    all_combinations = np.ix_(row, col)
    result = float(np.any(matrix[all_combinations] == 1))

    return result


def process_file(antigen_path, antibody_path, matrix_path, patch_size):
    antigen, antibody, matrix = load_data(antigen_path, antibody_path, matrix_path)
    antigen_patches = split_into_patches(antigen, patch_size)
    antibody_patches = split_into_patches(antibody, patch_size)
    label_out = np.zeros((len(antigen_patches), len(antibody_patches)), dtype=float)

    for i, antigen_patch in enumerate(antigen_patches):
        for j, antibody_patch in enumerate(antibody_patches):
            antigen_i = np.unique(antigen_patch)
            antibody_i = np.unique(antibody_patch)
            result = compute_map_label(antigen_i, antibody_i, matrix)
            label_out[i, j] = result
    if label_out.any() == 1:
        print("1")
        print(label_out.sum())

    return label_out


def map_atom(antigen, antibody, label_folder, feature_folder, index_folder, resize_scale, patch_size):
    # 获取原子索引
    # pdb_file_antigen = f'/mnt/Data2/23gsy/sematrain6/intermediate_files/03-chains_pdbs/{antigen}.pdb'
    # pdb_file_antibody = f'/mnt/Data2/23gsy/sematrain6/intermediate_files/03-chains_pdbs/{antibody}.pdb'
    # 文件路径
    folder_03 = '/mnt/Data2/23gsy/sematrain6/intermediate_files/03-chains_pdbs/'
    folder_04 = '/mnt/Data2/23gsy/sematrain6/intermediate_files/04-chains_pdbs/'

    # 构建PDB文件路径
    pdb_file_antigen_03 = os.path.join(folder_03, f'{antigen}.pdb')
    pdb_file_antigen_04 = os.path.join(folder_04, f'{antigen}.pdb')

    # 检查03文件夹下是否有antigen的PDB文件
    if os.path.exists(pdb_file_antigen_03):
        pdb_file_antigen=pdb_file_antigen_03
    else:
        pdb_file_antigen = pdb_file_antigen_04

    # 构建PDB文件路径
    pdb_file_antibody_03 = os.path.join(folder_03, f'{antibody}.pdb')
    pdb_file_antibody_04 = os.path.join(folder_04, f'{antibody}.pdb')

    # 检查03文件夹下是否有antigen的PDB文件
    if os.path.exists(pdb_file_antibody_03):
        pdb_file_antibody = pdb_file_antibody_03
    else:
        pdb_file_antibody = pdb_file_antibody_04

    res_file_antigen = f'/mnt/Data2/23gsy/sematrain6/grid_16R/{antigen}_resnames.npy'
    res_file_antibody = f'/mnt/Data2/23gsy/sematrain6/grid_16R/{antibody}_resnames.npy'
    feat_file_antigen = f'/mnt/Data2/23gsy/sematrain6/grid_16R/{antigen}.npy'
    feat_file_antibody = f'/mnt/Data2/23gsy/sematrain6/grid_16R/{antibody}.npy'
    antigen_atom_index = index.process_pdb_file(pdb_file_antigen, res_file_antigen)
    antibody_atom_index = index.process_pdb_file(pdb_file_antibody, res_file_antibody)

    # 调整数据大小
    antigen_feature_data = np.load(feat_file_antigen,allow_pickle=True)
    antibody_feature_data = np.load(feat_file_antibody,allow_pickle=True)
    antigen_resized_arr, antigen_scale_factors = resize_array(antigen_feature_data, resize_scale[0])
    antibody_resized_arr, antibody_scale_factors = resize_array(antibody_feature_data, resize_scale[1])

    # 获取输出索引
    antigen_out_index = resize.get_out_index(antigen_atom_index, antigen_scale_factors, antigen_feature_data,
                                             resize_scale[0])
    antibody_out_index = resize.get_out_index(antibody_atom_index, antibody_scale_factors, antibody_feature_data,
                                              resize_scale[1])

    # 存储文件
    np.save(os.path.join(index_folder, f"resized_{antigen}_index.npy"), antigen_out_index)
    np.save(os.path.join(index_folder, f"resized_{antibody}_index.npy"), antibody_out_index)
    np.save(os.path.join(feature_folder, f"resized_{antigen}_feature.npy"), antigen_resized_arr)
    np.save(os.path.join(feature_folder, f"resized_{antibody}_feature.npy"), antibody_resized_arr)
    print(antibody_resized_arr.shape)
    # 生成标签
    label = process_file(f"{index_folder}/resized_{antigen}_index.npy", f"{index_folder}/resized_{antibody}_index.npy",
                         f"/mnt/Data6/23gsy/graph-piston/300_resized/atom_label/{antigen},{antibody}_label.txt",
                         patch_size)

    # 保存标签到 label 文件夹下
    np.save(os.path.join(label_folder, f"{antigen},{antibody}_label.npy"), label)


def main():
    antigen_resized_scale = 32
    antibody_resized_scale = 32
    patch_size = 4
    label_folder = "/mnt/Data6/23gsy/graph-piston/75sema_resized/sema_train/resized_label"
    os.makedirs(label_folder, exist_ok=True)
    feature_folder = "/mnt/Data6/23gsy/graph-piston/75sema_resized/sema_train/resized_feature"
    os.makedirs(feature_folder, exist_ok=True)
    index_folder = "/mnt/Data6/23gsy/graph-piston/75sema_resized/sema_train/resized_index"
    os.makedirs(index_folder, exist_ok=True)
    train_list = [x.strip('\n') for x in
                  open("/mnt/Data6/23gsy/graph-piston/sema_train_wash.txt", 'r').readlines()]
    for list in train_list:
        antigen = list.split(',')[0]
        antibody = list.split(',')[1]
        # 构建文件路径
        check_antigen_file_path = f"/mnt/Data6/23gsy/graph-piston/75sema_resized/sema_train/resized_index/resized_{antigen}_index.npy"
        check_antibody_file_path = f"/mnt/Data6/23gsy/graph-piston/75sema_resized/sema_train/resized_index/resized_{antibody}_index.npy"

        # 检查文件是否存在
        if os.path.exists(check_antigen_file_path) and os.path.exists(check_antibody_file_path):
            print(f"Files for antigen {antigen} and antibody {antibody} already exist. Skipping...")
            continue  # 如果文件存在，跳过当前循环
        map_atom(antigen, antibody, label_folder, feature_folder, index_folder,
                 resize_scale=(antigen_resized_scale, antibody_resized_scale), patch_size=patch_size)


if __name__ == "__main__":
    main()