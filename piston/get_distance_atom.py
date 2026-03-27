from tqdm import tqdm
import numpy as np
import pdb
from sklearn.neighbors import KDTree
import os
from scipy.ndimage import zoom
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
def extract_filtered_coordinates(pdb_file, atom_info_array, cdr_coordinates):
    structure_dict = build_structure_dict(pdb_file)
    coordinates = extract_coordinates_from_structure(structure_dict, atom_info_array)

    filter_index = []
    for i, sub_list in enumerate(coordinates):
        if sub_list in cdr_coordinates:
            filter_index.append(i)

    return filter_index
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

        # #检查是否全部映射
        # unique_indices = np.unique(all_indices)
        # if flag == False:
        #     if len(unique_indices) != input_feat.shape[0]:
        #         print(len(unique_indices))
        #         print(input_feat.shape[0])
        #         m = input_feat.shape[0]
        #         all_indices_array = np.array(list(range(m)))
        #         result = list(set(all_indices_array) - set(unique_indices))
        #         print(result)
        #         raise ValueError("Cannot map all vertices")
    return patch_grid
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
def read_patch(pid, ch, config):
    patch_dir = config['dirs']['patches'] + pid + '/'
    check_path=patch_dir + '{}_{}_rho_wrt_center.npy'.format(pid, ch)
    # 判断文件是否存在
    if os.path.exists(check_path):
        # 文件存在时，将config['dirs']['chains_pdb']改为'./04/'
        patch_dir = config['dirs']['patches'] + pid + '/'
    else:
        # 文件不存在时，将config['dirs']['chains_pdb']改为'./03/'
        config['dirs']['patches']='/mnt/Data2/23gsy/sematrain6/intermediate_files/06-patches_16R/'
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
def convert_antibody_patch(ppi, config,flag = False):

    antibody = ppi.split(',')[1]
    antibody_pid = antibody.split('_')[0]
    antibody_ch = antibody.split('_')[1]
    print('antibody', antibody)
    antibody_grid = config['dirs']['grid'] + '{}_{}.npy'.format(antibody_pid, antibody_ch)
    # 加载 antigen_out_grid 数据
    antibody_grid_data = np.load(antibody_grid)

    # 判断 antigen_out_grid 的形状，并调整 radius
    if antibody_grid_data.shape == (antibody_grid_data.shape[0], antibody_grid_data.shape[1], 6):
        n = antibody_grid_data.shape[0]  # 获取 n 的大小
        radius = n // 2  # 设置 radius 为 n / 2
    # radius = 32
    #antibody_esm_feature=config['dirs']['esm']+'{}_esm2.npy'.format(antibody)
    p2_rho, p2_theta, p2_input_feat, p2_resnames, p2_coord_3d = read_patch(antibody_pid, antibody_ch, config)
    #antibody_esm_three_file=config['dirs']['esm'] + '{}_esmpca3.npy'.format(antibody)
    #pdb_file = config['dirs']['chains_pdb'] + antibody + ".pdb"
    # 假设config是一个字典，包含目录信息
    pdb_file_path = config['dirs']['chains_pdb'] + antibody + ".pdb"

    # 判断文件是否存在
    if os.path.exists(pdb_file_path):
        # 文件存在时，将config['dirs']['chains_pdb']改为'./04/'
        pdb_file = config['dirs']['chains_pdb'] + antibody + ".pdb"
    else:
        # 文件不存在时，将config['dirs']['chains_pdb']改为'./03/'
        config['dirs']['chains_pdb'] = '/mnt/Data2/23gsy/sematrain6/intermediate_files/04-chains_pdbs/'
        pdb_file = config['dirs']['chains_pdb'] + antibody + ".pdb"
    atom_info_array = p2_resnames
    cdr_pdb_file_path = config['dirs']['CDR'] + antibody_pid + ".pdb"
    # 判断文件是否存在
    if os.path.exists(cdr_pdb_file_path):
        # 文件存在时，将config['dirs']['chains_pdb']改为'./04/'
        cdr_pdb = config['dirs']['CDR'] + antibody_pid + ".pdb"
    else:
        # 文件不存在时，将config['dirs']['chains_pdb']改为'./03/'
        config['dirs']['CDR'] = '/mnt/Data2/23gsy/sematrain6/intermediate_files/02-antibody_cdr_pdb/'
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

    p2_patch_grid = compute_patch_grid(p2_x, p2_y, p2_coord_3d, radius,flag)  # (r, r, n_feat)
    print('p2_patch_grid',p2_patch_grid.shape)
    print("compute residues")


    return p2_patch_grid

def convert_antigen_patch(ppi, config,flag = False):

    antigen = ppi.split(',')[0]
    antigen_pid = antigen.split('_')[0]
    antigen_ch = antigen.split('_')[1]
    print('antigen',antigen)
    antigen_grid = config['dirs']['grid'] + '{}_{}.npy'.format(antigen_pid, antigen_ch)
    # 加载 antigen_out_grid 数据
    antigen_grid_data = np.load(antigen_grid)

    # 判断 antigen_out_grid 的形状，并调整 radius
    if antigen_grid_data.shape == (antigen_grid_data.shape[0], antigen_grid_data.shape[1], 6):
        n = antigen_grid_data.shape[0]  # 获取 n 的大小
        print('原来grid大小：',n)
        radius = n // 2  # 设置 radius 为 n / 2
    #antigen_out_resnames = config['dirs']['grid'] + '{}_{}_resnames.npy'.format(antigen_pid, antigen_ch)
    # radius = 32
    #antigen_esm_feature = config['dirs']['esm'] + '{}_esm2.npy'.format(antigen)
    p1_rho, p1_theta, p1_input_feat, p1_resnames, p1_coord_3d = read_patch(antigen_pid, antigen_ch, config)#加载数据
    #pdb_file = config['dirs']['chains_pdb'] + antigen + ".pdb"
    #antigen_esm_three_file = config['dirs']['esm'] + '{}_esmpca3.npy'.format(antigen)

    ##### 极坐标转化为直角坐标
    p1target_x, p1target_y = polar_to_cartesian(p1_rho, p1_theta)

    p1target_patch_grid = compute_patch_grid(p1target_x, p1target_y, p1_coord_3d, radius, flag)  # (r, r, n_feat)
    print("compute residues")

    print('p1target_patch_grid',p1target_patch_grid.shape)
    return p1target_patch_grid

def convert_antibody_to_images(ppi, config,flag = False):

    p2_coord_grid=convert_antibody_patch(ppi, config, flag=flag)
    return p2_coord_grid

def convert_antigen_to_images(ppi, config,flag = False):

    p1_coord_grid=convert_antigen_patch(ppi,config,flag=flag)
    return p1_coord_grid
# Now, you can call both functions and compute dist_grid

def resize_array(arr, resize_scale):
    target_size = (resize_scale,resize_scale,6)#目标大小
    scale_factors = [n / o for n, o in zip(target_size[:2], arr.shape[:2])]#计算缩放比例，即 resize_scale 与原始数组大小 arr.shape[:2] 的比值。
    scale_factors.append(1)
    resized_arr = zoom(arr, scale_factors, order=1)#线性插值法进行缩放
    return resized_arr
def process_ppi_images(input_file, config, flag=False):

    # Read PPI data from the input file
    with open(input_file, 'r') as f:
        for line in f:
            ppi = line.strip()  # Strip newlines and spaces
            if ppi:
                antigen = ppi.split(',')[0]
                antigen_pid = antigen.split('_')[0]
                antigen_ch = antigen.split('_')[1]
                antibody = ppi.split(',')[1]
                antibody_pid = antibody.split('_')[0]
                antibody_ch = antibody.split('_')[1]
                # Generate antibody and antigen grids

                check_out_grid = config['dirs']['dist_grid'] + '{}_dist_grids.npy'.format(ppi)
                if os.path.exists(check_out_grid):
                    print("Image is already computed {}. Skipping".format(ppi))
                    continue
                antigen_grids = convert_antigen_to_images(ppi, config, flag)
                antibody_grids = convert_antibody_to_images(ppi, config, flag)
                resized_antigen=resize_array(antigen_grids,32)
                resized_antibody=resize_array(antibody_grids,32)
                print('^^^^^^^^^^')
                print(resized_antigen.shape)
                print(resized_antibody.shape)
                # Assuming both grids are of the same length and aligned, you can compute the distances
                dist_grids = [np.sqrt(np.sum(np.square(p1 - p2), axis=-1)) for p1, p2 in zip(resized_antigen,resized_antibody)]
                # Print the shape of dist_grids
                print(f"Shape of dist_grids: {np.shape(dist_grids)}")
                output_file=f'./dist_grid/{ppi}_dist_grids.npy'
                # Save dist_grids as a .npy file
                np.save(output_file, dist_grids)
    return 1


# Example usage
if __name__ == "__main__":
    config = {
        'dirs': {
            'grid': '/mnt/Data2/23gsy/sematrain6/grid_16R/',
            'patches':'/mnt/Data2/23gsy/sematrain6/intermediate_files/05-patches_16R/',
            'chains_pdb':'/mnt/Data2/23gsy/sematrain6/intermediate_files/03-chains_pdbs/',
            'CDR':'/mnt/Data2/23gsy/sematrain6/intermediate_files/02-antibody_cdr_pdb/',
            'dist_grid':'/mnt/Data6/23gsy/graph-piston/dist_grid/'
        }
    }

    # Provide the path to input.txt
    input_file = 'sema_train_wash.txt'
    process_ppi_images(input_file, config, flag=False)

    # Do something with dist_grids (e.g., save, analyze, etc.)
    print('结束')
# Assuming both grids are of the same length and aligned, you can compute the distances

