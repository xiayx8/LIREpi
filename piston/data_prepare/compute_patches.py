import pymesh
import pdb
import time
import numpy as np

from sklearn.manifold import MDS
from masif.source.masif_modules.read_data_from_surface import compute_ddc, normalize_electrostatics
from masif.source.geometry.compute_polar_coordinates import call_mds, compute_thetas, dict_to_sparse, compute_theta_all_fast

from sklearn.neighbors import KDTree
import os
from tqdm import tqdm
import scipy
import networkx as nx
from sklearn.neighbors import KDTree

from utils.utils import get_date
# def compute_patch_chain(pid, ch, config):
#
#
#
#
#     return
#计算了两个网格的接口顶点的几何中心，并找到这些中心在对应网格中的最近邻顶点。最终，函数返回这两个几何中心和它们在网格中的索引。
def compute_patch_center(mesh1, mesh2, radius):

    # Compute patch center and select all verticies within the range "radius"

    iface_vert1 = get_iface_verticies(mesh1)#函数获取两个网格和的接口顶点。这些顶点是网格上感兴趣的区域。
    iface_vert2 = get_iface_verticies(mesh2)

    iface_vert_all = np.concatenate((iface_vert1, iface_vert2), axis=0)

    # Compute geometric center of antigen chain
    antigen_center_point = np.mean(iface_vert1, axis=0)#计算和接口顶点的几何中心点。
    # compute geometric center of antibody chain
    antibody_center_point = np.mean(iface_vert2, axis=0)
    
    kdt1 = KDTree(mesh1.vertices)
    #pdb.set_trace()
    d, indx_cent1 = kdt1.query(np.expand_dims(antigen_center_point, axis=0))
    # patch1_indx = kdt1.query_radius(np.expand_dims(center_point, axis=0), r=radius)

    kdt2 = KDTree(mesh2.vertices)
    d, indx_cent2 = kdt2.query(np.expand_dims(antibody_center_point, axis=0))

    return antigen_center_point,antibody_center_point, indx_cent1[0][0], indx_cent2[0][0] #, patch1_indx[0], patch2_indx[0]

def get_iface_verticies(mesh):#从网格中获取接口顶点
    iface = mesh.get_attribute('vertex_iface')#从网格对象 mesh 中获取名为 'vertex_iface' 的属性，存储在 iface 变量中。这个属性通常标识哪些顶点是网格的接口顶点
    vertices = mesh.vertices
    iface_indx = np.where(iface>=0)#找到 iface 中所有大于等于 0 的元素的索引。iface 中这些值通常表示接口顶点的索引或标识。
    #pdb.set_trace()
    if len(iface_indx[0])==0:
        print('WARNING:: No interface found!')
        iface_indx = np.where(iface==0)
    return vertices[iface_indx]


def compute_theta_all(D, vertices, faces, normals, idx, radius, patch_center_i):#计算给定点的所有邻居的极坐标角度
    # Reference: https://github.com/LPDI-EPFL/masif/blob/2a370518e0d0d0b0d6f153f2f10f6630ae91f149/source/geometry/compute_polar_coordinates.py#L300

    mymds = MDS(n_components=2, n_init=1, max_iter=50, dissimilarity='precomputed', n_jobs=10)
    all_theta = []
    i = patch_center_i
    if i % 100 == 0:
        print(i)
    # Get the pairs of geodesic distances.

    neigh = D[i].nonzero()
    ii = np.where(D[i][neigh] > 0)[1]
    neigh_i = neigh[1][ii]
    pair_dist_i = D[neigh_i, :][:, neigh_i]
    pair_dist_i = pair_dist_i.todense()

    # Plane_i: the 2D plane for all neighbors of i
    plane_i = call_mds(mymds, pair_dist_i)

    # Compute the angles on the plane.
    theta = compute_thetas(plane_i, i, vertices, faces, normals, neigh_i, idx)
    return theta


def compute_polar_coordinates(mesh, patch_center_i,  radius=12, max_vertices=200):#计算网格中每个小区域的极坐标（径向距离和角度）
    """
    # Reference: https://github.com/LPDI-EPFL/masif/blob/2a370518e0d0d0b0d6f153f2f10f6630ae91f149/source/geometry/compute_polar_coordinates.py#L19
    compute_polar_coordinates: compute the polar coordinates for every patch in the mesh.
    Returns:
        rho: radial coordinates for each patch. padded to zero.
        theta: angle values for each patch. padded to zero.
        neigh_indices: indices of members of each patch.
        mask: the mask for rho and theta
    """

    # Vertices, faces and normals
    vertices = mesh.vertices
    faces = mesh.faces
    norm1 = mesh.get_attribute('vertex_nx')#每个顶点的法线分量
    norm2 = mesh.get_attribute('vertex_ny')
    norm3 = mesh.get_attribute('vertex_nz')
    normals = np.vstack([norm1, norm2, norm3]).T#将法线分量组合成一个二维数组，每一行对应一个顶点的法线向量。

    # Graph
    G = nx.Graph()#创建一个空的图 G，节点数量等于顶点的数量，并将每个顶点添加为图中的节点。
    n = len(mesh.vertices)
    G.add_nodes_from(np.arange(n))

    # Get edges
    f = np.array(mesh.faces, dtype=int)#从face中提取信息
    rowi = np.concatenate([f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]], axis=0)#rowi 和 rowj 是顶点对（边）的索引，表示哪些顶点通过三角形连接在一起。
    rowj = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 2], f[:, 0], f[:, 1]], axis=0)
    edges = np.stack([rowi, rowj]).T#每条边由一对顶点构成
    verts = mesh.vertices

    # Get weights
    edgew = verts[rowi] - verts[rowj]#计算每条边的权重（边的长度）
    edgew = scipy.linalg.norm(edgew, axis=1)#欧氏距离
    wedges = np.stack([rowi, rowj, edgew]).T

    G.add_weighted_edges_from(wedges)#将带有权重的边添加到图 G 中，形成一个加权图。
    start = time.clock()

    dists = nx.all_pairs_dijkstra_path_length(G)#使用Dijkstra算法计算图中所有节点对之间的最短路径长度。

    d2 = {}
    for key_tuple in dists:
        d2[key_tuple[0]] = key_tuple[1]
    end = time.clock()
    print('Dijkstra took {:.2f}s'.format((end - start)))#保存，记录执行时间
    D = dict_to_sparse(d2)
    num_of_items_in_dist = len(d2.items())#将字典 d2 转换为稀疏矩阵 D，表示顶点对之间的距离。

    # Compute the faces per vertex.
    idx = {}#为每个顶点计算其所属的面索引，并存储在字典 idx 中。
    for ix, face in enumerate(mesh.faces):
        for i in range(3):
            if face[i] not in idx:
                idx[face[i]] = []
            idx[face[i]].append(ix)

    i = np.arange(D.shape[0])
    # Set diagonal elements to a very small value greater than zero..
    D[i, i] = 1e-8
    # Call MDS for all points.
    mds_start_t = time.clock()

    theta = compute_theta_all(D, vertices, faces, normals, idx, radius, patch_center_i)#计算角度

    # Output a few patches for debugging purposes.
    # extract a patch
    # for i in [0,100,500,1000,1500,2000]:
    #    neigh = D[i].nonzero()
    #    ii = np.where(D[i][neigh] < radius)[1]
    #    neigh_i = neigh[1][ii]
    #    subv, subn, subf = extract_patch(mesh, neigh_i, i)
    #    # Output the patch's rho and theta coords
    #    output_patch_coords(subv, subf, subn, i, neigh_i, theta[i], D[i, :])

    mds_end_t = time.clock()
    print('MDS took {:.2f}s'.format((mds_end_t - mds_start_t)))#记录MDS执行时间

    n = len(d2)#图中顶点的数量
    i = patch_center_i#选择patch_center_i作为中心顶点
    # Assemble output.

    dists_i = d2[i]#dists_i 获取 patch_center_i 顶点到其他所有顶点的距离，存储在字典 d2 中，d2[i] 是以 i 为起点的最短路径距离。
    sorted_dists_i = sorted(dists_i.items(), key=lambda kv: kv[1])#对 dists_i 的字典按距离值 kv[1] 进行升序排序，结果是一个包含顶点编号及其对应距离的列表。
    neigh = [int(x[0]) for x in sorted_dists_i[0:max_vertices]]#从排序后的列表中选取前 max_vertices 个顶点的编号，作为该区域的邻居顶点。

    theta_out = np.zeros((len(neigh)))#角度信息
    rho_out = np.zeros((len(neigh)))#径向距离信息
    mask_out = np.zeros((len(neigh)))#掩码，标记哪些顶点有效

    rho_out[:len(neigh)] = np.squeeze(np.asarray(D[i, neigh].todense()))#提取i到邻居节点距离的数据，变成一位数组，将径向距离赋值给rho_out数组
    theta_out[:len(neigh)] = np.squeeze(theta[neigh])#将计算得到的角度 theta 对应的邻居顶点部分提取出来
    mask_out[:len(neigh)] = 1#设置 mask_out 的前 len(neigh) 个值为 1，表示这些顶点是有效的邻居。
    # have the angles between 0 and 2*pi
    theta_out[theta_out < 0] += 2 * np.pi
    #返回邻居顶点的径向距离，邻居顶点的角度，索引，掩码（表示那些顶点是有效的邻居）
    return rho_out, theta_out, neigh, mask_out

def read_data_from_surface(ply_fn1,patch_center_i, config):
    """
    # Reference:
    #   https://github.com/LPDI-EPFL/masif/blob/2a370518e0d0d0b0d6f153f2f10f6630ae91f149/source/masif_modules/read_data_from_surface.py#L14
    # Read data from a ply file -- decompose into patches.
    # Returns:
    # list_desc: List of features per patch
    # list_coords: list of angular and polar coordinates.
    # list_indices: list of indices of neighbors in the patch.
    # list_sc_labels: list of shape complementarity labels (computed here).
    """
    mesh = pymesh.load_mesh(ply_fn1)

    # Normals:
    n1 = mesh.get_attribute("vertex_nx")
    n2 = mesh.get_attribute("vertex_ny")
    n3 = mesh.get_attribute("vertex_nz")
    normals = np.stack([n1, n2, n3], axis=1)

    # Compute the angular and radial coordinates.
    radius = config['ppi_const']['patch_r']
    # conclude all vertices in the patch
    points_in_patch = len(mesh.vertices)
    rho, theta, neigh_indices, mask = compute_polar_coordinates(mesh, patch_center_i, radius=radius,
                                                                max_vertices=points_in_patch)
    print(rho.shape)
    # Compute the principal curvature components for the shape index.
    mesh.add_attribute("vertex_mean_curvature")
    H = mesh.get_attribute("vertex_mean_curvature")#获取顶点的平均曲率
    mesh.add_attribute("vertex_gaussian_curvature")
    K = mesh.get_attribute("vertex_gaussian_curvature")#获取顶点的高斯曲率
    elem = np.square(H) - K
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem < 0] = 1e-8
    k1 = H + np.sqrt(elem)#主曲率
    k2 = H - np.sqrt(elem)
    # Compute the shape index
    si = (k1 + k2) / (k1 - k2)#形状指数，描述表面局部的几何形状
    si = np.arctan(si) * (2 / np.pi)

    # Normalize the charge.
    charge = mesh.get_attribute("vertex_charge")
    charge = normalize_electrostatics(charge)#获取电荷属性，归一化

    # Hbond features
    hbond = mesh.get_attribute("vertex_hbond")#氢键

    # Hydropathy features
    # Normalize hydropathy by dividing by 4.5
    hphob = mesh.get_attribute("vertex_hphob") / 4.5#疏水性

    # Iface labels (for ground truth only)
    if "vertex_iface" in mesh.get_attribute_names():
        iface_labels = mesh.get_attribute("vertex_iface")
    else:
        iface_labels = np.zeros_like(hphob)

    # n: number of patches, equal to the number of vertices.
    n = len(mesh.vertices)

    input_feat = np.zeros((len(neigh_indices), 5))

    # Compute the input features for each patch.
    vix = patch_center_i
    # Patch members.
    neigh_vix = np.array(neigh_indices)

    # Compute the distance-dependent curvature for all neighbors of the patch.
    patch_v = mesh.vertices[neigh_vix]
    patch_n = normals[neigh_vix]
    patch_cp = np.where(neigh_vix == vix)[0][0]  # central point
    mask_pos = np.where(mask == 1.0)[0]  # nonzero elements
    patch_rho = rho[mask_pos]  # nonzero elements of rho
    ddc = compute_ddc(patch_v, patch_n, patch_cp, patch_rho)#计算距离相关的曲率ddc，考虑到邻居顶点与中心点的距离来调整曲率特征。

    # impute missing shape indicies with mean value for the whole patch
    si_patch = si[neigh_vix]

    si_patch = np.nan_to_num(si_patch, nan=np.nanmean(si_patch)) # replace nan values


    input_feat[:len(neigh_vix), 0] = si_patch
    input_feat[:len(neigh_vix), 1] = ddc
    input_feat[:len(neigh_vix), 2] = hbond[neigh_vix]
    input_feat[:len(neigh_vix), 3] = charge[neigh_vix]
    input_feat[:len(neigh_vix), 4] = hphob[neigh_vix]#保存，形状指数，ddc，氢键、电荷、疏水性

    print(input_feat.shape)

    return input_feat, rho, theta, mask, neigh_indices, iface_labels, np.copy(mesh.vertices)


# def read_ply(pid, ch, config):
#     ply_file = config['dirs']['surface_ply'] + pid + '_' + ch + '.ply'
#     ply_header = []
#     ply_entries = []
#     header=True
#     with open(ply_file, 'r') as f:
#         for line in f.readlines():
#             if header:
#                 ply_header.append(line)
#             else:
#                 ply_entries.append(line)
#             if "end_header" in line:
#                 header=False
#     return ply_header, ply_entries


# def crop_ply_patch(pid, ch, patch_indx, config):
#
#     out_cropped_ply_patch = config['dirs']['patch_ply'] + pid+'_'+ch + '_patch.ply'
#
#     ply_header, ply_entries = read_ply(pid,ch, config)
#     ply_entries = np.array(ply_entries)
#     ply_entries = ply_entries[patch_indx]
#
#     # write cropped ply
#     with open(out_cropped_ply_patch, 'w') as out:
#         for header_line in ply_header:
#             out.write(header_line)
#         for entry_line in ply_entries:
#             out.write(entry_line)


def save_precompute(pid, ch, config, input_feat, rho, theta, mask, neigh_indices, iface_labels, verts, center_patch_i, patch_coord):

    out_patch_dir = config['dirs']['patches']
    my_precomp_dir = out_patch_dir + pid + '/'
    if not os.path.exists(my_precomp_dir):
        os.mkdir(my_precomp_dir)

    np.save(my_precomp_dir + pid + '_' + ch + '_rho_wrt_center', rho)
    np.save(my_precomp_dir + pid + '_' + ch + '_theta_wrt_center', theta)
    np.save(my_precomp_dir + pid + '_' + ch + '_input_feat', input_feat)
    np.save(my_precomp_dir + pid + '_' + ch + '_mask', mask)
    np.save(my_precomp_dir + pid + '_' + ch + '_list_indices', neigh_indices)
    np.save(my_precomp_dir + pid + '_' + ch + '_iface_labels', iface_labels)
    # Save x, y, z
    np.save(my_precomp_dir + pid + '_' + ch + '_X.npy', verts[center_patch_i, 0])
    np.save(my_precomp_dir + pid + '_' + ch + '_Y.npy', verts[center_patch_i, 1])
    np.save(my_precomp_dir + pid + '_' + ch + '_Z.npy', verts[center_patch_i, 2])

    np.save(my_precomp_dir + pid + '_' + ch + '_X_all.npy', verts[:, 0])
    np.save(my_precomp_dir + pid + '_' + ch + '_Y_all.npy', verts[:, 1])
    np.save(my_precomp_dir + pid + '_' + ch + '_Z_all.npy', verts[:, 2])

    np.save(my_precomp_dir + pid + '_' + ch + '_coordinates.npy', patch_coord)


# def save_precompute_interaction(pid, inter_feat, config):
#     """
#     Save interaction features into numpy files
#     :param pid: PDB ID
#     :param inter_feat: dictionary of interaction features
#     :param config: configuration dictionary
#     :return: None
#     """
#     out_patch_dir = config['dirs']['patches']
#     my_precomp_dir = out_patch_dir + pid + '/'
#
#     if not os.path.exists(my_precomp_dir):
#         os.mkdir(my_precomp_dir)
#
#     for key in inter_feat.keys():
#         np.save(my_precomp_dir + pid + '_{}.npy'.format(key), inter_feat[key])
#
#     return None

def compute_patches(ppi_list, config, overwrite=False):
    radius = config['ppi_const']['patch_r']

    print("**** [ {} ] Compute patches".format(get_date()))
    print("{}".format(ppi_list))

    for ppi in tqdm(ppi_list):
        print("Computing patch pair for {}".format(ppi))
        antigen = ppi.split(',')[0]
        antibody = ppi.split(',')[1]
        antigen_pid = antigen.split('_')[0]
        antigen_ch = antigen.split('_')[1]
        antibody_pid = antibody.split('_')[0]
        antibody_ch = antibody.split('_')[1]

        antigen_out_feat = config['dirs']['patches']+'{}/{}_{}_input_feat.npy'.format(antigen_pid, antigen_pid, antigen_ch)
        antibody_out_feat = config['dirs']['patches']+'{}/{}_{}_input_feat.npy'.format(antibody_pid, antibody_pid, antibody_ch)
        if os.path.exists(antigen_out_feat) and os.path.exists(antibody_out_feat):
            print('Patch already computed for {}. Skipping...'.format(ppi))
            continue

        ply_dir = config['dirs']['surface_ply']

        ply_fn1 = ply_dir + '{}_{}.ply'.format(antigen_pid, antigen_ch)
        ply_fn2 = ply_dir + '{}_{}.ply'.format(antibody_pid, antibody_ch)

        mesh1 = pymesh.load_mesh(ply_fn1)
        mesh2 = pymesh.load_mesh(ply_fn2)

        # Compute patch center
        center1,center2,antigen_patch_center,antibody_patch_center = compute_patch_center(mesh1, mesh2, radius)
        print('Center of the interaction: {},{}'.format(center1,center2))
        print('Center of the individual antigen and antibody: {},{}'.format(antigen_patch_center,antibody_patch_center))
        # crop_ply_patch(pid, ch1, patch1_indx, config)
        # crop_ply_patch(pid, ch2, patch2_indx, config)


        input_feat1, rho1, theta1, mask1, neigh_indices1, iface_labels1, verts1 = read_data_from_surface(ply_fn1,antigen_patch_center, config)
        input_feat2, rho2, theta2, mask2, neigh_indices2, iface_labels2, verts2 = read_data_from_surface(ply_fn2,antibody_patch_center, config)

        # Compute 3D coordinates of the patch (used to compute the distance between atoms)
        points_in_patch_antigen = len(neigh_indices1)
        points_in_patch_antibody = len(neigh_indices2)
        patch_coord1, patch_coord2 = np.zeros((points_in_patch_antigen, 3)), np.zeros((points_in_patch_antibody, 3))
        patch_coord1[:len(neigh_indices1)] = verts1[neigh_indices1]
        patch_coord2[:len(neigh_indices2)] = verts2[neigh_indices2]



        save_precompute(antigen_pid, antigen_ch, config, input_feat1, rho1, theta1, mask1, neigh_indices1, iface_labels1, verts1, antigen_patch_center, patch_coord1)
        save_precompute(antibody_pid, antibody_ch, config, input_feat2, rho2, theta2, mask2, neigh_indices2, iface_labels2, verts2, antibody_patch_center, patch_coord2)




# def compute_atom_dist(patch_coord1, patch_coord2):
#     # For each coordinate find the closest atom from the other side
#
#     kdt1 = KDTree(patch_coord1)
#     dist, _ = kdt1.query(patch_coord2)
#
#     return dist

# def compute_interaction_features(config):
#     # Compute distance between atoms:
#     if 'features' in config.keys() and 'atom_dist' in config['features'].keys() and config['features']['atom_dist']:
#         atom_dist = compute_atom_dist()

    # Compute
