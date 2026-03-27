import numpy as np
import pandas as pd
from Bio.PDB import *
from scipy.spatial import cKDTree
import pdb
from Bio import Entrez,SeqIO, BiopythonWarning
import warnings
import os

# Ignore biopython warnings
warnings.simplefilter('ignore', BiopythonWarning)

def get_start_res(resid, chain_id):#得到这条链上的残基起始编号
    chain_curr = ''
    start_res = []
    start_res_curr = resid[0]
    for i, res_i in enumerate(resid):
        if chain_id[i]!=chain_curr:
            start_res_curr=res_i
            chain_curr=chain_id[i]
        start_res.append(start_res_curr)
    return np.array(start_res)

def map_patch_indices(pid, ch, config):
    # produce "res_names.npy" that maps patch to residue names
    mapping_table = config['dirs']['patches'] + pid + '/' + pid + "_" + ch + "_map.csv"
    mapping_df = pd.read_csv(mapping_table)
    indices_np = np.load(config['dirs']['patches'] + pid + '/' + pid + '_' + ch + '_list_indices.npy')
    out_map = config['dirs']['patches'] + pid + '/' + pid + '_' + ch + '_resnames'

    mapping_df = mapping_df[mapping_df['patch_ind'].isin(indices_np)]
    res_names = np.array(['x' for i in range(len(indices_np))], dtype=object)
    for i,patch_i in enumerate(indices_np):
        tmp_df = mapping_df[mapping_df['patch_ind']==patch_i].reset_index(drop=True)
        res_name_i = '{}:{}:{}-{}:{}'.format(tmp_df.loc[0]['chain_id'],tmp_df.loc[0]['res_ind'],
                                           tmp_df.loc[0]['residue_name'],tmp_df.loc[0]['atom_ind'],
                                            tmp_df.loc[0]['atom_name']) #chain:resid:res_name-atom_id:atom_name
        res_names[i] = res_name_i
    np.save(out_map, res_names)

def map_patch_atom(ppi_list, config):
    print("Mapping each patch to a residue number...")
    print(ppi_list)
    for ppi in ppi_list:
        antigen = ppi.split(',')[0]
        antibody = ppi.split(',')[1]
        antigen_pid = antigen.split('_')[0]
        antigen_ch = antigen.split('_')[1]
        antibody_pid = antibody.split('_')[0]
        antibody_ch = antibody.split('_')[1]
        map_patch_atom_one(antigen_pid, antigen_ch, config)
        map_patch_atom_one(antibody_pid, antibody_ch, config)
        map_patch_indices(antigen_pid, antigen_ch, config)
        map_patch_indices(antibody_pid, antibody_ch, config)


def map_patch_atom_one(pid, ch, config):
    pdb_id = pid
    chain_name = ch
    patch_dir = config['dirs']['patches'] + pdb_id + '/'
    pdb_chain_dir = config['dirs']['chains_pdb']
    out_mappings_dir = config['dirs']['patches'] + pdb_id + '/'


    out_table = out_mappings_dir + "/" + pdb_id + "_" + chain_name  + "_map.csv"

    # Read coordinates of
    x_coord = np.load(patch_dir+"/{}_{}_X_all.npy".format(pdb_id, chain_name))#加载patch的xyz坐标，堆叠成一个num_patches,3大小的patch坐标矩阵
    y_coord = np.load(patch_dir + "/{}_{}_Y_all.npy".format(pdb_id, chain_name))
    z_coord = np.load(patch_dir + "/{}_{}_Z_all.npy".format(pdb_id, chain_name))
    patch_coord = np.column_stack((x_coord,y_coord,z_coord))

    # Read interface
    iface_labels = np.load(patch_dir+"/{}_{}_iface_labels.npy".format(pdb_id, chain_name))#读取patch的界面标签，用于标记patch是否在界面上

    # Read PDB structure
    pdb_path = "{}/{}_{}.pdb".format(pdb_chain_dir, pdb_id, chain_name)

    parser = PDBParser()
    pdb_struct = parser.get_structure('{}_{}'.format(pdb_id, chain_name), pdb_path)

    ## Get heavy atoms，筛选得到重原子，也就是非氢原子且不是异质原子（非水、非配体等）的重原子
    heavy_atoms=[]
    heavy_orig_map = {}
    k=0
    for i, atom in enumerate(pdb_struct.get_atoms()):
        tags = atom.parent.get_full_id()
        if atom.element!='H' and tags[3][0]==' ': # if heavy atom and not heteroatom
            heavy_orig_map[k]=i #map heavy atom index to original pdb index
            heavy_atoms.append(atom)
            k+=1
    #获取重原子的……
    atom_coord = np.array([list(atom.get_coord()) for atom in heavy_atoms])
    atom_names = np.array([atom.get_id() for atom in heavy_atoms])
    residue_id = np.array([atom.parent.id[1] for atom in heavy_atoms])
    residue_name = np.array([atom.parent.resname for atom in heavy_atoms])
    chain_id = np.array([atom.get_parent().get_parent().get_id() for atom in heavy_atoms])

    # get start residue
    start_res = get_start_res(residue_id, chain_id)#计算残基起始编号

    #Create KD Tree
   # patch_tree = cKDTree(patch_coord)
    pdb_tree = cKDTree(atom_coord)
    #对补丁坐标 patch_coord 进行查询，找到与每个补丁最近的重原子，并返回最近原子的距离 dist 和对应的索引 idx。
    dist, idx = pdb_tree.query(patch_coord) #idx is the index of pdb heavy atoms that close to every patch from [0 to N patches]
    result_pdb_idx=[]
    for i in idx:
        result_pdb_idx.append(heavy_orig_map[i])
    result_pdb_idx = np.array(result_pdb_idx) #index in original pdb
    #Combine everything to a table:
    df = pd.DataFrame({"patch_ind":range(0, len(result_pdb_idx)),
                       "atom_ind":result_pdb_idx,
                       "res_ind": residue_id[idx],
                       "atom_name":atom_names[idx],
                       "residue_name":residue_name[idx],
                       "chain_id":chain_id[idx],
                       "dist": dist,
                       "iface_label":iface_labels,
                       "start_res": start_res[idx]
                       })

    #返回patch索引、原子序号，残基编号、原子名、残基名，链id，patch到原子的距离、界面标签、起始残基编号
    df.to_csv(out_table, index=False)
