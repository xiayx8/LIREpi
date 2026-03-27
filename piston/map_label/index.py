" this code use to get atom index of resized npy using pdb document"
#从 PDB 文件中获取特定原子的序号，并根据给定的 NumPy 数组调整和填充结果。
from Bio import PDB
import numpy as np


def get_atom_number(structure, chain_id, residue_number, residue_name, atom_name):#PDB中返回每条链的原子序号
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if residue.id[1] == residue_number and residue.resname == residue_name:
                        for atom in residue:
                            if atom.id == atom_name:
                                return atom.serial_number
    return None


def process_pdb_array(data_array, structure):#根据data_array找到对应的原子序号，提取原子信息
    result_array = np.zeros_like(data_array, dtype=int)
    # result_array (r*2,r*2,10)
    # return (r*2,r*2,10)
    for index in np.ndindex(data_array.shape[:-1]):
        for i in range(data_array.shape[-1]):
            chain_id, residue_number, full_residue_name, atom_name = data_array[index][i].split(':')

        # 处理 full_residue_name，仅保留第一个冒号之前的部分
            residue_name = full_residue_name.split('-')[0]

            atom_number = get_atom_number(structure, chain_id, int(residue_number), residue_name, atom_name)
            result_array[index][i] = atom_number if atom_number is not None else -1

    return result_array


def process_pdb_file(pdb_file_path, path):#填充原子序号
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)

    data_array = np.load(path, allow_pickle=True)
    result_array = process_pdb_array(data_array, structure)


    return result_array



