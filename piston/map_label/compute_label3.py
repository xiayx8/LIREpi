from Bio.PDB import PDBParser, NeighborSearch
import numpy as np
import os


import multiprocessing


def preprocess_pdb(pdb_file):
    """预处理PDB文件，只保留以ATOM开头的行"""
    with open(pdb_file, 'r') as f:
        lines = f.readlines()

    # 过滤出以ATOM开头的行
    atom_lines = [line for line in lines if line.startswith("ATOM")]

    # 将过滤后的行写入新的PDB文件
    output_pdb_file = pdb_file.replace('.pdb', '_processed.pdb')
    with open(output_pdb_file, 'w') as f:
        f.write(''.join(atom_lines))

    return output_pdb_file


def generate_matrix(pdb_file_antigen, antigen_chain, pdb_file_antibody, antibody_chain):
    """生成相应的矩阵"""
    # 预处理PDB文件，只保留以ATOM开头的行
    pdb_file_antigen_processed = preprocess_pdb(pdb_file_antigen)
    pdb_file_antibody_processed = preprocess_pdb(pdb_file_antibody)

    parser = PDBParser(QUIET=True)

    # 解析抗原PDB文件
    structure_antigen = parser.get_structure('antigen', pdb_file_antigen_processed)
    model_antigen = structure_antigen[0]
    chain_antigen = model_antigen[antigen_chain]

    # 解析抗体PDB文件
    structure_antibody = parser.get_structure('antibody', pdb_file_antibody_processed)
    model_antibody = structure_antibody[0]
    chain_antibody = model_antibody[antibody_chain]

    # 获取抗体和抗原的原子序号列表
    antibody_atoms = []
    for atom in chain_antibody.get_atoms():
        if atom.element != "H" and atom.id != "HETATM":
            antibody_atoms.append(atom.get_serial_number())

    antigen_atoms = []
    for atom in chain_antigen.get_atoms():
        if atom.element != "H" and atom.id != "HETATM":
            antigen_atoms.append(atom.get_serial_number())

    # 初始化矩阵
    matrix = np.zeros((len(antibody_atoms), len(antigen_atoms)), dtype=float)

    antibody_coords = np.array(
        [atom.coord for atom in chain_antibody.get_atoms() if atom.element != "H" and atom.id != "HETATM"])
    antigen_coords = np.array(
        [atom.coord for atom in chain_antigen.get_atoms() if atom.element != "H" and atom.id != "HETATM"])

    # Calculate the pairwise distances using broadcasting
    distances = np.linalg.norm(antibody_coords[:, np.newaxis] - antigen_coords, axis=2)

    # Create a boolean matrix based on the distance condition
    matrix = (distances < 4).astype(float)

    return matrix, antibody_atoms, antigen_atoms


def write_output(file_name, antigen_atoms, antibody_atoms, result_matrix):
    output_folder = '/mnt/Data6/23gsy/graph-piston/300_resized/atom_label'
    
    # 创建output文件夹，如果不存在
    os.makedirs(output_folder, exist_ok=True)

    # 生成完整的文件路径
    output_file_path = os.path.join(output_folder, file_name)

    with open(output_file_path, 'w') as file:
        # 写入占位符行
        file.write("-1 " + " ".join(map(str, antigen_atoms)) + "\n")

        # 写入矩阵内容
        for i, row in zip(antibody_atoms, result_matrix):
            file.write(f"{i}  " + " ".join(map(str, row)) + "\n")


# 示例使用
if __name__ == "__main__":
    with open('/mnt/Data6/23gsy/graph-piston/sema_train_wash.txt', 'r') as input_file:
        for line in input_file:
            # 移除换行符并根据逗号分割抗原和抗体
            antigen, antibody = map(str.strip, line.split(','))

            # 生成相应的文件名
            output_file_name = f"{antigen},{antibody}_label.txt"

            # # 构建抗原和抗体的文件路径及链信息
            # pdb_file_antigen = f'/mnt/Data2/23gsy/sematrain6/intermediate_files/03-chains_pdbs/{antigen}.pdb'
            # antigen_chain = antigen[-1]
            # pdb_file_antibody = f'/mnt/Data2/23gsy/sematrain6/intermediate_files/03-chains_pdbs/{antibody}.pdb'
            # antibody_chain = antibody[-1]
            # 定义文件路径
            pdb_dir_03 = '/mnt/Data2/23gsy/sematrain6/intermediate_files/03-chains_pdbs'
            pdb_dir_04 = '/mnt/Data2/23gsy/sematrain6/intermediate_files/04-chains_pdbs'

            # 检查 antigen 对应的 pdb 文件是否存在
            pdb_file_antigen = os.path.join(pdb_dir_03, f'{antigen}.pdb')
            if not os.path.exists(pdb_file_antigen):
                # 如果 03 目录下没有，检查 04 目录
                pdb_file_antigen = os.path.join(pdb_dir_04, f'{antigen}.pdb')
                if not os.path.exists(pdb_file_antigen):
                    raise FileNotFoundError(f"File {antigen}.pdb not found in either {pdb_dir_03} or {pdb_dir_04}")

            # 获取 antigen 的链信息
            antigen_chain = antigen[-1]  # 假设 antigen 的最后一个字符是链标识符

            # 检查 antibody 对应的 pdb 文件是否存在
            pdb_file_antibody = os.path.join(pdb_dir_03, f'{antibody}.pdb')
            if not os.path.exists(pdb_file_antibody):
                # 如果 03 目录下没有，检查 04 目录
                pdb_file_antibody = os.path.join(pdb_dir_04, f'{antibody}.pdb')
                if not os.path.exists(pdb_file_antibody):
                    raise FileNotFoundError(f"File {antibody}.pdb not found in either {pdb_dir_03} or {pdb_dir_04}")

            # 获取 antibody 的链信息
            antibody_chain = antibody[-1]  # 假设 antibody 的最后一个字符是链标识符
            check_output_file_path=f'/mnt/Data6/23gsy/graph-piston/300_resized/atom_label/{antigen},{antibody}_label.txt'
            if os.path.exists(check_output_file_path):
                print(f"文件 {output_file_name} 已存在，跳过该PPI对。")
                continue
            # 生成矩阵及对应的原子序号
            result_matrix, antibody_atoms, antigen_atoms = generate_matrix(pdb_file_antigen, antigen_chain,
                                                                           pdb_file_antibody, antibody_chain)
            print(antigen)
            # 写入输出文件
            write_output(output_file_name, antigen_atoms, antibody_atoms, result_matrix)

