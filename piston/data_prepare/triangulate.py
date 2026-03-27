import pymesh #Importing pymesh here avoids library conflict (CXXABI_1.3.11)
import numpy as np
import os
import pdb

from shutil import copyfile, rmtree

# Local includes
from utils.utils import get_date, extract_pdb_chain

# MaSIF includes
import pymesh
import traceback

from masif.source.default_config.masif_opts import masif_opts
from masif.source.triangulation.computeMSMS import computeMSMS
from masif.source.triangulation.fixmesh import fix_mesh
from masif.source.input_output.extractPDB import extractPDB
from masif.source.input_output.save_ply import save_ply
from masif.source.triangulation.computeHydrophobicity import computeHydrophobicity
from masif.source.triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from masif.source.triangulation.computeAPBS import computeAPBS
from masif.source.triangulation.compute_normal import compute_normal
from sklearn.neighbors import KDTree

from tqdm import tqdm



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

def extract_unique_atoms(names):
    """
    从给定的名字数组中提取所有不同的链名、残基序号、残基名、原子名的组合

    参数:
        names (numpy.ndarray): 包含名字的数组

    返回值:
        list: 所有链名、残基序号、残基名、原子名的组合列表
    """
    unique_atoms = []
    for item in names:
        parts = item.split('_')
        chain_name = parts[0]
        residue_number = parts[1]
        residue_name = parts[3]
        atom_name = parts[4]
        unique_atoms.append((chain_name, residue_number, residue_name, atom_name))
    return unique_atoms

def extract_coordinates_from_pdb(coordiantes, unique_atoms):
    index = 0
    unique_coordinates=[]
    for atom in unique_atoms:
        value = coordiantes.get(atom)
        unique_coordinates.append(value)

    return unique_coordinates

def extract_pdb_coordinates(pdb_file, chain_id):
    atom_coordinates = {}

    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith('ATOM'):
                chain = line[21]
                if chain == chain_id:
                    residue_seq = str(line[22:26].strip())
                    residue_name = line[17:20].strip()
                    atom_name = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    key = (chain, residue_seq, residue_name, atom_name)
                    value = (x, y, z)
                    atom_coordinates[key] = value

    return atom_coordinates

def find_cdr_atom(names_array,pdb_file,cdr_file,chain_id):
    names = names_array
    full_pdb_coordinates = extract_pdb_coordinates(pdb_file, chain_id)
    unique_atoms = extract_unique_atoms(names)
    cdr_atom_coordinates = extract_cdr_coordinates(cdr_file, chain_id)
    pdb_coordinates = extract_coordinates_from_pdb(full_pdb_coordinates, unique_atoms)
    filter_index = [i for i, coord in enumerate(pdb_coordinates) if coord in cdr_atom_coordinates]

    return filter_index

def triangulate_one(pid, ch, config, pdb_filename,flag):#flag未被使用
    """
    triangulate one chain
    """
    chains_pdb_dir = config['dirs']['chains_pdb']
    cdr_pdb_dir = config['dirs']['CDR']
    tmp_pdb_dir = chains_pdb_dir + pid + '_' + ch + '/'
    if not os.path.exists(tmp_pdb_dir):
        os.mkdir(tmp_pdb_dir)

    print('1\n')
    # Extract chains for each interacting protein
    out_filename1 = tmp_pdb_dir + pid + '_' + ch
    extractPDB(pdb_filename, out_filename1+ '.pdb', ch)
    print('2\n')
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename1+ '.pdb', protonate=True)
    print('3\n')
    vertex_hbond = computeCharges(out_filename1, vertices1, names1)#计算氢键
    print('4\n')
    vertex_hphobicity = computeHydrophobicity(names1)#计算疏水性，每一种氨基酸都有对应的氢键值
    print('5\n')

    vertices2 = vertices1
    faces2 = faces1

    # Fix the mesh.
    mesh = pymesh.form_mesh(vertices2, faces2)
    regular_mesh = fix_mesh(mesh, config['mesh']['mesh_res'])#形成网格
    print('6\n')

    # Compute the normals
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)#计算顶点法线
    print('7\n')
    # hbonds，将氢键和疏水性值分配给新的正则化网格。
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
                                          vertex_hbond, masif_opts)
    print('8\n')
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, \
                                               vertex_hphobicity, masif_opts)
    print('9\n')
    #计算电荷
    vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1 + ".pdb", out_filename1)
    print('10\n')
    #copyfile(out_filename1, chains_pdb_dir + '{}_{}.pdb'.format(pid, ch))
    extract_pdb_chain(config['dirs']['cropped_pdb'] + pid + '.pdb',  chains_pdb_dir + '{}_{}.pdb'.format(pid, ch), ch)
    rmtree(tmp_pdb_dir)

    iface = np.zeros(len(regular_mesh.vertices))

    v3, f3, _, _, _ = computeMSMS(pdb_filename, protonate=True)
    # Regularize the mesh
    mesh = pymesh.form_mesh(v3, f3)
    # I believe It is not necessary to regularize the full mesh. This can speed up things by a lot.
    full_regular_mesh = mesh
    # Find the vertices that are in the iface.
    v3 = full_regular_mesh.vertices
    # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
    kdt = KDTree(v3)#使用 KDTree 查找正则化网格和完整复合体的顶点之间的距离，以确定接口顶点。以便后续为虚拟点添加特征
    d, r = kdt.query(regular_mesh.vertices)
    d = np.square(d)  # Square d, because this is how it was in the pyflann version.
    assert (len(d) == len(regular_mesh.vertices))
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0
    # Convert to ply and save.

    outply = config['dirs']['surface_ply'] + pid + '_' + ch
    save_ply(outply + ".ply", regular_mesh.vertices, \
             regular_mesh.faces, normals=vertex_normal, charges=vertex_charges, \
             normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity, \
             iface=iface)

    return

def triangulate_antigen(ppi, config, overwrite=False):
    antigen = ppi.split(',')[0]
    antigen_pid = antigen.split('_')[0]
    antigen_ch = antigen.split('_')[1]

    antigen_outply = config['dirs']['surface_ply'] + antigen_pid + '_' + antigen_ch + '.ply'

    if not overwrite and os.path.exists(antigen_outply):
        print("Triangulated structures already exist for {}. Skipping...".format(ppi))
        return

    #pdb_filename = config['dirs']['cropped_pdb'] + pid + '.pdb'
    try:
        antigen_pdb_filename = config['dirs']['cropped_pdb'] + antigen_pid + '.pdb'
        flag = False
        triangulate_one(antigen_pid, antigen_ch, config, antigen_pdb_filename,flag)
    except:
        print("WARNING:: can't triangulate cropped PDB")
        fail_to_tr_file = '/mnt/Data2/23gsy/sematrain6/fail_to_tr.txt'
        if not os.path.exists(fail_to_tr_file):
            # If the file doesn't exist, create it
            with open(fail_to_tr_file, 'w') as f:
                f.write(antigen + "\n")
        else:
            with open(fail_to_tr_file, 'r') as f:
                failed_antigens = f.readlines()

            failed_antigens = [line.strip() for line in failed_antigens]  # Remove newlines

            if antigen not in failed_antigens:
                # Add the antibody to fail_to_tr.txt if it's not already present
                with open(fail_to_tr_file, 'a') as f:
                    f.write(antigen + "\n")
        antigen_pdb_filename = config['dirs']['protonated_pdb'] + antigen_pid + '.pdb'
        flag = False
        triangulate_one(antigen_pid, antigen_ch, config, antigen_pdb_filename,flag)
    return

def triangulate_antibody(ppi, config, overwrite=False):
    antibody = ppi.split(',')[1]
    antibody_pid = antibody.split('_')[0]
    antibody_ch = antibody.split('_')[1]

    antibody_outply = config['dirs']['surface_ply'] + antibody_pid + '_' + antibody_ch + '.ply'
    if not overwrite and os.path.exists(antibody_outply):
        # Skip if ply file is already existsc
        print("Triangulated structures already exist for {}. Skipping...".format(ppi))
        return

    try:
        antibody_pdb_filename = config['dirs']['cropped_pdb'] + antibody_pid + '.pdb'
        flag = True
        triangulate_one(antibody_pid, antibody_ch, config, antibody_pdb_filename,flag)
    except:
        print("WARNING:: can't triangulate cropped PDB")
        # Check if the antibody is already in fail_to_tr.txt
        fail_to_tr_file = '/mnt/Data2/23gsy/sematrain6/fail_to_tr.txt'
        if not os.path.exists(fail_to_tr_file):
            # If the file doesn't exist, create it
            with open(fail_to_tr_file, 'w') as f:
                f.write(antibody+ "\n")
        else:
            with open(fail_to_tr_file, 'r') as f:
                failed_antibodies = f.readlines()

            failed_antibodies = [line.strip() for line in failed_antibodies]  # Remove newlines

            if antibody not in failed_antibodies:
                # Add the antibody to fail_to_tr.txt if it's not already present
                with open(fail_to_tr_file, 'a') as f:
                    f.write(antibody+ "\n")
        antibody_pdb_filename = config['dirs']['protonated_pdb'] + antibody_pid + '.pdb'
        flag = True
        triangulate_one(antibody_pid, antibody_ch, config, antibody_pdb_filename,flag)
        # Check if the antibody is already in fail_to_tr.txt

    return

def triangulate(ppi_list, config):
    print("\t[ {} ] Start triangulation... ".format(get_date()))
    print(ppi_list)

    processed_ppi = []

    for ppi in tqdm(ppi_list):
        antigen = ppi.split(',')[0]
        antibody = ppi.split(',')[1]
        antigen_pid = antigen.split('_')[0]
        antigen_ch = antigen.split('_')[1]
        antibody_pid = antibody.split('_')[0]
        antibody_ch = antibody.split('_')[1]

        antigen_outply = config['dirs']['surface_ply'] + antigen_pid + '_' + antigen_ch + '.ply'
        antibody_outply = config['dirs']['surface_ply'] + antibody_pid + '_' + antibody_ch + '.ply'
        if os.path.exists(antigen_outply) and os.path.exists(antibody_outply) or os.path.exists(f"{config['dirs']['grid']}/{antigen}.npy") and os.path.exists(f"{config['dirs']['grid']}/{antibody}.npy"):
            # Skip if ply file is already exists
            print("Triangulated structures already exist for {}. Skipping...".format(ppi))
            processed_ppi.append(ppi)
            continue

        try:
            antigen_pdb_filename = config['dirs']['cropped_pdb'] + antigen_pid + '.pdb'
            antibody_pdb_filename = config['dirs']['cropped_pdb'] + antibody_pid + '.pdb'
            triangulate_one(antigen_pid, antigen_ch, config, antigen_pdb_filename)
            triangulate_one(antibody_pid, antibody_ch, config, antibody_pdb_filename)
        except:
            print("Can't process {}".format(ppi))
            traceback.print_exc()
            #exit()

        if os.path.exists(antigen_outply) and os.path.exists(antibody_outply):
            # append to processed if output fiile exists
            processed_ppi.append(ppi)
            continue

    return processed_ppi

