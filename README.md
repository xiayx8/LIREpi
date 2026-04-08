# LRiEpi

## Project Introduction
This work mainly focuses on the task of B-cell epitope prediction. The model simultaneously utilizes the structural and sequence information of proteins: Firstly, it extracts structural features using MaSIF and Vision Transformer, and sequence features using ESM-2; then, it fuses the structural and sequence features through the FiLM modulation mechanism, and finally feeds them into an equivariant graph neural network to complete the epitope prediction. Compared with methods that only use single-modal features, this framework can more fully represent the information of antigenic residues, thereby improving the prediction performance. 

## Method Process
The overall process is as follows: 

### Structural Feature Extraction
`piston/data_prepare/data_prepare.py` - This script will automatically invoke other scripts in this folder, namely triangulating the molecular surface, mapping the 3D fine-grained vertices to a 2D grid, and obtaining the structural representation in npy format.
There are two ways to run this script:
Run a single PPI：
python ../piston infer --pdb_dir ./ --ppi 3V6O_A,3V6O_C --out_dir ./path_to_output_directory 
Run a list of PPIs: 
python ../piston infer --pdb_dir ./ --list ./path_to_list --out_dir ./path_to_output_directory 

The dependent packages of MaSIF can be downloaded through this link: https://github.com/LPDI-EPFL/masif

python piston/map_label/compute_label3.py - Calculate the atomic distance between two chains to obtain the epitope label, preparing for the input of subsequent code scripts.
python piston/map_label/sema_map_atom1.py - Standardize the image size to prepare for the input of the subsequent ViT model.
python piston/get_distance_atom.py - Calculate the spatial proximity distance.
python piston/6fea-13fea2.py - Merge the six-dimensional features of the two chains with the spatial proximity distance features into a 13-dimensional image.
python piston/muti_piston_vector3.py - Extract protein structural features through the Vision Transformer model. 

### Sequence Feature Extraction
python computeesm.py —— Extract sequence features through the ESM-2 protein large language model 

### Training
Run python 3graph.py - simultaneously input protein structure features and sequence features, implement multimodal feature fusion within this script, conduct self-distillation training, and finally predict epitope sites through an equivariant graph neural network. 

### Dual-input modal training
python double_input37.py - simultaneously input the structural and sequence features of two chains, and the pre-trained LRiEpi method ultimately predicts whether the two chains bind. 
 
