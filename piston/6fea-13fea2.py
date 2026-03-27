import numpy as np
import os
# 文件路径定义
grid_folder = '/mnt/Data6/23gsy/graph-piston/32sema_resized/sema_train/resized_feature'
dist_folder='/mnt/Data6/23gsy/graph-piston/dist_grid'
input_file = '/mnt/Data6/23gsy/graph-piston/sema_train_wash.txt'
# Read PPI data from the input file
with open(input_file, 'r') as f:
    for line in f:
        ppi = line.strip()  # Strip newlines and spaces
        print(ppi)
        antigen = ppi.split(',')[0]
        antigen_pid = antigen.split('_')[0]
        antigen_ch = antigen.split('_')[1]
        antibody = ppi.split(',')[1]
        antibody_pid = antibody.split('_')[0]
        antibody_ch = antibody.split('_')[1]
        check_out_grid =f'/mnt/Data6/23gsy/graph-piston/32sema_resized/sema_train/grid/{antigen},{antibody}.npy'
        if os.path.exists(check_out_grid):
            print("Image is already computed {}. Skipping".format(ppi))
            continue
        # 加载数据
        file_antigen = f'/resized_{antigen}_feature.npy'
        file_antibody = f'/resized_{antibody}_feature.npy'
        file_dist = f'/{antigen},{antibody}_dist_grids.npy'
        data_antigen = np.load(grid_folder + file_antigen)
        data_antibody = np.load(grid_folder + file_antibody)
        data_dist = np.load(dist_folder + file_dist)
        data_dist=np.expand_dims(data_dist,axis=-1)
        # 确保数据的维度正确
        assert data_antigen.shape == (32, 32, 6), f"Expected shape for 1a2y_C.npy: (32, 32, 6), got {data_antigen.shape}"
        assert data_antibody.shape == (32, 32, 6), f"Expected shape for 1a2y_B.npy: (32, 32, 6), got {data_antibody.shape}"


        # 创建一个新的空数组 (32, 32, 13) 用于存储最终结果
        result = np.zeros((32, 32, 13))

        # 获取1a2y_C的前5维特征并写入result的前5维
        result[:, :, 0:5] = data_antigen[:, :, 0:5]

        # 获取1a2y_B的前5维特征并写入result的6到10维
        result[:, :, 5:10] = data_antibody[:, :, 0:5]

        # 将1a2y_C_1a2y_B_dist文件的值调整为(32, 32, 1)并写入result的第11维
        result[:, :, 10:11] = data_dist

        # 获取1a2y_C的第6维特征并写入result的第12维
        result[:, :, 11] = data_antigen[:, :, 5]

        # 获取1a2y_B的第6维特征并写入result的第13维
        result[:, :, 12] = data_antibody[:, :, 5]

        # 保存最终结果到新的npy文件
        output_file = f'/mnt/Data6/23gsy/graph-piston/32sema_resized/sema_train/grid/{antigen},{antibody}.npy'
        np.save(output_file, result)
        print(result.shape)

# # 假设你的文件名是 'your_file.npy'
# file_path = './grid_16R/1a2y_C_1a2y_B.npy'
#
# # 加载 .npy 文件
# data = np.load(file_path)
# # 打印前三行的详细数据
# print("前三行的数据：")
# for i in range(3):
#     print(f"第 {i+1} 行数据：\n{data[i]}")
# # # 打印前 3 行数据
# # print("前三行的数据：")
# # # print(data_1a2y_C[:3])
# # # print(data_1a2y_B[:3])
# # # print(data_1a2y_C_B_dist[:3])
# # print(data[:3])
# # print(f"文件 {output_file} 已成功保存，形状为 {result.shape}")
