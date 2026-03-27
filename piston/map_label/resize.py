"""this code meant to resize npy to unified size"""
import numpy as np
from scipy.ndimage import zoom


def resize_array(arr, resize_scale):
    target_size = (resize_scale,resize_scale,6)#目标大小
    scale_factors = [n / o for n, o in zip(target_size[:2], arr.shape[:2])]#计算缩放比例，即 resize_scale 与原始数组大小 arr.shape[:2] 的比值。
    scale_factors.append(1)
    resized_arr = zoom(arr, scale_factors, order=1)#线性插值法进行缩放
    return resized_arr, scale_factors


def find_original_position(resized_pos, scale_factors):#根据缩放比例 scale_factors，找到缩放后的数组中一个位置 resized_pos 在原始数组中的对应位置。
    return [int(pos / scale) for pos, scale in zip(resized_pos[:2], scale_factors[:2])]


def find_corresponding_area_in_original(resized_pos, scale_factors, original_shape):
    start = [int(pos / scale) for pos, scale in zip(resized_pos[:2], scale_factors[:2])]#计算缩放后的位置 resized_pos 在原始数组中的起始位置，方法是 缩放后位置 / 缩放比例。
    end = [min(int((pos + 1) / scale), shape) for pos, scale, shape in
           zip(resized_pos[:2], scale_factors[:2], original_shape[:2])]#计算对应区域的结束位置，方法是 ((缩放后位置 + 1) / 缩放比例)，并确保不会超过原始数组的边界。
    return start, end


def get_out_index(index_arr,scale_factors,arr,resize_scale):
    out_index = []
    for i in range(resize_scale):
        row = []
        for j in range(resize_scale):
            resized_pos = (i, j)
            original_start, original_end = find_corresponding_area_in_original(resized_pos, scale_factors, arr.shape)#，计算每个缩放后位置 (i, j) 对应原始数组中的区域范围 original_start 到 original_end。
            # 映射 index 文件
            mapped_index = np.unique(index_arr[original_start[0]:original_end[0], original_start[1]:original_end[1]])#提取原始区域中的唯一索引值，并将这些索引添加到 out_index 中。
            row.append(mapped_index.tolist())

        out_index.append(row)

    out_index = np.array(out_index, dtype=object)

    return out_index
