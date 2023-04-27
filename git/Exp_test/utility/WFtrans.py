import os
import tifffile as tif
from PIL import Image
import numpy as np

folder_name = 'Testbench/Fairsim/div2k_test_trans'  # 修改为你的文件夹名字
result_folder_name = 'Testbench/Fairsim/div2k_wf'  # 修改为保存结果的文件夹名字

# 创建保存结果的文件夹
if not os.path.exists(result_folder_name):
    os.makedirs(result_folder_name)

for file_name in os.listdir(folder_name):
    if file_name.endswith('.tif'):
        file_path = os.path.join(folder_name, file_name)
        result_file_path = os.path.join(result_folder_name, os.path.splitext(file_name)[0] + '.png')
        with tif.TiffFile(file_path) as tif_file:
            images = tif_file.asarray()
            avg_image = np.mean(images[:9], axis=0)  # 取前9张图片的平均值
        image = Image.fromarray(avg_image.astype('uint8'))  # 转为PIL image
        image.save(result_file_path)  # 保存为PNG格式
