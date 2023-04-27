import os
import tifffile as tif
from PIL import Image

folder_name = 'Testbench/div2k_test'  # 修改为你的文件夹名字
result_folder_name = 'Testbench\Fairsim\div2k_GT'  # 修改为保存结果的文件夹名字

# 创建保存结果的文件夹
if not os.path.exists(result_folder_name):
    os.makedirs(result_folder_name)

for file_name in os.listdir(folder_name):
    if file_name.endswith('.tif'):
        file_path = os.path.join(folder_name, file_name)
        result_file_path = os.path.join(result_folder_name, os.path.splitext(file_name)[0] + '.png')
        with tif.TiffFile(file_path) as tif_file:
            images = tif_file.asarray()
            last_image = images[-1]  # 取最后一张图片
        image = Image.fromarray(last_image)
        image.save(result_file_path)
