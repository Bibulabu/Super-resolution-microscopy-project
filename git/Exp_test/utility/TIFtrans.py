import os
import tifffile as tif

folder_name = 'Test_data/microtubules_simulated'  # 修改为你的文件夹名字
result_folder_name = 'Test_data/microtubules_sim_9'  # 修改为保存结果的文件夹名字

# 创建保存结果的文件夹
if not os.path.exists(result_folder_name):
    os.makedirs(result_folder_name)

for file_name in os.listdir(folder_name):
    if file_name.endswith('.tif'):
        file_path = os.path.join(folder_name, file_name)
        result_file_path = os.path.join(result_folder_name, file_name)
        with tif.TiffFile(file_path) as tif_file:
            images = tif_file.asarray()
            images = images[:9]  # 取前9张图片
        tif.imwrite(result_file_path, images)