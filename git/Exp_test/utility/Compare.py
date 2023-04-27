# plot one fairsim vs gt
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# SR: Testbench\Fairsim\fairsim_out
# GT: Testbench\Fairsim\div2k_GT


# 接收两个图像的路径
fair_path = "Testbench\microtube_real\microtubules_wf\wf_16.jpg"
gt_path = "Testbench\microtube_real\microtubules_gt/16.png"
#sr_path = 'Testbench/Fairsim/div2k_GT/0701_0.png'

# 读取图像
fair_img = cv2.imread(fair_path)
gt_img = cv2.imread(gt_path)

# 图像处理
if fair_img is None:
    print("无法读取Super Resolution图像，请检查路径是否正确")
    exit()
if gt_img is None:
    print("无法读取Ground Truth图像，请检查路径是否正确")
    exit()

# resize
if fair_img.shape[:2] != (512, 512):
    fair_img = cv2.resize(fair_img, (512, 512))

# crop
x, y, w, h = 8, 8, 492, 492  # define the crop rectangle
fair_img = fair_img[y:y+h, x:x+w]  # crop the image using array slicing
gt_img = gt_img[y:y+h, x:x+w]  # crop the image using array slicing


# 计算 PSNR
fair_img,gt_img = np.array(fair_img)/255.0, np.array(gt_img)/255.0

MSE = np.mean( (fair_img-gt_img)**2 )
psnr = 20*np.log10(1/np.sqrt(MSE))

print("PSNR = ", psnr)

# 计算 SSIM
ssim_value = ssim(fair_img, gt_img, multichannel=True,win_size=3, data_range=1)
print("SSIM = ", ssim_value)

# 将两个图像绘制在一起
merged_img = cv2.hconcat([fair_img, gt_img])

# 在图像上标注 PSNR 和 SSIM
cv2.putText(merged_img, "PSNR: {:.2f}".format(psnr), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(merged_img, "SSIM: {:.2f}".format(ssim_value), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# 显示合并后的图像
cv2.imshow("Fairsim vs Ground Truth", merged_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



""" # loop over testbench for average psnr & ssim
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

# SR: Testbench\Fairsim\fairsim_out
# GT: Testbench\Fairsim\div2k_GT

fair_folder = "Testbench/Fairsim/fairsim_out/"
gt_folder = "Testbench/Fairsim/div2k_GT/"

psnr_total = 0
ssim_total = 0

num_img = 31

for i in range(1, num_img):
    # Generate file paths for the current pair of images
    fair_path = os.path.join(fair_folder, str(i) + ".png")
    gt_path = os.path.join(gt_folder, "{:04d}_0.png".format(700+i))

    # Read in the images
    fair_img = cv2.imread(fair_path)
    gt_img = cv2.imread(gt_path)

    # Check that both images were read in successfully
    if fair_img is None:
        print("无法读取Super Resolution图像，请检查路径是否正确")
        exit()
    if gt_img is None:
        print("无法读取Ground Truth图像，请检查路径是否正确")
        exit()

    # Resize the images if they are not already 512x512
    if fair_img.shape[:2] != (512, 512):
        fair_img = cv2.resize(fair_img, (512, 512))

    # Crop the images
    x, y, w, h = 8, 8, 492, 492  # define the crop rectangle
    fair_img = fair_img[y:y+h, x:x+w]  # crop the image using array slicing
    gt_img = gt_img[y:y+h, x:x+w]  # crop the image using array slicing

    # Calculate PSNR and SSIM
    fair_img, gt_img = np.array(fair_img)/255.0, np.array(gt_img)/255.0
    MSE = np.mean( (fair_img-gt_img)**2 )
    psnr = 20*np.log10(1/np.sqrt(MSE))
    ssim_value = ssim(fair_img, gt_img, multichannel=True, win_size=3, data_range=1)

    # Add PSNR and SSIM to totals
    psnr_total += psnr
    ssim_total += ssim_value

    # Print the PSNR and SSIM for the current pair of images
    print("Pair {:2d}: PSNR = {:.2f}, SSIM = {:.4f}".format(i, psnr, ssim_value))

# Calculate and print the average PSNR and SSIM over all pairs of images
avg_psnr = psnr_total / 30
avg_ssim = ssim_total / 30
print("Average PSNR = {:.2f}".format(avg_psnr))
print("Average SSIM = {:.4f}".format(avg_ssim)) """