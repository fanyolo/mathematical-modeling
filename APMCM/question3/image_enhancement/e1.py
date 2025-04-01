import cv2
import numpy as np

def calculate_psnr(img1, img2):
    """计算峰值信噪比(PSNR)"""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    mse_r = np.mean((img1[:,:,0] - img2[:,:,0]) ** 2)
    mse_g = np.mean((img1[:,:,1] - img2[:,:,1]) ** 2)
    mse_b = np.mean((img1[:,:,2] - img2[:,:,2]) ** 2)
    
    mse = (mse_r + mse_g + mse_b) / 3.0
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_uciqe(img):
    """计算水下图像质量评价(UCIQE)"""
    # 转换到LAB色彩空间
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    
    # 1. 计算色度
    chroma = np.sqrt(np.square(a) + np.square(b))
    
    # 2. 计算饱和度变异系数
    mu_c = np.mean(chroma)
    std_c = np.std(chroma)
    if (mu_c == 0):
        mu_c = 1e-6
    sat_var = std_c / mu_c
    
    # 3. 计算亮度对比度
    top_15_percent = np.percentile(l, 85)
    bottom_15_percent = np.percentile(l, 15)
    con_l = top_15_percent - bottom_15_percent
    
    # 4. 计算平均饱和度
    avg_sat = np.mean(chroma)
    
    # UCIQE参数
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    
    uciqe = c1 * sat_var + c2 * con_l + c3 * avg_sat
    return uciqe/100  # 归一化到0-1范围

def calculate_uiqm(img):
    """计算水下图像质量评价指标 UIQM"""
    # 确保图像为 RGB 格式
    if (img.shape[2] == 3):
        img_rgb = img
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 转换到 LAB 色彩空间
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    A = A.astype(np.float32) - 128
    B = B.astype(np.float32) - 128

    # 计算 UICM
    RG_var = np.var(A)
    YB_var = np.var(B)
    RG_mean = np.mean(A)
    YB_mean = np.mean(B)
    UICM = -0.0268 * np.sqrt(RG_var + YB_var) + 0.1586 * np.sqrt(RG_mean**2 + YB_mean**2)

    # 计算 UISM
    L = L.astype(np.float32)
    dx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(dx**2 + dy**2) + 1e-6  # 防止除零
    block_size = 8
    h, w = L.shape
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    EME = 0
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = gradient_magnitude[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            G_max = np.max(block)
            G_min = np.min(block) + 1e-6  # 防止除零
            EME += 20 * np.log10(G_max / G_min)
    UISM = EME / (num_blocks_h * num_blocks_w)

    # 计算 UIConM
    kernel_size = (17, 17)
    mean_local = cv2.blur(L, kernel_size)
    mean_sq_local = cv2.blur(L**2, kernel_size)
    contrast_local = np.sqrt(mean_sq_local - mean_local**2)
    UIConM = np.mean(contrast_local)

    # 计算 UIQM
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    UIQM = c1 * UICM + c2 * UISM + c3 * UIConM

    return UIQM/100