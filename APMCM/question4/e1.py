import cv2
import numpy as np

def calculate_psnr(img1, img2):
    """计算峰值信噪比(PSNR)"""
    # 确保图像类型一致且为float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # 分别计算每个通道的MSE
    mse_r = np.mean((img1[:,:,0] - img2[:,:,0]) ** 2)
    mse_g = np.mean((img1[:,:,1] - img2[:,:,1]) ** 2)
    mse_b = np.mean((img1[:,:,2] - img2[:,:,2]) ** 2)
    
    # 计算总体MSE
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
    """计算 UIQM 指标"""
    img = img.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # 1. 计算色彩丰富度指标(UICM)
    rg = a - 128
    yb = b - 128
    rg_mean = np.mean(rg)
    yb_mean = np.mean(yb)
    rg_var = np.var(rg)
    yb_var = np.var(yb)
    uicm = -0.0268 * np.sqrt(rg_var + yb_var) + 0.1586 * np.sqrt(rg_mean**2 + yb_mean**2)
    
    # 2. 计算清晰度指标(UISM)
    sobel_kernel_size = 3
    dx = cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=sobel_kernel_size)
    dy = cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=sobel_kernel_size)
    mag = np.sqrt(dx**2 + dy**2)
    
    # 使用EME (Enhancement Measure Estimation)
    block_size = 8
    h, w = l.shape
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    eme = 0
    
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # 修正索引错误，将(i+1)改为j+1
            block = mag[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            # 添加非空检查
            if block.size > 0:
                block_min = np.min(block)
                block_max = np.max(block)
                if block_min > 0:
                    eme += 20 * np.log10(block_max/block_min)
    
    uism = eme / (num_blocks_h * num_blocks_w) if (num_blocks_h * num_blocks_w) > 0 else 0
    
    # 3. 计算对比度指标(UIConM)
    local_mean = cv2.boxFilter(l, -1, (11,11))
    local_var = cv2.boxFilter(l**2, -1, (11,11)) - local_mean**2
    uiconm = np.sum(np.sqrt(local_var)) / (h * w)
    
    # 最终UIQM计算
    w1, w2, w3 = 0.282, 0.2953, 0.4227
    uiqm = w1 * uicm + w2 * uism + w3 * uiconm
    return uiqm/100