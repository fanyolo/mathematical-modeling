import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


    

# 1. 改进伽马变换
def gamma_correction(img, gamma=0.8):
    """改进的伽马变换增强亮度"""
    # 转换到LAB空间
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # 对L通道进行伽马校正
    l_float = l.astype(np.float32) / 255.0
    l_gamma = np.power(l_float, gamma) * 255.0
    
    # 添加细节增强
    kernel_sharpen = np.array([[-0.3,-0.3,-0.3],
                              [-0.3, 3.4,-0.3],
                              [-0.3,-0.3,-0.3]])
    l_enhanced = cv2.filter2D(l_gamma, -1, kernel_sharpen)
    
    # 合并通道
    result = cv2.merge([np.uint8(l_enhanced), a, b])
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

# 2. 直方图均衡化
def histogram_equalization(img):
    """直方图均衡化增强"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    result = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

# 3. 改进自适应直方图均衡化
def adaptive_histogram_equalization(img):
    """改进的CLAHE自适应直方图均衡化"""
    # 转换到LAB空间
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # 创建CLAHE对象并调整参数
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    
    # 添加边缘保护的细节增强
    bilateral = cv2.bilateralFilter(l_clahe, d=5, sigmaColor=50, sigmaSpace=50)
    detail_layer = cv2.subtract(l_clahe, bilateral)
    enhanced_l = cv2.add(l_clahe, detail_layer)
    
    # 合并通道
    result = cv2.merge([enhanced_l, a, b])
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

# 4. 对数变换增强
def logarithmic_transform(img):
    """对数变换增强"""
    c = 255 / (np.log(1 + np.max(img)))
    result = c * np.log(1 + img.astype(np.float32))
    return np.uint8(result)

# 5. 改进CLAHE与伽马结合
def enhanced_clahe_gamma(img, clip_limit=3.0, gamma=0.85):
    """改进的CLAHE与伽马结合增强"""
    # 转换到LAB空间
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE处理
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    
    # 伽马校正
    l_float = l_clahe.astype(np.float32) / 255.0
    l_gamma = np.power(l_float, gamma) * 255.0
    
    # 局部对比度增强
    kernel_local = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    l_enhanced = cv2.filter2D(l_gamma, -1, kernel_local)
    
    # 细节层提取和增强
    gaussian = cv2.GaussianBlur(l_enhanced, (0,0), 3)
    detail = cv2.subtract(l_enhanced, gaussian)
    l_final = cv2.add(l_enhanced, detail)
    
    # 确保值在合理范围内
    l_final = np.clip(l_final, 0, 255).astype(np.uint8)
    
    # 合并通道
    result = cv2.merge([l_final, a, b])
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)


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
    # 确保输入图像在0-1范围内
    img = np.clip(img.astype(np.float32) / 255.0, 0, 1)
    
    # 转换到LAB空间
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # 1. 计算色彩丰富度指标(UICM)
    rg = a - 128
    yb = b - 128
    rg_mean = np.mean(rg)
    yb_mean = np.mean(yb)
    rg_var = np.var(rg) + 1e-6  # 添加小的常数避免除零
    yb_var = np.var(yb) + 1e-6
    uicm = -0.0268 * np.sqrt(rg_var + yb_var) + 0.1586 * np.sqrt(rg_mean**2 + yb_mean**2)
    
    # 2. 计算清晰度指标(UISM)
    sobel_kernel_size = 3
    dx = cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=sobel_kernel_size)
    dy = cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=sobel_kernel_size)
    mag = np.sqrt(dx**2 + dy**2)
    
    block_size = 8
    h, w = l.shape
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    eme = 0
    valid_blocks = 0
    
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = mag[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            if block.size > 0:
                block_min = np.min(block)
                block_max = np.max(block)
                if block_min > 1e-6:  # 避免取对数时除零
                    eme += 20 * np.log10(block_max/block_min)
                    valid_blocks += 1
    
    uism = eme / valid_blocks if valid_blocks > 0 else 0
    
    # 3. 计算对比度指标(UIConM)
    local_mean = cv2.boxFilter(l, -1, (11,11))
    local_var = cv2.boxFilter(l**2, -1, (11,11)) - local_mean**2
    local_var = np.maximum(local_var, 0)  # 确保方差非负
    uiconm = np.sum(np.sqrt(local_var)) / (h * w + 1e-6)  # 避免除零
    
    # 最终UIQM计算
    w1, w2, w3 = 0.0282, 0.2953, 3.5753
    uiqm = w1 * uicm + w2 * uism + w3 * uiconm
    
    # 确保返回有效的数值
    if np.isnan(uiqm) or np.isinf(uiqm):
        return 0.0
        
    return np.clip(uiqm / 100, 0, 1)  # 归一化到0-1范围

def find_best_result(img):
    """根据UCIQE和UIQM的平均值找出最佳结果"""
    # 应用各种算法
    results = {
        'gamma_correction': gamma_correction(img.copy()),
        'histogram_equalization': histogram_equalization(img.copy()),
        'adaptive_histogram_equalization': adaptive_histogram_equalization(img.copy()),
        'logarithmic_transform': logarithmic_transform(img.copy()),
        'enhanced_clahe_gamma': enhanced_clahe_gamma(img.copy())
    }
    
    # 计算每种方法的得分
    scores = {}
    for name, result in results.items():
        uciqe = calculate_uciqe(result)
        uiqm = calculate_uiqm(result)
        avg_score = (uciqe*0.3 + uiqm*0.7) / 2
        scores[name] = {'image': result, 'uciqe': uciqe, 'uiqm': uiqm, 'avg_score': avg_score}
    
    # 找出得分最高的结果
    best_method = max(scores.items(), key=lambda x: x[1]['avg_score'])
    best_image = best_method[1]['image']
    best_name = best_method[0]
    return best_image, scores, best_name

# 主程序
if __name__ == "__main__":
    img_path = os.path.abspath(r'D:\ytb\fj1\12348_img_.png')  # 输入图像路径
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_dir = r'D:\ytb\3.1p'  # 输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取最佳结果
    best_result, scores, best_method = find_best_result(img)

    # 输出每种算法的评分
    for method, score in scores.items():
        print(f"算法: {method}")
        print(f"UCIQE: {score['uciqe']:.4f}")
        print(f"UIQM: {score['uiqm']:.4f}")
        print(f"平均值: {score['avg_score']:.4f}")
        print()

    print(f"选择的最佳算法: {best_method}")

    # 构建输出路径
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    output_name = f"{name}1{ext}"
    output_path = os.path.join(output_dir, output_name)

    # 保存结果
    best_result_bgr = cv2.cvtColor(best_result, cv2.COLOR_RGB2BGR)
    success, encoded_image = cv2.imencode(ext, best_result_bgr)
    if success:
        encoded_image.tofile(output_path)