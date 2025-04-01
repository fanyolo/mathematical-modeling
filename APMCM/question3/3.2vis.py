import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 图像路径设置
img_path = r'D:\ytb\fj1\12348_img_.png'  # 修改为你的图片路径

# 图像读取与错误处理
try:
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("图像加载失败")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
except Exception as e:
    print(f"错误：无法读取图像文件 '{img_path}'")
    print(f"详细错误信息: {str(e)}")
    sys.exit(1)

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

# 修改apply_all_enhancements函数

def apply_all_enhancements(img):
    from mpl_toolkits.mplot3d import Axes3D
    
    # 获取增强后的图像
    result_gamma = gamma_correction(img.copy())
    result_hist = histogram_equalization(img.copy())
    result_adapt = adaptive_histogram_equalization(img.copy())
    result_log = logarithmic_transform(img.copy())
    result_enhanced = enhanced_clahe_gamma(img.copy())
    
    titles = ['Original Image', 'Gamma Correction', 'Histogram Equalization', 
             'Adaptive Histogram', 'Logarithmic Transform', 'Enhanced CLAHE+Gamma']
    images = [img, result_gamma, result_hist, result_adapt, result_log, result_enhanced]
    
    # 创建显示窗口
    plt.figure(figsize=(20, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 显示结果
    for i, (image, title) in enumerate(zip(images, titles)):
        # 原图
        plt.subplot(3, 6, i+1)
        plt.imshow(image)
        psnr = calculate_psnr(img, image)
        uciqe = calculate_uciqe(image)
        uiqm = calculate_uiqm(image)
        plt.title(f'{title}\nPSNR: {psnr:.2f}\nUCIQE: {uciqe:.3f}\nUIQM: {uiqm:.3f}', 
                 fontsize=8, pad=10)
        plt.axis('off')
        
        # 热力图
        plt.subplot(3, 6, i+7)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l = lab[:,:,0]
        im = plt.imshow(l, cmap='magma')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.title(f'{title} Luminance Heatmap', fontsize=8)
        plt.axis('off')
        
        # 3D表面图
        ax = plt.subplot(3, 6, i+13, projection='3d')
        y, x = np.mgrid[0:l.shape[0]:5, 0:l.shape[1]:5]  # 降采样以提高性能
        z = l[::5, ::5]  # 对应降采样
        surf = ax.plot_surface(x, y, z, cmap='magma', 
                             linewidth=0, antialiased=True)
        ax.view_init(elev=30, azim=45)  # 设置视角
        ax.set_title(f'{title} 3D Surface', fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('enhancement_comparison.jpg', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()

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
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    
    # 计算色度
    chroma = np.sqrt(np.square(a) + np.square(b))
    
    # 计算饱和度变异系数
    mu_c = np.mean(chroma)
    std_c = np.std(chroma)
    if (mu_c == 0):
        mu_c = 1e-6
    sat_var = std_c / mu_c
    
    # 计算亮度对比度
    top_15_percent = np.percentile(l, 85)
    bottom_15_percent = np.percentile(l, 15)
    con_l = top_15_percent - bottom_15_percent
    
    # 计算平均饱和度
    avg_sat = np.mean(chroma)
    
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    uciqe = c1 * sat_var + c2 * con_l + c3 * avg_sat
    return uciqe/100

def calculate_uiqm(img):
    """计算水下图像质量测度(UIQM)"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    
    # 计算色彩丰富度指标(UICM)
    rg = a - 128
    yb = b - 128
    rg_mean = np.mean(rg)
    yb_mean = np.mean(yb)
    rg_var = np.var(rg)
    yb_var = np.var(yb)
    uicm = -0.0268 * np.sqrt(rg_var + yb_var) + 0.1586 * np.sqrt(rg_mean**2 + yb_mean**2)
    
    # 计算清晰度指标(UISM)
    sobel_dx = cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3)
    sobel_dy = cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_dx**2 + sobel_dy**2)
    
    block_size = 8
    h, w = l.shape
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    eme = 0
    
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = mag[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            block_min = np.min(block)
            block_max = np.max(block)
            if block_min > 0:
                eme += 20 * np.log10(block_max/block_min)
    
    uism = eme / (num_blocks_h * num_blocks_w)
    
    # 计算对比度指标(UIConM)
    local_mean = cv2.boxFilter(l, -1, (11,11))
    local_var = cv2.boxFilter(l**2, -1, (11,11)) - local_mean**2
    uiconm = np.sum(np.sqrt(local_var)) / (h * w)
    
    # 最终UIQM计算
    w1, w2, w3 = 0.282, 0.2953, 0.4227
    uiqm = w1 * uicm + w2 * uism + w3 * uiconm
    return uiqm/100

# 应用增强算法并显示结果
apply_all_enhancements(img)