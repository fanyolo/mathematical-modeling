import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 图像路径设置
img_path = r'D:\ytb\fj1\12348_img_.png'

# 图像读取与错误处理
try:
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("图像加载失败")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
except Exception as e:
    print(f"错误：无法读取图像文件 '{img_path}'")
    print(f"详细错误信息: {str(e)}")
    sys.exit(1)

# 1. 维纳滤波去模糊
def wiener_deblur(img, kernel_size=5, K=0.01):
    """维纳滤波去模糊"""
    img_float = img.astype(np.float32) / 255.0
    
    # 分别处理每个通道
    result = np.zeros_like(img_float)
    for i in range(3):
        channel = img_float[:,:,i]
        # 傅里叶变换
        freq = np.fft.fft2(channel)
        freq_shift = np.fft.fftshift(freq)
        
        # 估计模糊核
        h = np.zeros((kernel_size, kernel_size))
        h[kernel_size//2, kernel_size//2] = 1
        h = np.fft.fft2(h, channel.shape)
        h = np.fft.fftshift(h)
        
        # 维纳滤波
        h_conj = np.conj(h)
        deblurred = freq_shift * h_conj / (np.abs(h)**2 + K)
        
        # 逆变换
        result[:,:,i] = np.abs(np.fft.ifft2(np.fft.ifftshift(deblurred)))
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)

# 2. Richardson-Lucy去模糊
def richardson_lucy_deblur(img, iterations=30):
    """Richardson-Lucy去模糊算法"""
    img_float = img.astype(np.float32) / 255.0
    kernel = np.ones((5,5)) / 25  # 简单的均值模糊核
    result = img_float.copy()
    
    for _ in range(iterations):
        est_conv = cv2.filter2D(result, -1, kernel)
        relative_blur = img_float / (est_conv + 1e-10)
        correction = cv2.filter2D(relative_blur, -1, kernel)
        result *= correction
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)

# 3. 非局部均值去模糊
def nlm_deblur(img, h=10, templateWindowSize=7, searchWindowSize=21):
    """改进的非局部均值去模糊"""
    result = np.zeros_like(img)
    
    # 分别处理每个通道
    for i in range(3):
        # 减小h值以保留更多细节
        result[:,:,i] = cv2.fastNlMeansDenoising(
            img[:,:,i],
            None,
            h=7,  # 降低h值
            templateWindowSize=5,  # 减小模板窗口
            searchWindowSize=15  # 减小搜索窗口
        )
    
    # 添加锐化处理
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    result = cv2.filter2D(result, -1, kernel)
    
    return result

# 4. 导向滤波去模糊
def guided_filter_deblur(img, radius=8, eps=100):
    """导向滤波去模糊"""
    guided = cv2.ximgproc.guidedFilter(
        guide=img,
        src=img,
        radius=radius,
        eps=eps
    )
    # 增强边缘
    result = cv2.addWeighted(img, 1.5, guided, -0.5, 0)
    return np.clip(result, 0, 255).astype(np.uint8)

# 5. 结合型去模糊(组合多种方法)
def combined_deblur(img):
    """改进的组合去模糊方法"""
    # 1. 使用导向滤波
    guided = cv2.ximgproc.guidedFilter(
        guide=img,
        src=img,
        radius=4,  # 减小半径
        eps=50
    )
    
    # 2. 应用非局部均值，参数更温和
    nlm = nlm_deblur(guided)
    
    # 3. 提取细节层
    gaussian = cv2.GaussianBlur(nlm, (0,0), 3)
    detail = cv2.subtract(nlm, gaussian)
    
    # 4. 增强细节
    enhanced_detail = cv2.multiply(detail, 1.5)
    
    # 5. 边缘保护的锐化
    kernel_sharpen = np.array([[-0.5,-0.5,-0.5],
                              [-0.5, 5.0,-0.5],
                              [-0.5,-0.5,-0.5]])
    sharpened = cv2.filter2D(nlm, -1, kernel_sharpen)
    
    # 6. 合并结果
    result = cv2.addWeighted(sharpened, 0.7, 
                            cv2.add(nlm, enhanced_detail), 0.3, 0)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def calculate_psnr(img1, img2):
    """计算峰值信噪比(PSNR)"""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # 计算每个通道的MSE
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
    if mu_c == 0:
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
    
    # 计算色彩丰度指标(UICM)
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

def apply_all_deblur(img):
    """应用所有去模糊算法并显示结果"""

    # 获取去模糊后的图像
    result_wiener = wiener_deblur(img.copy())
    result_rl = richardson_lucy_deblur(img.copy())
    result_guided = guided_filter_deblur(img.copy())

    titles = ['Original Image', 'Wiener Filter', 'Richardson-Lucy', 'Guided Filter']
    images = [img, result_wiener, result_rl, result_guided]

    # 创建显示窗口
    plt.figure(figsize=(15, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # 显示结果
    for i, (image, title) in enumerate(zip(images, titles)):
        # 原始/处理后的图像
        plt.subplot(2, 4, i+1)
        plt.imshow(image)
        psnr = calculate_psnr(img, image)
        uciqe = calculate_uciqe(image)
        uiqm = calculate_uiqm(image)
        plt.title(f'{title}\nPSNR: {psnr:.2f}, UCIQE: {uciqe:.3f}, UIQM: {uiqm:.3f}',
                  fontsize=8, pad=10)
        plt.axis('off')

        # 拉普拉斯变换图 - 增强显示效果
        plt.subplot(2, 4, i+5)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 使用更大的kernel size增强边缘检测
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        laplacian_abs = np.abs(laplacian)
        
        # 增强对比度
        p2, p98 = np.percentile(laplacian_abs, (2, 98))
        laplacian_norm = np.clip((laplacian_abs - p2) / (p98 - p2 + 1e-8), 0, 1)
        
        # 使用热力图显示
        im = plt.imshow(laplacian_norm, cmap='magma')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.title(f'{title} Edge Detection', fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('deblur_comparison_laplacian.jpg', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()

# 应用增强算法并显示结果
apply_all_deblur(img)