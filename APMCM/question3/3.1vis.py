import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 使用原始字符串表示法(r'')来避免转义字符问题
img_path = r'D:\ytb\fj1\160_img_.png'  # 使用 r 前缀

try:
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("图像加载失败")
except Exception as e:
    print(f"错误：无法读取图像文件 '{img_path}'")
    print(f"详细错误信息: {str(e)}")
    sys.exit(1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV读取的是BGR格式，转换为RGB格式

# 方法一：灰度世界假设（Gray World Algorithm）
def gray_world(img):
    # 转换为浮点型
    img_float = img.astype(np.float32)

    # 计算每个通道的均值
    b_avg, g_avg, r_avg = cv2.mean(img_float)[:3]
    avg_gray = (b_avg + g_avg + r_avg) / 3

    # 计算增益系数
    scale_b = avg_gray / b_avg
    scale_g = avg_gray / g_avg
    scale_r = avg_gray / r_avg

    # 限制增益系数范围，防止过度增强
    max_scale = 1.5
    min_scale = 0.5
    scale_b = np.clip(scale_b, min_scale, max_scale)
    scale_g = np.clip(scale_g, min_scale, max_scale)
    scale_r = np.clip(scale_r, min_scale, max_scale)

    # 应用增益并合并通道
    b, g, r = cv2.split(img_float)
    b = b * scale_b
    g = g * scale_g
    r = r * scale_r
    balanced = cv2.merge([b, g, r])

    # 应用轻微的高斯模糊，平滑像素值分布
    balanced = cv2.GaussianBlur(balanced, (3, 3), 0)

    # 确保数值范围并转换回 uint8
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)

    return balanced

# 方法二：白斑算法（White Patch Algorithm）
def white_patch(img):
    # 转换为浮点型
    img_float = img.astype(np.float32)
    
    # 1. 预处理：双边滤波，减少噪声同时保留边缘
    img_filtered = cv2.bilateralFilter(img_float, d=5, sigmaColor=75, sigmaSpace=75)
    
    # 2. 分离通道
    b, g, r = cv2.split(img_filtered)
    
    # 3. 计算每个通道的高百分位数（如99.5%），以避免异常值影响
    percentile = 99.5
    max_b = np.percentile(b, percentile)
    max_g = np.percentile(g, percentile)
    max_r = np.percentile(r, percentile)
    
    # 4. 计算缩放因子，不施加严格的限制
    max_rgb = max(max_r, max_g, max_b)
    scale_b = max_rgb / max_b if max_b > 0 else 1
    scale_g = max_rgb / max_g if max_g > 0 else 1
    scale_r = max_rgb / max_r if max_r > 0 else 1
    
    # 5. 应用缩放
    b = b * scale_b
    g = g * scale_g
    r = r * scale_r
    
    # 6. 限制像素值范围
    b = np.clip(b, 0, 255)
    g = np.clip(g, 0, 255)
    r = np.clip(r, 0, 255)
    
    # 7. 合并通道
    balanced = cv2.merge([b, g, r])
    
    # 8. 应用轻微的高斯模糊，减少可能的伪影
    balanced = cv2.GaussianBlur(balanced, (3, 3), 0)
    
    # 9. 转换为 uint8 类型
    balanced = balanced.astype(np.uint8)
    
    return balanced

def white_patch_enhanced(img):
    # 转换为浮点型
    img_float = img.astype(np.float32)
    
    # 1. 轻度双边滤波预处理
    img_filtered = cv2.bilateralFilter(img_float, d=5, sigmaColor=50, sigmaSpace=50)
    
    # 2. 分离通道
    b, g, r = cv2.split(img_filtered)
    
    # 3. 使用自适应阈值选择参考白点
    percentile = 99.9  # 提高percentile以更好地识别白点
    max_b = np.percentile(b, percentile)
    max_g = np.percentile(g, percentile)
    max_r = np.percentile(r, percentile)
    
    # 4. 计算自适应缩放因子
    max_rgb = max(max_r, max_g, max_b)
    scale_b = max_rgb / max_b if max_b > 0 else 1
    scale_g = max_rgb / max_g if max_g > 0 else 1
    scale_r = max_rgb / max_r if max_r > 0 else 1
    
    # 5. 应用非线性映射，保护暗部细节
    gamma = 0.85
    b = np.power(b / 255.0, gamma) * 255.0 * scale_b
    g = np.power(g / 255.0, gamma) * 255.0 * scale_g
    r = np.power(r / 255.0, gamma) * 255.0 * scale_r
    
    # 6. 限制范围
    b = np.clip(b, 0, 255)
    g = np.clip(g, 0, 255)
    r = np.clip(r, 0, 255)
    
    # 7. 合并通道
    balanced = cv2.merge([b, g, r])
    
    # 8. 边缘感知的细节增强
    kernel_sharpen = np.array([[-0.3,-0.3,-0.3],
                              [-0.3, 3.4,-0.3],
                              [-0.3,-0.3,-0.3]])
    balanced = cv2.filter2D(balanced, -1, kernel_sharpen)
    
    # 9. 最终范围限制和类型转换
    result = np.clip(balanced, 0, 255).astype(np.uint8)
    
    return result

# 方法三：灰度边缘算法（Gray Edge Algorithm）
def gray_edge(img):
    img = img.astype(np.float32)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    R_grad = cv2.Laplacian(R, cv2.CV_32F)
    G_grad = cv2.Laplacian(G, cv2.CV_32F)
    B_grad = cv2.Laplacian(B, cv2.CV_32F)
    avgR = np.mean(np.abs(R_grad))
    avgG = np.mean(np.abs(G_grad))
    avgB = np.mean(np.abs(B_grad))
    avgGray = (avgR + avgG + avgB) / 3
    scaleR = avgGray / avgR
    scaleG = avgGray / avgG
    scaleB = avgGray / avgB
    R = R * scaleR
    G = G * scaleG
    B = B * scaleB
    result = cv2.merge([R, G, B])
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def gray_edge_enhanced(img, sigma=1):
    # 转换为浮点型
    img_float = img.astype(np.float32)
    
    # 1. 双边滤波预处理
    img_smooth = cv2.bilateralFilter(img_float, 9, 75, 75)
    
    # 2. 分离通道
    b, g, r = cv2.split(img_smooth)
    
    # 3. 使用Scharr算子替代Sobel，减少锯齿
    grad_b_x = cv2.Scharr(b, cv2.CV_32F, 1, 0)
    grad_b_y = cv2.Scharr(b, cv2.CV_32F, 0, 1)
    grad_g_x = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    grad_g_y = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    grad_r_x = cv2.Scharr(r, cv2.CV_32F, 1, 0)
    grad_r_y = cv2.Scharr(r, cv2.CV_32F, 0, 1)
    
    # 4. 计算梯度幅值并平滑
    grad_b = cv2.GaussianBlur(np.sqrt(grad_b_x**2 + grad_b_y**2), (3,3), 0)
    grad_g = cv2.GaussianBlur(np.sqrt(grad_g_x**2 + grad_g_y**2), (3,3), 0)
    grad_r = cv2.GaussianBlur(np.sqrt(grad_r_x**2 + grad_r_y**2), (3,3), 0)
    
    # 5. 计算阈值掩码
    grad_mag = (grad_b + grad_g + grad_r) / 3
    threshold = np.percentile(grad_mag, 90)
    mask = grad_mag > threshold
    
    # 6. 计算均值并应用平滑的缩放
    avg_b = np.mean(np.abs(b[mask]))
    avg_g = np.mean(np.abs(g[mask]))
    avg_r = np.mean(np.abs(r[mask]))
    
    avg_rgb = (avg_r + avg_g + avg_b) / 3
    scale_b = avg_rgb / avg_b if avg_b > 0 else 1
    scale_g = avg_rgb / avg_g if avg_g > 0 else 1
    scale_r = avg_rgb / avg_r if avg_r > 0 else 1
    
    # 7. 应用缩放和最终平滑
    b = cv2.bilateralFilter(np.clip(b * scale_b, 0, 255), 5, 50, 50)
    g = cv2.bilateralFilter(np.clip(g * scale_g, 0, 255), 5, 50, 50)
    r = cv2.bilateralFilter(np.clip(r * scale_r, 0, 255), 5, 50, 50)
    
    balanced = cv2.merge([b, g, r])
    return balanced.astype(np.uint8)

# 方法四：LAB颜色空间调整
def lab_adjust(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    avgA = np.mean(A)
    avgB = np.mean(B)
    A = A - (avgA - 128)
    B = B - (avgB - 128)
    # 将 A 和 B 的像素值限制在 0-255，并转换为 uint8 类型
    A = np.clip(A, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)
    # 确保 L、A、B 三个通道的数据类型和尺寸一致
    lab = cv2.merge([L, A, B])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# 在应用算法之前添加这个函数定义
def plot_rgb_histogram(img):
    colors = ('r', 'g', 'b')
    labels = ('Red', 'Green', 'Blue')
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        # 计算直方图
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])

        
        # 绘制带填充的曲线
        plt.fill_between(range(256), hist.ravel(), 
                        alpha=0.3,  # 填充透明度
                        color=color, 
                        label=label)
        
        # 绘制曲线轮廓
        plt.plot(hist, 
                color=color, 
                alpha=0.8,  # 线条透明度
                linewidth=1)  # 线条宽度
    
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.1)
    plt.legend()  # 添加图例
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Pixel Count')

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
    """计算水下图像质量测度(UIQM)"""
    # 转换到LAB色彩空间
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    
    # 1. 计算色彩丰富度指标(UICM)
    rg = a - 128  # 红绿对立通道
    yb = b - 128  # 黄蓝对立通道
    
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
            block = mag[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            block_min = np.min(block)
            block_max = np.max(block)
            if block_min > 0:
                eme += 20 * np.log10(block_max/block_min)
    
    uism = eme / (num_blocks_h * num_blocks_w)
    
    # 3. 计算对比度指标(UIConM)
    local_mean = cv2.boxFilter(l, -1, (11,11))
    local_var = cv2.boxFilter(l**2, -1, (11,11)) - local_mean**2
    uiconm = np.sum(np.sqrt(local_var)) / (h * w)
    
    # 最终UIQM计算
    w1, w2, w3 = 0.282, 0.2953, 0.4227  # 权重参数
    uiqm = w1 * uicm + w2 * uism + w3 * uiconm
    
    # 归一化到0-1范围
    return uiqm/100

def apply_all_enhancements(img):
    """应用所有增强算法并显示结果"""
    # 获取增强后的图像
    result_gray_world = gray_world(img.copy())
    result_white_patch = white_patch_enhanced(img.copy())
    result_gray_edge = gray_edge_enhanced(img.copy())
    result_lab_adjust = lab_adjust(img.copy())
    
    titles = ['Original Image', 'Gray World', 'White Patch', 'Gray Edge', 'LAB Adjust']
    images = [img, result_gray_world, result_white_patch, result_gray_edge, result_lab_adjust]
    
    # 创建显示窗口
    plt.figure(figsize=(20, 8))
    plt.subplots_adjust(hspace=0.3)
    
    # 显示结果
    for i, (image, title) in enumerate(zip(images, titles)):
        # 原始/处理后的图像
        plt.subplot(2, 5, i+1)
        plt.imshow(image)
        
        # 计算评价指标
        psnr = calculate_psnr(img, image)
        uciqe = calculate_uciqe(image)
        uiqm = calculate_uiqm(image)
        
        plt.title(f'{title}\nPSNR: {psnr:.2f}\nUCIQE: {uciqe:.3f}\nUIQM: {uiqm:.3f}', 
                 fontsize=8, pad=15)
        plt.axis('off')
        
        # 灰度图
        plt.subplot(2, 5, i+6)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        plt.imshow(gray, cmap='gray')
        plt.title(f'{title} Grayscale', fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('enhancement_comparison.jpg', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()

# 应用各个算法
result_gray_world = gray_world(img.copy())
result_white_patch = white_patch_enhanced(img.copy())  # 使用增强版white patch
result_gray_edge = gray_edge_enhanced(img.copy())      # 使用增强版gray edge
result_lab_adjust = lab_adjust(img.copy())

# 定义标题和图像列表
titles = ['Original Image', 'Gray World', ' White Patch', ' Gray Edge', 'LAB Adjust']
images = [img, result_gray_world, result_white_patch, result_gray_edge, result_lab_adjust]

# 创建对比图和直方图
plt.figure(figsize=(20, 12))

# 显示图像和指标
for i in range(5):
    # 图像显示在上方
    plt.subplot(3, 5, i+1)
    plt.imshow(images[i])
    
    # 计算指标
    psnr = calculate_psnr(img, images[i])
    uciqe = calculate_uciqe(images[i])
    uiqm = calculate_uiqm(images[i])
    
    plt.title(f'{titles[i]}\nPSNR: {psnr:.2f}\nUCIQE: {uciqe:.3f}\nUIQM: {uiqm:.3f}', fontsize=8)
    plt.axis('off')
    
    # RGB直方图显示在中间
    plt.subplot(3, 5, i+6)
    # 使用更柔和的RGB配色
    colors = ['#FF9999', '#99CC99', '#9999FF']  # 柔和的红、绿、蓝
    x = np.arange(256)
    for j, color in enumerate(colors):
        hist = cv2.calcHist([images[i]], [j], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.8)
        # 添加填充
        plt.fill_between(x, hist.flatten(), alpha=0.3, color=color)
    plt.title(f'{titles[i]} RGB Histogram', fontsize=8)
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.2)
    
    # LAB直方图显示在下方
    plt.subplot(3, 5, i+11)
    lab = cv2.cvtColor(images[i], cv2.COLOR_RGB2LAB)
    colors = ['#2E4057', '#C17767', '#7C6A7A']  # 更优雅的配色
    labels = ['L', 'a', 'b']
    x = np.arange(256)
    for j, (color, label) in enumerate(zip(colors, labels)):
        hist = cv2.calcHist([lab], [j], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.8, label=label)
        # 添加填充
        plt.fill_between(x, hist.flatten(), alpha=0.3, color=color)
    plt.title(f'{titles[i]} LAB Histogram', fontsize=8)
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=6)

plt.tight_layout()
plt.savefig('comparison_result.jpg', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()
