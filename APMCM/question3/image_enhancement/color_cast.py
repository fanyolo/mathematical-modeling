import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


    


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
    """计算 UIQM 指标，增加错误处理"""
    try:
        # 确保输入图像在0-1范围内
        img = np.clip(img.astype(np.float32) / 255.0, 0, 1)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # 1. 计算色彩丰富度指标(UICM)
        rg = a - 128
        yb = b - 128
        rg_mean = np.mean(rg)
        yb_mean = np.mean(yb)
        rg_var = np.var(rg) + 1e-6  # 添加小常数避免除零
        yb_var = np.var(yb) + 1e-6
        uicm = -0.0268 * np.sqrt(rg_var + yb_var) + 0.1586 * np.sqrt(rg_mean**2 + yb_mean**2)
        
        # 2. 计算清晰度指标(UISM)
        dx = cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(dx**2 + dy**2) + 1e-6
        
        block_size = 8
        h, w = l.shape
        num_blocks_h = max(h // block_size, 1)
        num_blocks_w = max(w // block_size, 1)
        eme = 0
        valid_blocks = 0
        
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = mag[i*block_size:min((i+1)*block_size, h), 
                          j*block_size:min((j+1)*block_size, w)]
                if block.size > 0:
                    block_min = np.min(block)
                    block_max = np.max(block)
                    if block_min > 1e-6:
                        eme += 20 * np.log10(block_max/block_min)
                        valid_blocks += 1
        
        uism = eme / max(valid_blocks, 1)
        
        # 3. 计算对比度指标(UIConM)
        local_mean = cv2.boxFilter(l, -1, (11,11))
        local_var = cv2.boxFilter(l**2, -1, (11,11)) - local_mean**2
        local_var = np.maximum(local_var, 0)  # 确保方差非负
        uiconm = np.sum(np.sqrt(local_var + 1e-6)) / max(h * w, 1)
        
        # 最终UIQM计算
        w1, w2, w3 = 0.0282, 0.2953, 3.5753
        uiqm = w1 * uicm + w2 * uism + w3 * uiconm
        
        # 检查计算结果是否有效
        if np.isnan(uiqm) or np.isinf(uiqm):
            return 0.0
            
        return np.clip(uiqm / 100, 0, 1)  # 归一化到0-1范围
        
    except Exception as e:
        print(f"UIQM计算错误: {str(e)}")
        return 0.0  # 发生错误时返回默认值
def find_best_result(img):
    """根据UCIQE和UIQM的平均值找出最佳结果"""
    # 应用各种算法
    results = {
        'gray_world': gray_world(img.copy()),
        'white_patch': white_patch_enhanced(img.copy()),
        'gray_edge': gray_edge_enhanced(img.copy()),
        'lab_adjust': lab_adjust(img.copy())
    }
    
    # 计算每种方法的得分
    scores = {}
    for name, result in results.items():
        uciqe = calculate_uciqe(result)
        uiqm = calculate_uiqm(result)
        avg_score = (uciqe + uiqm) / 2
        scores[name] = {'image': result, 'uciqe': uciqe, 'uiqm': uiqm, 'avg_score': avg_score}
    
    # 找出得分最高的结果
    best_method = max(scores.items(), key=lambda x: x[1]['avg_score'])
    best_name = best_method[0]
    best_image = best_method[1]['image']
    
    return best_image, scores, best_name

# 主程序
if __name__ == "__main__":
    img_path = os.path.abspath(r'D:\ytb\fj1\909_img_.png')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_dir = r'D:\ytb\3.1p'
    os.makedirs(output_dir, exist_ok=True)

    # 获取最佳结果
    best_result = find_best_result(img)

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
