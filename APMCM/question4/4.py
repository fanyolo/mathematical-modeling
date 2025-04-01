import os
import cv2
import pandas as pd
import numpy as np
from color_cast import find_best_result as color_cast_enhance
from low_light import find_best_result as low_light_enhance
from deblur import find_best_result as deblur_enhance

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

def process_image_with_metrics(img_path, output_dir):
    """处理图片并计算评价指标"""
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. 色偏校正
    color_cast_result, _, _ = color_cast_enhance(img)
    
    # 2. 低光照增强
    low_light_result, _, _ = low_light_enhance(color_cast_result)
    
    # 3. 去模糊处理
    deblur_result, _, _ = deblur_enhance(low_light_result)
    
    # 计算评价指标
    psnr = calculate_psnr(img, deblur_result)
    uciqe = calculate_uciqe(deblur_result)
    uiqm = calculate_uiqm(deblur_result)
    # 确保指标不为 None
    if psnr is None:
        psnr = 0.0
    if uciqe is None:
        uciqe = 0.0
    if uiqm is None:
        uiqm = 0.0   
    # 保存结果
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    output_name = f"{name}_enhanced{ext}"
    output_path = os.path.join(output_dir, output_name)
    
    result_bgr = cv2.cvtColor(deblur_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    
    return {
        'psnr': psnr,
        'uciqe': uciqe,
        'uiqm': uiqm,
        'output_path': output_path
    }

def process_images_from_excel(excel_path, img_folder, output_dir):
    # 读取Excel文件
    try:
        df = pd.read_excel(excel_path)
        if 'image file name' not in df.columns:
            raise ValueError("Excel必须包含'image file name'列")
        # 确保必要的列存在
        required_columns = ['PSNR', 'UCIQE', 'UIQM']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Excel必须包含'{col}'列")
    except Exception as e:
        print(f"读取Excel文件出错: {str(e)}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每张图片
    for idx, row in df.iterrows():
        try:
            img_name = row['image file name']
            img_path = os.path.join(img_folder, img_name)
            
            if not os.path.exists(img_path):
                print(f"找不到图片: {img_name}")
                continue
                
            # 处理图片并获取指标
            metrics = process_image_with_metrics(img_path, output_dir)
            
            # 更新Excel中的指标值
            df.at[idx, 'PSNR'] = metrics['psnr']
            df.at[idx, 'UCIQE'] = metrics['uciqe']
            df.at[idx, 'UIQM'] = metrics['uiqm']
            
            print(f"已处理图片: {img_name}")
            print(f"PSNR: {metrics['psnr']:.4f}")
            print(f"UCIQE: {metrics['uciqe']:.4f}")
            print(f"UIQM: {metrics['uiqm']:.4f}")
            
        except Exception as e:
            print(f"处理图片 {img_name} 时出错: {str(e)}")
            continue
    
    # 保存回原Excel文件
    df.to_excel(excel_path, index=False)
    print(f"结果已保存到原Excel文件: {excel_path}")
# 主程序
if __name__ == "__main__":
    excel_path = r"D:\ytb\111.xlsx" # Excel文件路径
    img_folder = r"D:\ytb\Attachment 2"  # 图片文件夹路径
    output_dir = r"D:\ytb\4p"         # 输出目录

    process_images_from_excel(excel_path, img_folder, output_dir)