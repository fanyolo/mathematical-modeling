import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# 使用 Pillow 读取图片并转换为 OpenCV 格式
def read_image_with_pillow(image_path):
    try:
        img = Image.open(image_path)
        img_cv = np.array(img)
        if img.mode == "RGB":
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        return img_cv
    except Exception as e:
        print(f"无法读取图片: {image_path}, 错误: {e}")
        return None

# 图像退化分析函数
def analyze_image(image_path):
    img = read_image_with_pillow(image_path)  # 使用 Pillow 读取图片
    if img is None:
        print(f"无法读取图像: {image_path}")
        return 0, 0, 0, 0, 0, 0, 0
    # RGB 分析
    image = Image.open(image_path)
    image = image.convert("RGB")  # 确保图像是 RGB 模式

    # 提取 RGB
    rgb_data = np.array(image)
    r, g, b = rgb_data[:, :, 0], rgb_data[:, :, 1], rgb_data[:, :, 2]

    # 计算每个颜色通道的平均值
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)

    # 计算颜色通道间的最大差异
    rgb_diff = max(abs(mean_r - mean_g), abs(mean_g - mean_b), abs(mean_b - mean_r))

    # HSV 分析
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv_image[:, :, 0]
    mu_H = np.mean(hue)
    sigma_H = np.std(hue)  # 使用标准差来衡量色调的分散度

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算图像的平均亮度和方差
    mean_brightness = np.mean(gray)
    std_dev_brightness = np.std(gray)

    # 计算图像的拉普拉斯变换并计算方差（模糊程度）
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    # 计算图像的傅里叶变换
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)  # 将零频移至中心

    # 计算频谱的幅度
    magnitude = np.abs(fshift)

    # 获取高频区域（图像的右上角部分）
    rows, cols = magnitude.shape
    center_row, center_col = rows // 2, cols // 2
    high_freq = magnitude[:center_row, center_col:]

    # 计算高频区域的能量
    EH = np.sum(high_freq ** 2)
    # 根据阈值计算退化指标
    color_cast = abs(rgb_diff - T_COLOR)
    low_light = abs(mean_brightness - T_LIGHT)
    blur = abs(variance - T_BLUR)
    return color_cast, low_light, blur, mu_H, sigma_H, std_dev_brightness, EH

# 参数设定
T_COLOR = 50  # 颜色偏移阈值
T_LIGHT = 50  # 低光照阈值
T_BLUR = 300  # 模糊阈值

# 综合判断图像类型
def classify_image_by_metrics(color_cast, low_light, blur, mu_H, sigma_H, std_dev_brightness, EH):
    # 定义每个指标的权重
    weight_color_cast = 0.25
    weight_low_light = 0.25
    weight_blur = 0.2
    weight_mu_H = 0.15
    weight_sigma_H = 0.1
    weight_std_dev_brightness = 0.05
    weight_EH = 0.1

    # 归一化处理，将指标值缩放到0到1之间
    normalized_color_cast = min(color_cast / T_COLOR, 1)
    normalized_low_light = min(low_light / T_LIGHT, 1)
    normalized_blur = min(blur / T_BLUR, 1)
    normalized_mu_H = min(mu_H / 360, 1)  # 假设色调的范围是0-360
    normalized_sigma_H = min(sigma_H / 100, 1)  # 假设色调标准差的最大值为100
    normalized_std_dev_brightness = min(std_dev_brightness / 50, 1)
    normalized_EH = min(EH / 4E+15, 1)

    # 计算加权得分
    score = (normalized_color_cast * weight_color_cast +
             normalized_low_light * weight_low_light +
             normalized_blur * weight_blur +
             normalized_mu_H * weight_mu_H +
             normalized_sigma_H * weight_sigma_H +
             normalized_std_dev_brightness * weight_std_dev_brightness +
             normalized_EH * weight_EH)

    # 根据加权得分来分类
    if score >= 0.8:
        return "Clear", normalized_color_cast, normalized_low_light, normalized_blur, normalized_mu_H, normalized_sigma_H, normalized_std_dev_brightness, normalized_EH, score
    elif score >= 0.63:
        return "Color Cast", normalized_color_cast, normalized_low_light, normalized_blur, normalized_mu_H, normalized_sigma_H, normalized_std_dev_brightness, normalized_EH, score
    elif score >= 0.50:
        return "Low Light", normalized_color_cast, normalized_low_light, normalized_blur, normalized_mu_H, normalized_sigma_H, normalized_std_dev_brightness, normalized_EH, score
    else:
        return "Blur", normalized_color_cast, normalized_low_light, normalized_blur, normalized_mu_H, normalized_sigma_H, normalized_std_dev_brightness, normalized_EH, score


# 图像分类函数
def classify_images(image_folder, output_excel, output_details_excel):
    results = []
    details = []

    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        # 分析图像退化类型和指标
        color_cast, low_light, blur, mu_H, sigma_H, std_dev_brightness, EH = analyze_image(file_path)

        # 综合判断图像类型
        classification, normalized_color_cast, normalized_low_light, normalized_blur, normalized_mu_H, normalized_sigma_H, normalized_std_dev_brightness, normalized_EH, score = classify_image_by_metrics(
        color_cast, low_light, blur, mu_H, sigma_H, std_dev_brightness, EH)

        # 记录结果
        results.append({
            "image file name": file_name,
            "Degraded Image Classification": classification,
            "PSNR": None,
            "UCIQE": None,
            "UIQM": None
        })

        # 记录退化指标和归一化后的指标
        details.append({
            "image file name": file_name,
            "color_cast": color_cast,
            "low_light": low_light,
            "blur": blur,
            "mu_H": mu_H,
            "sigma_H": sigma_H,
            "std_dev_brightness": std_dev_brightness,
            "EH": EH,
            "normalized_color_cast": normalized_color_cast,
            "normalized_low_light": normalized_low_light,
            "normalized_blur": normalized_blur,
            "normalized_mu_H": normalized_mu_H,
            "normalized_sigma_H": normalized_sigma_H,
            "normalized_std_dev_brightness": normalized_std_dev_brightness,
            "normalized_EH": normalized_EH,
            "score": score
        })

    # 保存图像分类结果到 Excel
    df_results = pd.DataFrame(results)
    df_results.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"分类结果已保存至 {output_excel}")

    # 保存图像退化指标到 Excel
    df_details = pd.DataFrame(details)
    df_details.to_excel(output_details_excel, index=False, engine='openpyxl')
    print(f"退化指标已保存至 {output_details_excel}")


# 设置图像文件夹和输出路径
image_folder = r"D:\数模亚太杯\Attachment 1"
output_excel = r"D:\数模亚太杯\Answer1.xlsx"  # 使用 .xlsx 格式
output_details_excel = r"D:\数模亚太杯\question_1.xlsx"
classify_images(image_folder, output_excel, output_details_excel)
