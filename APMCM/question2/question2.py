import numpy as np
import cv2
import matplotlib.pyplot as plt

# 传输系数计算函数
def transmission_coefficient(depth, beta):
    return np.exp(-beta * depth)

# 颜色偏移退化模型
def color_shift_model(J, B, depth, beta):
    # 计算每个颜色通道的传输率
    tc_R = transmission_coefficient(depth, beta['R'])
    tc_G = transmission_coefficient(depth, beta['G'])
    tc_B = transmission_coefficient(depth, beta['B'])

    # 计算退化后的颜色分量
    Ic_R = J[:, :, 2] * tc_R + B[:, :, 2] * (1 - tc_R)  # Red channel
    Ic_G = J[:, :, 1] * tc_G + B[:, :, 1] * (1 - tc_G)  # Green channel
    Ic_B = J[:, :, 0] * tc_B + B[:, :, 0] * (1 - tc_B)  # Blue channel

    # 合并颜色通道
    Ic = cv2.merge((Ic_B, Ic_G, Ic_R))
    return np.clip(Ic, 0, 255).astype(np.uint8)

# 低光照退化模型
def low_light_model(J, B, depth, beta):
    # 计算传输率
    t = transmission_coefficient(depth, beta['overall'])

    # 通过深度信息调整传输系数
    t = t[:, :, np.newaxis]  # 扩展维度以适应图像通道

    # 计算退化图像
    I = J * t + B * (1 - t)

    return np.clip(I, 0, 255).astype(np.uint8)

# 添加模糊效果
def add_blur(J, kernel_size=(5, 5)):
    return cv2.GaussianBlur(J, kernel_size, 0)

# 读取图像
print("该程序展示问题二的结果可视化(需保证代码文件和查看图片文件在同一文件下)")
image_path = input("请输入要查看图片的名称：")
#J = cv2.imread('19_img_.png')  # 清晰图像#测试时用
J = cv2.imread(image_path)
if J is None:
    print("无法加载图像，请检查路径")
    exit()

# 假设环境光均匀，所有像素值为50
B = np.full(J.shape, fill_value=50)

# 示例参数
beta = {'R': 0.1, 'G': 0.1, 'B': 0.02, 'overall': 0.15}
depth = np.ones((J.shape[0], J.shape[1])) * 10 # 假设深度为10米

# 进行颜色偏移建模
degraded_image_color_shift = color_shift_model(J, B, depth, beta)
# 进行低光照建模
degraded_image_low_light = low_light_model(J, B, depth, beta)
# 添加模糊效果
blurred_image = add_blur(J, kernel_size=(15, 15))  # 使用较大的内核模拟较强的模糊



# 显示结果
plt.figure(figsize=(15, 8))

# 原始清晰图像
plt.subplot(1, 4, 1)
plt.title('Clear Image')
plt.imshow(cv2.cvtColor(J, cv2.COLOR_BGR2RGB))
plt.axis('off')

# 颜色偏移退化图像
plt.subplot(1, 4, 2)
plt.title('Degraded Image (Color Shift)')
plt.imshow(cv2.cvtColor(degraded_image_color_shift, cv2.COLOR_BGR2RGB))
plt.axis('off')

# 低光照退化图像
plt.subplot(1, 4, 3)
plt.title('Degraded Image (Low Light)')
plt.imshow(cv2.cvtColor(degraded_image_low_light, cv2.COLOR_BGR2RGB))
plt.axis('off')

# 模糊图像
plt.subplot(1, 4, 4)
plt.title('Blurred Image')
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()


# 提取 RGB 数据
def extract_rgb(image):
    r, g, b = image[:,:,2], image[:,:,1], image[:,:,0]  # OpenCV uses BGR order
    return r, g, b

# 绘制3D柱状图对比原图和色彩偏移后的RGB分布
def plot_3d_comparative_rgb_histograms(original_image, degraded_image):
    # 提取 RGB 通道数据
    r_orig, g_orig, b_orig = extract_rgb(original_image)
    r_degraded, g_degraded, b_degraded = extract_rgb(degraded_image)

    # 计算直方图
    hist_r_orig, bins_r = np.histogram(r_orig.flatten(), bins=256, range=(0, 255))
    hist_g_orig, bins_g = np.histogram(g_orig.flatten(), bins=256, range=(0, 255))
    hist_b_orig, bins_b = np.histogram(b_orig.flatten(), bins=256, range=(0, 255))

    hist_r_degraded, _ = np.histogram(r_degraded.flatten(), bins=256, range=(0, 255))
    hist_g_degraded, _ = np.histogram(g_degraded.flatten(), bins=256, range=(0, 255))
    hist_b_degraded, _ = np.histogram(b_degraded.flatten(), bins=256, range=(0, 255))

    # 设置3D图像
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 计算位置
    x_r = bins_r[:-1]
    x_g = bins_g[:-1]
    x_b = bins_b[:-1]

    # 绘制红色通道的柱状图
    ax.bar(x_r, hist_r_orig, zs=0, zdir='y', width=1, color='red', alpha=0.6, label='Original Red')
    ax.bar(x_r, hist_r_degraded, zs=1, zdir='y', width=1, color='red', alpha=0.3, label='Degraded Red')

    # 绘制绿色通道的柱状图
    ax.bar(x_g, hist_g_orig, zs=2, zdir='y', width=1, color='green', alpha=0.6, label='Original Green')
    ax.bar(x_g, hist_g_degraded, zs=3, zdir='y', width=1, color='green', alpha=0.3, label='Degraded Green')

    # 绘制蓝色通道的柱状图
    ax.bar(x_b, hist_b_orig, zs=4, zdir='y', width=1, color='blue', alpha=0.6, label='Original Blue')
    ax.bar(x_b, hist_b_degraded, zs=5, zdir='y', width=1, color='blue', alpha=0.3, label='Degraded Blue')

    # 设置标签和标题
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Channel')
    ax.set_zlabel('Frequency')
    ax.set_title('3D Comparison of RGB Histograms: Original vs. Degraded')

    # 设置 Y 轴的刻度
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(['Original Red', 'Degraded Red', 'Original Green', 'Degraded Green', 'Original Blue', 'Degraded Blue'])

    # 显示图例
    ax.legend()

    plt.tight_layout()
    plt.show()

# 绘制 3D 对比图
plot_3d_comparative_rgb_histograms(J, degraded_image_color_shift)

def plot_rgb_pie_chart(image):
    r, g, b = extract_rgb(image)
    r_percentage = np.sum(r) / (r.size)
    g_percentage = np.sum(g) / (g.size)
    b_percentage = np.sum(b) / (b.size)

    # 绘制饼状图
    plt.figure(figsize=(6, 6))
    plt.pie([r_percentage, g_percentage, b_percentage], labels=['Red', 'Green', 'Blue'], autopct='%1.1f%%', startangle=90, colors=['red', 'green', 'blue'])
    plt.title('RGB Distribution (Pie Chart) - Original Image')
    plt.axis('equal')  # 保证饼状图是圆形
    plt.show()

# 绘制原图的二维 RGB 饼状图
plot_rgb_pie_chart(J)

# 计算图像亮度均值与标准差
def calculate_mean_std(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    std = np.std(gray)
    return mean, std

# 计算图像对比度（标准差）
def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

# 计算拉普拉斯算子（局部对比度与锐度）
def laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

# 计算图像动态范围
def calculate_dynamic_range(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val = np.min(gray)
    max_val = np.max(gray)
    return max_val - min_val

# 计算色彩饱和度
def calculate_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return np.mean(saturation)

# 计算图像梯度（亮度变化）
def calculate_gradient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return np.mean(magnitude)

# 计算图像噪声（噪声方差估计）
def calculate_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_var = np.var(gray)
    return noise_var


# 计算原图的特征
original_mean, original_std = calculate_mean_std(J)
original_contrast = calculate_contrast(J)
original_laplacian_var = laplacian_variance(J)
original_dynamic_range = calculate_dynamic_range(J)
original_saturation = calculate_saturation(J)
original_gradient = calculate_gradient(J)
original_noise = calculate_noise(J)

# 计算低光处理后的图像特征
degraded_mean, degraded_std = calculate_mean_std(degraded_image_low_light)
degraded_contrast = calculate_contrast(degraded_image_low_light)
degraded_laplacian_var = laplacian_variance(degraded_image_low_light)
degraded_dynamic_range = calculate_dynamic_range(degraded_image_low_light)
degraded_saturation = calculate_saturation(degraded_image_low_light)
degraded_gradient = calculate_gradient(degraded_image_low_light)
degraded_noise = calculate_noise(degraded_image_low_light)

# 可视化结果
features = {
    "Mean": (original_mean, degraded_mean),
    "Standard Deviation": (original_std, degraded_std),
    "Contrast": (original_contrast, degraded_contrast),
    "Laplacian Variance": (original_laplacian_var, degraded_laplacian_var),
    "Dynamic Range": (original_dynamic_range, degraded_dynamic_range),
    "Saturation": (original_saturation, degraded_saturation),
    "Gradient": (original_gradient, degraded_gradient),
    "Noise Variance": (original_noise, degraded_noise),
}

# 绘制折线图
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制原图和低光图像的折线图
feature_names = list(features.keys())
original_values = [x[0] for x in features.values()]
degraded_values = [x[1] for x in features.values()]

# 原图使用蓝色折线和圆形标记
ax.plot(feature_names, original_values, color='blue', marker='o', label='Original Image', linestyle='-', markersize=8, linewidth=2)

# 低光图使用红色折线和方形标记
ax.plot(feature_names, degraded_values, color='red', marker='s', label='Low Light Image', linestyle='-', markersize=8, linewidth=2)

# 设置图形标题与标签
ax.set_title("Comparison of Features: Original vs Low Light Image")
ax.set_xlabel("Feature")
ax.set_ylabel("Value")
ax.set_xticklabels(feature_names, rotation=45, ha="right")
ax.legend()

# 显示图形
plt.tight_layout()
plt.show()


