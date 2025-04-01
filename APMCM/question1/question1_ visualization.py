from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

#RGB可视化   读取图片并转换为 RGB 模式
print("该程序展示问题一的结果可视化(需保证代码文件和查看图片文件在同一文件下)")
image_path = input("请输入要查看图片的名称：")
#image_path = 'image_165.png'  # 测试时候用
image = Image.open(image_path)
image = image.convert("RGB")  # 确保图像是 RGB 模式

width, height = image.size
# 提取 RGB
rgb_data = np.array(image)
r, g, b = rgb_data[:,:,0], rgb_data[:,:,1], rgb_data[:,:,2]
#二维三个图分开
# 绘制 RGB 分量的直方图
plt.figure(figsize=(15, 5))
# 红色
plt.subplot(1, 3, 1)
plt.hist(r.flatten(), bins=256, color='red', alpha=0.6)
plt.title("Red Channel Histogram")
plt.xlim(0, 255)
# 绿色
plt.subplot(1, 3, 2)
plt.hist(g.flatten(), bins=256, color='green', alpha=0.6)
plt.title("Green Channel Histogram")
plt.xlim(0, 255)
# 蓝色
plt.subplot(1, 3, 3)
plt.hist(b.flatten(), bins=256, color='blue', alpha=0.6)
plt.title("Blue Channel Histogram")
plt.xlim(0, 255)
# 显示图形
plt.tight_layout()
plt.show()

#二维，增加可视化，三个图放到一起更加直观
plt.figure(figsize=(10, 6))
# 红色
plt.hist(r.flatten(), bins=256, color='red', alpha=0.6, label='Red Channel')
# 绿色
plt.hist(g.flatten(), bins=256, color='green', alpha=0.6, label='Green Channel')
# 蓝色
plt.hist(b.flatten(), bins=256, color='blue', alpha=0.6, label='Blue Channel')
# 设置标题和轴范围
plt.title("RGB Channel Histograms")
plt.xlim(0, 255)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
# 添加图例
plt.legend()
# 显示图形
plt.show()

#绘制3D图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# 计算每个颜色通道的直方图数据
r_hist, r_bins = np.histogram(r.flatten(), bins=256, range=(0, 255))
g_hist, g_bins = np.histogram(g.flatten(), bins=256, range=(0, 255))
b_hist, b_bins = np.histogram(b.flatten(), bins=256, range=(0, 255))
# 设置每个颜色通道直方图的 X 轴位置（即每个直方条的中心）
r_x = (r_bins[:-1] + r_bins[1:]) / 2
g_x = (g_bins[:-1] + g_bins[1:]) / 2
b_x = (b_bins[:-1] + b_bins[1:]) / 2
# 红色
ax.bar(r_x, r_hist, zs=0, zdir='y', color='red', alpha=0.6, label='Red')
# 绿色
ax.bar(g_x, g_hist, zs=1, zdir='y', color='green', alpha=0.6, label='Green')
# 蓝色
ax.bar(b_x, b_hist, zs=2, zdir='y', color='blue', alpha=0.6, label='Blue')
# 设置轴标签
ax.set_xlabel('Pixel Intensity')
ax.set_ylabel('Color Channels')
ax.set_zlabel('Frequency')
# 设置 Y 轴的范围，这样可以显示所有的通道
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Red Channel', 'Green Channel', 'Blue Channel'])
# 设置标题
ax.set_title("RGB Channel Histograms (3D)")
# 添加图例
ax.legend()
# 显示图形
plt.show()

# 加载图像并转换为HSV空间
image = cv2.imread(image_path)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 提取色调(Hue)通道
hue = hsv_image[:, :, 0]

# 计算色调均值和标准差
N = hue.size
mu_H = np.mean(hue)
sigma_H = np.sqrt(np.sum((hue - mu_H) ** 2) / N)

# 可视化
# 1. 显示色调(Hue)通道
plt.figure(figsize=(10, 6))

# 显示原始图像中的Hue通道
plt.subplot(2, 2, 2)
plt.imshow(hue, cmap='hsv')
plt.title("Hue Channel")
plt.axis('off')

# 2. 绘制色调直方图
plt.subplot(2, 2, 3)
plt.hist(hue.ravel(), bins=256, range=(0, 256), color='orange', alpha=0.7)
plt.axvline(mu_H, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mu_H:.2f}')
plt.axvline(mu_H + sigma_H, color='green', linestyle='dashed', linewidth=2, label=f'Standard Deviation: {sigma_H:.2f}')
plt.axvline(mu_H - sigma_H, color='green', linestyle='dashed', linewidth=2)
plt.title("Hue Histogram")
plt.xlabel("Hue Value")
plt.ylabel("Frequency")
plt.legend()

# 3. 显示色调的均值和标准差
plt.subplot(2, 2, 4)
plt.bar([0], [mu_H], width=0.1, color='red', label=f'Mean: {mu_H:.2f}')
plt.bar([1], [sigma_H], width=0.1, color='green', label=f'Std Dev: {sigma_H:.2f}')
plt.title("Mean and Std Dev of Hue")
plt.xticks([0, 1], ['Mean', 'Std Dev'])
plt.ylabel("Value")
plt.legend()

# 4. 显示图像本身
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.tight_layout()
plt.show()

def is_low_light(image):
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的平均亮度
    mean_brightness = np.mean(gray_image)

    # 计算图像的亮度标准差
    std_dev_brightness = np.std(gray_image)

    return gray_image, mean_brightness, std_dev_brightness

image = cv2.imread(image_path)

# 判断是否为低光图像
gray_image,  mean_brightness, std_dev_brightness = is_low_light(image)

# 将平均亮度和标准差信息转化为字符串
brightness_text = f"Avg Brightness: {mean_brightness:.2f}, Std Dev: {std_dev_brightness:.2f}"

# 在图像上添加平均亮度和标准差文本
cv2.putText(gray_image, brightness_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 显示灰度图像
cv2.imshow("Gray Image with Brightness and Std Dev", gray_image)  # 显示灰度图像
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()  # 关闭窗口

def is_blurry_laplacian(image):
    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的拉普拉斯变换
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

    # 计算拉普拉斯方差
    variance = laplacian.var()

    return  variance

def is_blurry_fft(image):
    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的傅里叶变换
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)  # 将零频移至中心

    # 计算频谱的幅度
    magnitude = np.abs(fshift)

    # 获取高频区域（图像的右上角部分）
    rows, cols = magnitude.shape
    center_row, center_col = rows // 2, cols // 2
    high_freq = magnitude[:center_row, center_col:]

    # 计算高频区域的能量
    EH = np.sum(high_freq**2)

    return  EH

image = cv2.imread(image_path)

# 判断图像是否模糊（基于拉普拉斯变换）
variance = is_blurry_laplacian(image)
# 判断图像是否模糊（基于频率域分析）
high_freq_energy = is_blurry_fft(image)

# 输出拉普拉斯变换和频率域分析结果
print(f"拉普拉斯变换方差: {variance:.2f}")
print(f"频率域高频能量: {high_freq_energy:.2f}")

# 可视化原图、拉普拉斯变换和频率域分析结果
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算拉普拉斯变换
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)

# 计算傅里叶变换的幅度
f = np.fft.fft2(gray_image)
fshift = np.fft.fftshift(f)
magnitude = np.abs(fshift)

# 可视化
plt.figure(figsize=(15, 7))
# 原图
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
# 拉普拉斯变换结果
plt.subplot(1, 3, 2)
plt.imshow(laplacian_abs, cmap='gray')
plt.title("Laplace Transform Result")
plt.axis('off')
# 傅里叶变换幅度
plt.subplot(1, 3, 3)
plt.imshow(np.log(magnitude + 1), cmap='gray')  # 对频谱进行对数变换，使高频更易于观察
plt.title("Fourier Transform Magnitude")
plt.axis('off')

plt.show()
