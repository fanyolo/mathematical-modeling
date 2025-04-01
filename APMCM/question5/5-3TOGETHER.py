# Python

import cv2
import numpy as np
import os

def combine_three_images_horizontally(img_path1, img_path2, img_path3, output_path, resize_dim=(800, 800), spacing=20):
    """
    将三张图片水平排列在一张图片中，中间添加指定间隔，并提高输出图的分辨率。

    参数：
        img_path1 (str): 第一张图片的路径。
        img_path2 (str): 第二张图片的路径。
        img_path3 (str): 第三张图片的路径。
        output_path (str): 合并后图片的保存路径。
        resize_dim (tuple): 调整图片大小为 (宽度, 高度)（默认为800x800，以提高分辨率）。
        spacing (int): 图片之间的间隔像素数（默认为20）。
    """
    # 读取图片
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    img3 = cv2.imread(img_path3)

    # 检查图片是否成功读取
    if img1 is None:
        print(f"无法读取图片: {img_path1}")
        return
    if img2 is None:
        print(f"无法读取图片: {img_path2}")
        return
    if img3 is None:
        print(f"无法读取图片: {img_path3}")
        return

    # 调整图片大小
    img1 = cv2.resize(img1, resize_dim, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, resize_dim, interpolation=cv2.INTER_AREA)
    img3 = cv2.resize(img3, resize_dim, interpolation=cv2.INTER_AREA)

    # 创建间隔
    spacer = np.ones((resize_dim[1], spacing, 3), dtype=np.uint8) * 255  # 白色间隔

    # 合并图片和间隔
    combined_img = np.hstack((img1, spacer, img2, spacer, img3))

    # 保存合并后的图片
    cv2.imwrite(output_path, combined_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # 增加JPEG质量
    print(f"合并后的图片已保存到: {output_path}")

# 示例用法
if __name__ == "__main__":
    img_path1 = r'D:\ytb\5\PSNR_comparisonen.png'
    img_path2 = r'D:\ytb\5\UCIQE_comparisonen.png'
    img_path3 = r'D:\ytb\5\UIQM_comparisonen.png'
    output_path = r'D:\ytb\5\333_high_res.jpg'
    combine_three_images_horizontally(
        img_path1, 
        img_path2, 
        img_path3, 
        output_path, 
        resize_dim=(800, 800),  # 提高分辨率
        spacing=40  # 增加间隔以适应更大的图片
    )