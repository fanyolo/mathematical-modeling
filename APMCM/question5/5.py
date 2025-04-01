# Python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compare_metrics_visualizations(excel1, excel2, output_dir):
    """
    从两个Excel表中读取PSNR、UCIQE、UIQM数据，并生成多种对比图形。
    
    参数：
        excel1 (str): 第一个Excel文件路径。
        excel2 (str): 第二个Excel文件路径。
        output_dir (str): 输出图像的保存目录。
    """
    # 读取Excel文件
    df1 = pd.read_excel(excel1)
    df2 = pd.read_excel(excel2)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 指标列表
    metrics = ['PSNR', 'UCIQE', 'UIQM']
    
    for metric in metrics:
        # 1. 柱状图
        plt.figure(figsize=(8,6))
        bar_width = 0.35
        index = range(len(df1))
        
        plt.bar([i - bar_width/2 for i in index], df1[metric], bar_width, label='单场景')
        plt.bar([i + bar_width/2 for i in index], df2[metric], bar_width, label='复杂场景')
        
        plt.title(f'{metric} 对比柱状图')
        plt.xlabel('样本')
        plt.ylabel(metric)
        plt.xticks(index, [f'Sample {i+1}' for i in index])
        plt.legend()
        plt.tight_layout()
        
        output_path_bar = os.path.join(output_dir, f'{metric}_bar_comparison.png')
        plt.savefig(output_path_bar)
        plt.close()
        print(f"{metric} 柱状图已保存到: {output_path_bar}")
        
        # 2. 箱线图
        plt.figure(figsize=(8,6))
        sns.boxplot(data=[df1[metric], df2[metric]], palette=['skyblue', 'salmon'])
        plt.title(f'{metric} 对比箱线图')
        plt.xlabel('场景类型')
        plt.ylabel(metric)
        plt.xticks([0,1], ['单场景', '复杂场景'])
        plt.tight_layout()
        
        output_path_box = os.path.join(output_dir, f'{metric}_box_comparison.png')
        plt.savefig(output_path_box)
        plt.close()
        print(f"{metric} 箱线图已保存到: {output_path_box}")
        
        # 3. 散点图
        plt.figure(figsize=(8,6))
        plt.scatter(df1.index, df1[metric], color='blue', label='单场景', alpha=0.7)
        plt.scatter(df2.index, df2[metric], color='red', label='复杂场景', alpha=0.7)
        
        plt.title(f'{metric} 对比散点图')
        plt.xlabel('样本')
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        
        output_path_scatter = os.path.join(output_dir, f'{metric}_scatter_comparison.png')
        plt.savefig(output_path_scatter)
        plt.close()
        print(f"{metric} 散点图已保存到: {output_path_scatter}")
        
        # 4. 雷达图
        labels = df1.columns if metric in df1.columns else []
        angles = np.linspace(0, 2 * np.pi, len(df1), endpoint=False).tolist()
        angles += angles[:1]  # 完成雷达图闭合
        
        fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
        
        values1 = df1[metric].tolist()
        values2 = df2[metric].tolist()
        values1 += values1[:1]
        values2 += values2[:1]
        
        ax.plot(angles, values1, linewidth=1, linestyle='solid', label='单场景')
        ax.fill(angles, values1, alpha=0.1)
        
        ax.plot(angles, values2, linewidth=1, linestyle='solid', label='复杂场景')
        ax.fill(angles, values2, alpha=0.1)
        
        ax.set_title(f'{metric} 对比雷达图', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        output_path_radar = os.path.join(output_dir, f'{metric}_radar_comparison.png')
        plt.savefig(output_path_radar, bbox_inches='tight')
        plt.close()
        print(f"{metric} 雷达图已保存到: {output_path_radar}")
        
    # 其他可视化方式如热力图、堆叠面积图等也可根据需求添加

# 示例用法
if __name__ == "__main__":
    excel1 = r'D:\ytb\3excel.xlsx'
    excel2 = r'D:\ytb\111.xlsx'
    output_dir = r'D:\ytb\5'
    compare_metrics_visualizations(excel1, excel2, output_dir)