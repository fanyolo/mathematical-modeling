import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体（优先使用 SimHei，若无则默认）
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv("bga_solder_ball_positions.csv")

# 随机生成缺陷（缺陷率 10%）
import numpy as np
rng = np.random.default_rng(42)
num_defects = int(0.1 * len(df))
defect_indices = rng.choice(len(df), size=num_defects, replace=False)

df["缺陷"] = 0
df.loc[defect_indices, "缺陷"] = 1

# 颜色映射
colors = df["缺陷"].map({0: "blue", 1: "red"})

# 绘图
plt.figure(figsize=(6, 6))
plt.scatter(df["x"], df["y"], c=colors, s=20, edgecolors="k", linewidths=0.3)

# 图例与标签
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='正常焊球', markerfacecolor='blue', markersize=6),
    plt.Line2D([0], [0], marker='o', color='w', label='缺陷焊球', markerfacecolor='red', markersize=6)
]
plt.legend(handles=handles, loc='upper right', fontsize=10)
plt.title("缺陷率 10% 下的焊球分布图", fontsize=12)
plt.xlabel("x 坐标（mm）", fontsize=10)
plt.ylabel("y 坐标（mm）", fontsize=10)
plt.axis("equal")
plt.grid(True)
plt.tight_layout()

# 保存图像
plt.savefig("bga_geometry_snapshot_defect10_cn.png", dpi=300)
plt.show()