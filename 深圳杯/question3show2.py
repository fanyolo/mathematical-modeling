import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 加载焊球位置数据
df = pd.read_csv("bga_solder_ball_positions.csv")

# 模拟 500 次缺陷率 10%
simulations = 500
count = np.zeros(len(df))
rng = np.random.default_rng(0)

for _ in range(simulations):
    defect_rate = 0.1
    n_defects = int(defect_rate * len(df))
    defect_indices = rng.choice(len(df), size=n_defects, replace=False)
    count[defect_indices] += 1

df["缺陷概率"] = count / simulations

# 中文图例热图绘制
plt.figure(figsize=(6, 6))
sc = plt.scatter(df["x"], df["y"], c=df["缺陷概率"], cmap="hot", s=50, edgecolors='k')
cb = plt.colorbar(sc)
cb.set_label("缺焊概率 / 应力敏感性", fontsize=10)

plt.title("空间缺焊敏感性热图", fontsize=14)
plt.xlabel("x 坐标（mm）", fontsize=12)
plt.ylabel("y 坐标（mm）", fontsize=12)
plt.axis("equal")
plt.grid(True)
plt.tight_layout()

plt.show()
