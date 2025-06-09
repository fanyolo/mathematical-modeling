import numpy as np
import matplotlib.pyplot as plt

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 缺焊率 (%) 与等效杨氏模量（GPa）
缺焊率 = np.array([0.0, 0.05, 0.1, 0.2, 0.3])
等效杨氏模量 = np.array([6.78, 6.53, 6.24, 5.85, 5.51])
应力放大 = 等效杨氏模量[0] / 等效杨氏模量

# 空间网格
n_points = 100
x = np.linspace(0, 1, n_points)
y = np.linspace(0, 1, n_points)
X, Y = np.meshgrid(x, y)

# 构造应力场
def 构造应力场(放大系数, 种子):
    基本应力 = np.exp(-10 * ((X - 1) ** 2 + (Y - 1) ** 2))
    噪声 = 0.2 * np.random.default_rng(种子).normal(size=基本应力.shape)
    return 基本应力 * 放大系数 + 噪声

应力帧 = [构造应力场(f, int(r * 100)) for r, f in zip(缺焊率, 应力放大)]
最大应力值 = np.max(应力帧)

# 横向拼图
fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)

for i, ax in enumerate(axes):
    im = ax.imshow(应力帧[i], extent=(0, 1, 0, 1), origin="lower", cmap="inferno", vmin=0, vmax=最大应力值)
    ax.set_title(f"缺焊率：{缺焊率[i]*100:.1f}%", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

# 添加统一色条
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, orientation='horizontal', pad=0.1)
cbar.set_label("等效 Von Mises 应力（单位）", fontsize=12)

plt.show()

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, ax in enumerate(axes):
    cs = ax.contour(X, Y, 应力帧[i], levels=8, cmap='magma')
    ax.set_title(f"缺焊率 {缺焊率[i]*100:.0f}%")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("等效应力等高线图")
plt.tight_layout()
plt.show()
