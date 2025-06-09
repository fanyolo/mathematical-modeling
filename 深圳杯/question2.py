import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# --- 中文字体设置 ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- QFN封装材料参数 ---
# 环氧树脂(overmold)
E1 = 16e9  # 杨氏模量 (Pa)
CTE1 = 15e-6  # 热膨胀系数 (1/°C)

# 芯片
E2 = 131e9  # 杨氏模量 (Pa)
CTE2 = 2.8e-6  # 热膨胀系数 (1/°C)

# 铜
E3 = 117e9  # 杨氏模量 (Pa)
CTE3 = 17e-6  # 热膨胀系数 (1/°C)

# PCB板
E4 = 22e9  # 杨氏模量 (Pa)
CTE4 = 18e-6  # 热膨胀系数 (1/°C)

# --- QFN封装几何参数 ---
# 芯片
L1 = 5e-3  # 长宽 (m)
LD1 = L1 * np.sqrt(2) / 2  # 半对角长度 (m)

# 环氧树脂
L2 = 10e-3  # 长宽 (m)
LD2 = L2 * np.sqrt(2) / 2  # 半对角长度 (m)

# 铜焊盘
L3 = 7e-3  # 长宽 (m)
LD3 = L3 * np.sqrt(2) / 2  # 半对角长度 (m)

# 焊料
LEN1 = 0.5e-3  # 长度 (m)
H1 = 0.1e-3  # 高度 (m)

# 厚度参数
H2 = 0.5e-3  # 环氧树脂厚度 (m)
H3 = 0.3e-3  # 芯片厚度 (m)
H4 = 0.2e-3  # 铜焊盘厚度 (m)
H5 = 1.0e-3  # QFN封装总厚度 (m)


# --- 计算对角线方向的等效参数 ---
def calculate_equivalent_properties():
    """计算QFN封装在角点位置沿对角线方向的等效参数"""

    # 1. 计算各层沿对角线的截面积
    # 第一层 (仅环氧树脂)
    A1 = LD2 * H2  # 对角线截面积
    f1_epoxy = 1.0

    # 第二层 (环氧树脂和芯片)
    A2 = LD2 * H3
    f2_epoxy = (LD2 - LD1) / LD2  # 对角线方向面积分数
    f2_chip = LD1 / LD2

    # 第三层 (环氧树脂和铜焊盘)
    A3 = LD2 * H4
    f3_epoxy = (LD2 - LD3) / LD2
    f3_copper = LD3 / LD2

    # 焊料层 (简化为铜)
    A_solder = 4 * (LEN1 * H1)  # 4个角的焊料

    # 总截面积
    A_total = A1 + A2 + A3 + A_solder

    # 2. 计算各层等效参数
    # 杨氏模量
    E_layer1 = f1_epoxy * E1
    E_layer2 = f2_epoxy * E1 + f2_chip * E2
    E_layer3 = f3_epoxy * E1 + f3_copper * E3
    E_solder = E3  # 焊料简化为铜

    # 热膨胀系数
    CTE_layer1 = CTE1
    CTE_layer2 = (f2_epoxy * E1 * CTE1 + f2_chip * E2 * CTE2) / (f2_epoxy * E1 + f2_chip * E2)
    CTE_layer3 = (f3_epoxy * E1 * CTE1 + f3_copper * E3 * CTE3) / (f3_epoxy * E1 + f3_copper * E3)
    CTE_solder = CTE3

    # 3. 计算整体等效参数 (串联模型)
    # 等效杨氏模量
    E_eq = (A1 * E_layer1 + A2 * E_layer2 + A3 * E_layer3 + A_solder * E_solder) / A_total

    # 等效热膨胀系数 (考虑热-机械耦合)
    numerator = (A1 * E_layer1 * CTE_layer1 + A2 * E_layer2 * CTE_layer2 +
                 A3 * E_layer3 * CTE_layer3 + A_solder * E_solder * CTE_solder)
    denominator = (A1 * E_layer1 + A2 * E_layer2 + A3 * E_layer3 + A_solder * E_solder)
    CTE_eq = numerator / denominator

    return E_eq, CTE_eq, A1, A2, A3, A_solder


# 计算等效参数
E_eq, CTE_eq, A1, A2, A3, A_solder = calculate_equivalent_properties()

# --- 输出结果 ---
print("=== QFN封装角点对角线方向等效参数 ===")
print(f"1. 各层截面积 (m²):")
print(f"   第一层(纯环氧树脂): {A1:.2e}")
print(f"   第二层(环氧树脂+芯片): {A2:.2e}")
print(f"   第三层(环氧树脂+铜焊盘): {A3:.2e}")
print(f"   焊料层: {A_solder:.2e}")

print("\n2. 等效参数:")
print(f"   等效杨氏模量: {E_eq / 1e9:.2f} GPa")
print(f"   等效热膨胀系数: {CTE_eq * 1e6:.2f} ppm/°C")

# --- 可视化封装结构 ---
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制封装结构示意图
layers = [
    {"name": "焊料层", "height": H1, "color": "gold", "hatch": "//"},
    {"name": "第三层(环氧树脂+铜)", "height": H4, "color": "lightblue", "hatch": "xx"},
    {"name": "第二层(环氧树脂+芯片)", "height": H3, "color": "lightgreen", "hatch": ".."},
    {"name": "第一层(纯环氧树脂)", "height": H2, "color": "pink", "hatch": "++"}
]

current_height = 0
for layer in layers:
    ax.bar(0, layer["height"], bottom=current_height,
           width=0.5, color=layer["color"],
           edgecolor="black", hatch=layer["hatch"],
           label=layer["name"])
    current_height += layer["height"]

ax.set_title("QFN封装结构示意图", fontsize=14)
ax.set_ylabel("厚度 (mm)")
ax.set_ylim(0, H5 * 1.1)

# 设置固定刻度位置和标签
yticks = np.linspace(0, H5, 6)  # 创建6个均匀分布的刻度
ax.set_yticks(yticks)  # 先设置刻度位置
ax.set_yticklabels([f"{y * 1000:.1f}" for y in yticks])  # 再设置标签

ax.set_xticks([])
ax.legend()

plt.tight_layout()
plt.show()