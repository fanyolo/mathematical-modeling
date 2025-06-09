import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.font_manager import FontProperties, FontManager
import os


# --- 增强版中文字体设置 ---
def set_chinese_font():
    """尝试多种方法设置中文字体"""
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 方法1：尝试常见中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei',
                     'Noto Sans CJK SC', 'FangSong', 'KaiTi', 'sans-serif']

    # 方法2：检查系统字体
    available_fonts = set([f.name for f in FontManager().ttflist])
    for font in chinese_fonts:
        if font in available_fonts:
            try:
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [font]
                print(f"成功设置字体: {font}")
                return True
            except:
                continue

    # 方法3：尝试指定字体文件
    font_files = [
        'C:/Windows/Fonts/msyh.ttc',  # Windows 雅黑
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'  # Linux
    ]

    for font_file in font_files:
        if os.path.exists(font_file):
            try:
                font_prop = FontProperties(fname=font_file)
                plt.rcParams['font.family'] = font_prop.get_name()
                print(f"成功从文件加载字体: {font_file}")
                return True
            except:
                continue

    print("警告: 无法设置中文字体，中文显示可能不正常")
    return False


# 执行字体设置
set_chinese_font()

# 确保禁用LaTeX文本渲染
plt.rcParams['text.usetex'] = False

# --- 材料属性 ---
E1_pcb = 1.5e10
CTE1_pcb = 20e-6
E2_solder = 2.0e9
CTE2_solder = 25e-6
E3_bga = 1.2e10
CTE3_bga = 30e-6

L = 10  # 改为毫米单位
mesh_resolution = 5

# --- 创建网格 ---
x = np.linspace(0, L, mesh_resolution + 1)
y = np.linspace(0, L, mesh_resolution + 1)
xx, yy = np.meshgrid(x, y)
points_coords = np.column_stack([xx.ravel(), yy.ravel()])

# 创建三角形网格
triangles = []
for i in range(mesh_resolution):
    for j in range(mesh_resolution):
        v0 = i * (mesh_resolution + 1) + j
        v1 = v0 + 1
        v2 = v0 + (mesh_resolution + 1)
        v3 = v2 + 1

        triangles.append([v0, v1, v2])
        triangles.append([v1, v3, v2])

triangles = np.array(triangles)
num_cells = len(triangles)


# --- 材料区域定义 ---
class PCB_Region:
    def inside(self, x_centroid): return x_centroid[0] < L / 2 and x_centroid[1] < L / 2


class BGA_Region:
    def inside(self, x_centroid): return x_centroid[0] > L / 2 and x_centroid[1] > L / 2


pcb_detector = PCB_Region()
bga_detector = BGA_Region()

tags_cell = np.zeros(num_cells, dtype=int)
for i in range(num_cells):
    vertex_indices = triangles[i]
    coords = points_coords[vertex_indices]
    centroid = coords.mean(axis=0)
    if pcb_detector.inside(centroid):
        tags_cell[i] = 1
    elif bga_detector.inside(centroid):
        tags_cell[i] = 2

# --- 计算面积分数 ---
A1_pcb_area = (L / 2) ** 2
A3_bga_area = (L / 2) ** 2
A_total = L ** 2
A2_solder_area = A_total - A1_pcb_area - A3_bga_area

f1_pcb_frac = A1_pcb_area / A_total
f2_solder_frac = A2_solder_area / A_total
f3_bga_frac = A3_bga_area / A_total

# --- 计算等效参数 ---
denominator_tensile = 0
if E1_pcb > 0: denominator_tensile += f1_pcb_frac / E1_pcb
if E2_solder > 0: denominator_tensile += f2_solder_frac / E2_solder
if E3_bga > 0: denominator_tensile += f3_bga_frac / E3_bga
E_eq_tensile = 1.0 / denominator_tensile if denominator_tensile > 0 else 0

E_eq_bending = (f1_pcb_frac * E1_pcb +
                f2_solder_frac * E2_solder +
                f3_bga_frac * E3_bga)

numerator_cte = (CTE1_pcb * E1_pcb * f1_pcb_frac +
                 CTE2_solder * E2_solder * f2_solder_frac +
                 CTE3_bga * E3_bga * f3_bga_frac)
denominator_cte_E_eff = (E1_pcb * f1_pcb_frac +
                         E2_solder * f2_solder_frac +
                         E3_bga * f3_bga_frac)
alpha_eq = numerator_cte / denominator_cte_E_eff if denominator_cte_E_eff > 0 else 0

print(f"PCB 面积分数 (f1): {f1_pcb_frac:.2f}")
print(f"焊球 面积分数 (f2): {f2_solder_frac:.2f}")
print(f"BGA 面积分数 (f3): {f3_bga_frac:.2f}")
print(f"--- 等效参数 ---")
print(f"等效拉伸杨氏模量 (Reuss): {E_eq_tensile:.2e} Pa")
print(f"等效弯曲杨氏模量 (Voigt): {E_eq_bending:.2e} Pa")
print(f"等效热膨胀系数: {alpha_eq:.2e} 1/°C")

# --- 顶点属性计算 ---
connectivity_verts_to_cells = [[] for _ in range(len(points_coords))]
for cell_idx, triangle in enumerate(triangles):
    for vertex_idx in triangle:
        connectivity_verts_to_cells[vertex_idx].append(cell_idx)

triang = Triangulation(points_coords[:, 0], points_coords[:, 1], triangles)

num_points = points_coords.shape[0]
alpha_values_at_vertices = np.zeros(num_points)
E_values_at_vertices = np.zeros(num_points)

for i in range(num_points):
    neighboring_cells_indices = connectivity_verts_to_cells[i]

    if neighboring_cells_indices:
        ctes, Es = [], []
        for cell_idx in neighboring_cells_indices:
            tag = tags_cell[cell_idx]
            if tag == 1:  # PCB
                ctes.append(CTE1_pcb)
                Es.append(E1_pcb)
            elif tag == 2:  # BGA
                ctes.append(CTE3_bga)
                Es.append(E3_bga)
            else:  # Solder
                ctes.append(CTE2_solder)
                Es.append(E2_solder)

        alpha_values_at_vertices[i] = np.mean(ctes)
        E_values_at_vertices[i] = np.mean(Es)

# --- 可视化 ---
# 图1：热膨胀系数分布 (单独显示)
plt.figure(figsize=(8, 6))
tpc1 = plt.tripcolor(triang, alpha_values_at_vertices, cmap='viridis', shading='gouraud')
plt.title("热膨胀系数分布 (CTE)")
plt.xlabel("X 坐标 (mm)")
plt.ylabel("Y 坐标 (mm)")
plt.colorbar(tpc1, label="热膨胀系数 (1/°C)")
plt.tight_layout()
plt.show()

# 图2：杨氏模量分布 (单独显示)
plt.figure(figsize=(8, 6))
tpc2 = plt.tripcolor(triang, E_values_at_vertices, cmap='plasma', shading='gouraud')
plt.title("杨氏模量分布 (E)")
plt.xlabel("X 坐标 (mm)")
plt.ylabel("Y 坐标 (mm)")
plt.colorbar(tpc2, label="杨氏模量 (Pa)")
plt.tight_layout()
plt.show()

# 图3：等效参数比较 (单独显示)
plt.figure(figsize=(8, 6))
x_labels = ["拉伸模量\n(Reuss)", "弯曲模量\n(Voigt)", "热膨胀系数\n(复合)"]
x_pos = np.arange(3)
values = [E_eq_tensile, E_eq_bending, alpha_eq]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

plt.bar(x_pos[:2], values[:2], color=colors[:2], width=0.6, label="模量 (Pa)")
plt.ylabel("模量值 (Pa)", color=colors[0])
plt.tick_params(axis='y', color=colors[0])

ax2 = plt.twinx()
ax2.bar(x_pos[2], values[2], color=colors[2], width=0.6, label="热膨胀系数")
ax2.set_ylabel("热膨胀系数 (1/°C)", color=colors[2])
ax2.tick_params(axis='y', color=colors[2])

plt.xticks(x_pos, x_labels)
plt.title("等效材料参数比较")

# 处理图例警告
handles1, labels1 = plt.gca().get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
plt.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=2)

plt.tight_layout()
plt.show()