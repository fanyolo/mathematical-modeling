import gmsh 
import numpy as np
import pandas as pd
from dolfinx.io import gmshio
from dolfinx.fem import (FunctionSpace, Function, Constant,
                         dirichletbc, form, locate_dofs_topological)
from ufl import VectorElement

from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc
import ufl

# --- 应力张量计算函数 ---
def get_material_stress_tensor(u_displacement, material_props, delta_T_val, identity_tensor):
    E_val = material_props["E"]
    nu_val = material_props["nu"]
    alpha_val = material_props["alpha"]
    mu_val = E_val / (2 * (1 + nu_val))
    lambda_val = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
    strain_total = ufl.sym(ufl.grad(u_displacement))
    strain_thermal = alpha_val * delta_T_val * identity_tensor
    strain_elastic = strain_total - strain_thermal
    stress = lambda_val * ufl.tr(strain_elastic) * identity_tensor + 2 * mu_val * strain_elastic
    return stress

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# 焊球坐标读取与转换
# 焊球坐标读取（直接使用文件中真实位置）
df_positions = pd.read_csv("bga_solder_ball_positions.csv")

# 不做任何偏移或缩放，直接提取坐标
ball_positions_all = df_positions[["x", "y"]].to_numpy()

# 保证焊球数量正确
assert ball_positions_all.shape[0] == 437


rates = [0.0, 0.05, 0.1, 0.2, 0.3]
stress_results_list = []
CTE_results_list = []
Eeq_results_list = []

for defect_rate in rates:
    if rank == 0:
        print(f"开始处理缺陷率：{defect_rate*100:.1f}%")

    gmsh.initialize()
    gmsh.model.add(f"MultiMaterial_BGA_defect_{defect_rate:.2f}")

    L_pcb, W_pcb, H_pcb = 0.14, 0.14, 0.0016
    L_bga, W_bga = 0.026, 0.026
    H_sub, H_mold = 0.0008, 0.00117
    L_chip, H_chip, H_chipcap = 0.0045, 0.0002, 0.00097

    diameter_ball = 0.0007
    h_total_ball = 0.0005
    r_ball = diameter_ball / 2
    h_ball_cylinder = h_total_ball

    deltaT = 100.0

    ox_bga, oy_bga = (L_pcb - L_bga) / 2, (W_pcb - W_bga) / 2
    pcb_geom = gmsh.model.occ.addBox(0, 0, 0, L_pcb, W_pcb, H_pcb)
    sub_geom = gmsh.model.occ.addBox(ox_bga, oy_bga, H_pcb, L_bga, W_bga, H_sub)
    mold_geom = gmsh.model.occ.addBox(ox_bga, oy_bga, H_pcb + H_sub, L_bga, W_bga, H_mold)

    chip_origin_x = L_pcb / 2 - L_chip / 2
    chip_origin_y = W_pcb / 2 - L_chip / 2
    chip_geom = gmsh.model.occ.addBox(chip_origin_x, chip_origin_y, H_pcb + H_sub + H_mold, L_chip, L_chip, H_chip)
    chipcap_geom = gmsh.model.occ.addBox(chip_origin_x, chip_origin_y, H_pcb + H_sub + H_mold + H_chip, L_chip, L_chip, H_chipcap)

    n_total_balls = ball_positions_all.shape[0]
    n_defect_balls = int(defect_rate * n_total_balls)
    rng = np.random.default_rng(42)
    defect_indices = rng.choice(n_total_balls, size=n_defect_balls, replace=False)

    current_ball_tags = []
    for i, (x_coord, y_coord) in enumerate(ball_positions_all):
        if i in defect_indices:
            continue
        tag = gmsh.model.occ.addCylinder(x_coord, y_coord, H_pcb, 0, 0, h_ball_cylinder, r_ball)
        current_ball_tags.append(tag)

    gmsh.model.occ.synchronize()
    pcb_tag, sub_tag, mold_tag, chip_tag, chipcap_tag, solder_tag = 1, 2, 3, 4, 5, 6
    gmsh.model.addPhysicalGroup(3, [pcb_geom], pcb_tag, name="PCB")
    gmsh.model.addPhysicalGroup(3, [sub_geom], sub_tag, name="Substrate")
    gmsh.model.addPhysicalGroup(3, [mold_geom], mold_tag, name="Mold")
    gmsh.model.addPhysicalGroup(3, [chip_geom], chip_tag, name="Chip")
    gmsh.model.addPhysicalGroup(3, [chipcap_geom], chipcap_tag, name="ChipCap")
    if current_ball_tags:
        gmsh.model.addPhysicalGroup(3, current_ball_tags, solder_tag, name="SolderBalls")

    gmsh.model.mesh.generate(3)
    mesh, cell_tags, _ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
    gmsh.finalize()

    materials_data = {
        pcb_tag: {"E": 28.6e9, "nu": 0.3, "alpha": 17e-6},
        sub_tag: {"E": 73.3e9, "nu": 0.3, "alpha": 21e-6},
        mold_tag: {"E": 11.7e9, "nu": 0.35, "alpha": 23e-6},
        chip_tag: {"E": 130e9, "nu": 0.3, "alpha": 2.6e-6},
        chipcap_tag: {"E": 11.7e9, "nu": 0.35, "alpha": 23e-6},
        solder_tag: {"E": 50e9, "nu": 0.35, "alpha": 20e-6}
    }

    element = VectorElement("CG", mesh.ufl_cell(), degree=1)
    V_space = FunctionSpace(mesh, element)

    u_trial = Function(V_space, name="Displacement")
    v_test = ufl.TestFunction(V_space)
    I_tensor = ufl.Identity(mesh.geometry.dim)
    dx_measure = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

    def bottom_surface_selector(x): return np.isclose(x[2], 0.0)
    mesh_dim = mesh.topology.dim
    facets_bottom = locate_entities_boundary(mesh, mesh_dim - 1, bottom_surface_selector)
    dofs_bottom = locate_dofs_topological(V_space, mesh_dim - 1, facets_bottom)
    bc_bottom = dirichletbc(Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0))), dofs_bottom, V_space)

    bilinear_form_terms = []
    for tag, mat_props in materials_data.items():
        if tag == solder_tag and not current_ball_tags:
            continue
        sigma_tensor_material = get_material_stress_tensor(u_trial, mat_props, deltaT, I_tensor)
        bilinear_form_terms.append(ufl.inner(sigma_tensor_material, ufl.sym(ufl.grad(v_test))) * dx_measure(tag))

    if not bilinear_form_terms:
        raise RuntimeError("未定义任何弱形式条目，可能未生成有效几何。")

    a_bilinear_form = form(sum(bilinear_form_terms))
    zero_vector = Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))
    L_linear_form = form(ufl.dot(zero_vector, v_test) * dx_measure)

    if rank == 0:
        print(f"  正在求解缺陷率为 {defect_rate*100:.1f}% 的情况...")
    problem = LinearProblem(a_bilinear_form, L_linear_form, bcs=[bc_bottom], petsc_options={})
    uh_solution = problem.solve()
    if rank == 0:
        print("  求解完成。")

    def pcb_top_right_corner_selector(x):
        return np.logical_and(np.isclose(x[0], L_pcb, atol=1e-3), np.isclose(x[1], W_pcb, atol=1e-3))
    facets_corner = locate_entities_boundary(mesh, mesh_dim - 1, pcb_top_right_corner_selector)
    corner_dofs = locate_dofs_topological(V_space, mesh_dim - 1, facets_corner)

    avg_diag_disp = np.nan
    if corner_dofs.size > 0:
        u_at_corner_dofs_flat = uh_solution.x.array[corner_dofs]
        if u_at_corner_dofs_flat.size % V_space.dofmap.bs == 0:
            u_corner_vectors = u_at_corner_dofs_flat.reshape(-1, V_space.dofmap.bs)
            pcb_diag_vector_xy = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
            projected_displacements = u_corner_vectors @ pcb_diag_vector_xy
            avg_diag_disp = np.mean(np.abs(projected_displacements))

    if np.isnan(avg_diag_disp):
        equiv_strain = np.nan
        equiv_stress_nominal = np.nan
        equiv_CTE = np.nan
        equiv_E_modulus = np.nan
    else:
        pcb_diagonal_length = np.sqrt(L_pcb**2 + W_pcb**2)
        equiv_strain = avg_diag_disp / pcb_diagonal_length
        equiv_stress_nominal = materials_data[pcb_tag]["E"] * equiv_strain
        equiv_CTE = equiv_strain / deltaT
        equiv_E_modulus = equiv_stress_nominal / equiv_strain if equiv_strain != 0 else 0.0

    stress_results_list.append(equiv_stress_nominal / 1e6 if not np.isnan(equiv_stress_nominal) else np.nan)
    CTE_results_list.append(equiv_CTE * 1e6 if not np.isnan(equiv_CTE) else np.nan)
    Eeq_results_list.append(equiv_E_modulus / 1e9 if not np.isnan(equiv_E_modulus) else np.nan)

    if rank == 0:
        print(f"  缺陷率为 {defect_rate*100:.1f}% 的结果：")
        print(f"    等效热膨胀系数（CTEeq）：{CTE_results_list[-1]:.2f} ppm/°C")
        print(f"    等效杨氏模量（Eeq）：{Eeq_results_list[-1]:.2f} GPa")

if rank == 0:
    print("\n--- 缺陷率变化下的等效参数汇总 ---")
    print("缺陷率(%)\t等效杨氏模量 Eeq (GPa)\t等效热膨胀系数 CTEeq (ppm/°C)")
    for r_val, e_val, cte_val in zip(rates, Eeq_results_list, CTE_results_list):
        e_str = f"{e_val:.2f}" if not np.isnan(e_val) else "N/A"
        cte_str = f"{cte_val:.2f}" if not np.isnan(cte_val) else "N/A"
        print(f"{r_val*100:.1f}\t\t{e_str}\t\t{cte_str}")
