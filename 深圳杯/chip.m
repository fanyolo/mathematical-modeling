clc; clear;

%% -------------------- 材料 & 几何参数 ------------------------
E_PCB = 28.6e9;    CTE_PCB = 17e-6;    t_PCB = 1.6e-3;
E_sub = 73.3e9;    CTE_sub = 21e-6;    t_sub = 0.8e-3;
E_mold= 11.7e9;    CTE_mold= 23e-6;    t_mold= 1.17e-3;
E_chip=130e9;      CTE_chip= 2.6e-6;   t_chip= 0.2e-3;

t_pack = t_sub + t_mold + t_chip;
E_pack = (E_sub*t_sub + E_mold*t_mold + E_chip*t_chip) / t_pack;
CTE_pack = (E_sub*t_sub*CTE_sub + E_mold*t_mold*CTE_mold + E_chip*t_chip*CTE_chip) / ...
           (E_sub*t_sub + E_mold*t_mold + E_chip*t_chip);

r_ball = 0.35e-3;  A_ball = pi * r_ball^2;
H_ball = 0.5e-3;   nu_s  = 0.35;
G_solder = 50e9 / (2*(1+nu_s));
k_ball = G_solder * A_ball / H_ball;

L_half = sqrt((26e-3/2)^2 * 2);
A_unit = 1e-3;
K_PCB = E_PCB * (t_PCB * A_unit) / L_half;
K_pack= E_pack* (t_pack * A_unit) / L_half;

n_vals = 0:4;
Eeq = zeros(size(n_vals));
CTEeq = zeros(size(n_vals));

for idx = 1:length(n_vals)
    n = n_vals(idx);
    n_balls = 4 - n;
    if n_balls > 0
        K_cluster = n_balls * k_ball;
        K_total = 1 / (1/K_PCB + 1/K_pack + 1/K_cluster);
    else
        K_total = 1 / (1/K_PCB + 1/K_pack + 1/1e-12);
    end
    L_total = 2 * L_half;
    A_total = (t_PCB + t_pack) * A_unit;
    Eeq(idx) = K_total * L_total / A_total;

    base_CTE = (E_PCB*t_PCB*CTE_PCB + E_pack*t_pack*CTE_pack) / ...
               (E_PCB*t_PCB + E_pack*t_pack);
    CTEeq(idx) = base_CTE + (CTE_pack - base_CTE) * (n / 4);
end

% 将结果组织为表格（Table）形式
ResultTable = table( ...
    n_vals.', ...
    Eeq(:)/1e9, ...
    CTEeq(:)*1e6, ...
    'VariableNames', {'缺失焊球数', '等效模量_GPa', '等效CTE_ppm_per_C'});

% 显示表格
disp('焊球缺陷下等效参数变化表：');
disp(ResultTable);


%% -------------------- 图1：焊球缺陷图（2x2） ------------------------
figure('Name','图2：角部焊球缺陷示意图','Position',[100 100 1000 250]);
for idx = 1:5
    n = idx-1;
    balls = ones(2,2);
    if n > 0
        rng(idx); id = randperm(4,n); balls(id) = 0;
    end
    subplot(1,5,idx);
    imagesc(balls); colormap(gray); axis equal off;
    title(['缺失球数 = ' num2str(n)]);
end
sgtitle('角部 2×2 焊球阵列缺陷示意');

%% -------------------- 图2：最大缺陷下 2D 热变形（矢量图） ------------------------
figure('Name','图3：热膨胀矢量场（最大缺陷）','Position',[100 100 600 500]);
[x, y] = meshgrid(0:1:26, 0:1:26);
alpha_ref = CTEeq(end); deltaT = 100;
u = alpha_ref * deltaT * (x - 13); v = alpha_ref * deltaT * (y - 13);
quiver(x, y, u*1e6, v*1e6, 'r');  % 单位μm
axis equal; grid on;
xlabel('X (mm)'); ylabel('Y (mm)');
title('最大缺陷条件下热膨胀方向矢量图');

%% -------------------- 图3：仅展示缺失 2 个焊球下的热变形 ------------------------
figure('Name','图4：热膨胀变形（缺失2个焊球）','Position',[100 100 800 600]);

% 固定缺失焊球数为2
n = 2;  % 缺失球数
pkg_len = 26;
[Xs, Ys] = meshgrid(linspace(0, pkg_len, 50), linspace(0, pkg_len, 50));

% 计算等效CTE
base_CTE = (E_PCB*t_PCB*CTE_PCB + E_pack*t_pack*CTE_pack) / ...
           (E_PCB*t_PCB + E_pack*t_pack);
CTE_now = base_CTE + (CTE_pack - base_CTE) * (n / 4);  % 线性插值

% 热变形模型
deltaT = 100;  % °C 升温
Xc = Xs - pkg_len/2;
Yc = Ys - pkg_len/2;
R = sqrt(Xc.^2 + Yc.^2);
U = CTE_now * deltaT * R;  % 径向膨胀位移

% surf 图绘制
surf(Xs, Ys, U*1e6);  % 单位 μm
shading interp; colormap turbo; colorbar;
title('缺失 2 个角部焊球时的热膨胀变形（surf）');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('\DeltaZ (μm)');
axis([0 pkg_len 0 pkg_len 0 max(U(:)*1e6)*1.2]);
view(35, 35); grid on;
