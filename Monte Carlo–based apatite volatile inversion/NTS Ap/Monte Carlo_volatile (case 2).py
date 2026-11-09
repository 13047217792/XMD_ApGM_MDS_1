import string
import numpy as np
from Function import (X_OH_calc, KD_fluorine, calc_ap_chem, colored_curve, extract_csv_columns, extract_csv, make_cross_ring_marker,
                      make_ring_with_center_dot_marker, make_ring_cross_gap_marker)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.family'] = 'Times New Roman'

F_low_fluorine = np.linspace(0.01, 1, 23) # 记得倒转
F_low_fluorine = np.array(F_low_fluorine)
w_low_fluorine = 0
F_high_fluorine = np.linspace(0.025, 1, 40) # 记得倒转
F_high_fluorine = np.array(F_high_fluorine)
w_high_fluorine = np.linspace(0.10, 0.90, 33)
# m_H2O_saturation = 5.0 # 熔体水饱和含量

initial_melt_F_min, initial_melt_F_max = 0.001, 0.0500 # %
initial_melt_Cl_min, initial_melt_Cl_max = 0.01000, 0.5000 # %
D_F_undersaturation_min, D_F_undersaturation_max = 0.0, 2.0
D_Cl_undersaturation_min, D_Cl_undersaturation_max = 0.0, 1.0
F_cri_min, F_cri_max = 0.4, 0.6
D_F_mineral_min, D_F_mineral_max = 2.0, 6.0 # 最后阶段的含水矿物结晶 #(2, 8)为0，(2, 6)为1, (2, 5)为2, (2, 6)为3
D_Cl_mineral_min, D_Cl_mineral_max = 0.4, 1.0 # 0,1为初次，(0.4, 1.0)为1，(0.5, 1)为2, (0.5, 1.0)为3
D_H2O_mineral_min, D_H2O_mineral_max = 1.0, 2.0 # 0,1
D_F_fluid_min, D_F_fluid_max = 0, 20
D_Cl_fluid_min, D_Cl_fluid_max = 20, 150

fp = r"F:\论文写作\Revision\GR-apatite modelling\磷灰石成分蒙特卡洛模拟\Volatile simulation with post-recrystallization.csv"
fp1 = r"F:\论文写作\Revision\GR-apatite modelling\磷灰石成分蒙特卡洛模拟\DS & PTS 蒙特卡洛结果\Result_T_Composition_best.csv"
col_list = extract_csv_columns(fp)
# initial_parameter = extract_csv_columns(fp1)
parameter = extract_csv(fp1)
parameter = parameter.head(1)
N_mc = 100   # MC正演总次数，按需调整，5万、10万都可以
n = len(w_high_fluorine)
topN_params_high_fluorine=[]; topN_params_low_fluorine=[]
# topN_params = np.empty((N_mc * n, 10))

for i in range(N_mc):
    # 随机取一行
    # one_row = parameter.sample(n=1).iloc[0]
    one_row = parameter.iloc[0]
    initial_melt_F = one_row["F_initial"] # np.random.uniform(initial_melt_F_min, initial_melt_F_max)
    initial_melt_Cl = one_row["Cl_initial"] # np.random.uniform(initial_melt_Cl_min, initial_melt_Cl_max)
    D_F_undersaturation = one_row["D_F_mineral/melt"] # np.random.uniform(D_F_undersaturation_min, D_F_undersaturation_max)
    D_Cl_undersaturation = one_row["D_Cl_mineral/melt"] #np.random.uniform(D_Cl_undersaturation_min, D_Cl_undersaturation_max)
    F_cri = one_row["F_cri"] #np.random.uniform(F_cri_min, F_cri_max)
    D_F_mineral = np.random.uniform(D_F_mineral_min, D_F_mineral_max)
    D_Cl_mineral = np.random.uniform(D_Cl_mineral_min, D_Cl_mineral_max)
    D_F_fluid = one_row["D_F_fluid/melt"] # np.random.uniform(D_F_fluid_min, D_F_fluid_max)
    D_Cl_fluid = one_row["D_Cl_fluid/melt"] # np.random.uniform(D_Cl_fluid_min, D_Cl_fluid_max)
    m_H2O_saturation = one_row["m_H2O_saturation"]
    D_H2O_mineral = np.random.uniform(D_H2O_mineral_min, D_H2O_mineral_max)
    topN_params_low_fluorine.append([initial_melt_F, initial_melt_Cl,
                                     D_F_undersaturation, D_Cl_undersaturation, F_cri,
                                     D_F_mineral, D_Cl_mineral,
                                     D_F_fluid, D_Cl_fluid, 0.0, m_H2O_saturation, D_H2O_mineral])
    for w in w_high_fluorine:
        topN_params_high_fluorine.append([initial_melt_F, initial_melt_Cl,
                                     D_F_undersaturation, D_Cl_undersaturation, F_cri,
                                     D_F_mineral, D_Cl_mineral,
                                     D_F_fluid, D_Cl_fluid, w, m_H2O_saturation])
"""
for f0 in initial_melt_F:
    for cl0 in initial_melt_Cl:
        for d_f_undersaturation in D_F_undersaturation:
            for d_cl_undersaturation in D_Cl_undersaturation:
                for f_cri in F_cri:
                    for d_f_mineral in D_F_mineral:
                        for d_cl_mineral in D_Cl_mineral:
                            for d_f_fluid in D_F_fluid:
                                for d_cl_fluid in D_Cl_fluid:
                                    topN_params_low_fluorine.append([f0, cl0, d_f_undersaturation, d_cl_undersaturation, f_cri,
                                                                     d_f_mineral, d_cl_mineral,
                                                                     d_f_fluid, d_cl_fluid, 1.0])
                                    for w in w_high_fluorine:
                                        topN_params_high_fluorine.append([f0, cl0, d_f_undersaturation, d_cl_undersaturation, f_cri,
                                                                          d_f_mineral, d_cl_mineral,
                                                                      d_f_fluid, d_cl_fluid, w_high_fluorine])
"""
"""
m_H2O = np.full(len(F_high_fluorine), m_H2O_saturation)
X_H2O = m_H2O / 18.01528 / 1.70
X_OH_800 = X_OH_calc(1073.15, X_H2O); X_OH_900 = X_OH_calc(1173.15, X_H2O); X_OH_1000 = X_OH_calc(1273.15, X_H2O)
"""
XOH_high_fluorine, XF_high_fluorine, XCl_high_fluorine = 0.64998, 0.04944, 0.30058  # 高氟网状磷灰石中挥发份含量
XOH_low_fluorine, XF_low_fluorine, XCl_low_fluorine = 0.234189, 0.765811, 0.0 # 低氟网状磷灰石中挥发份含量
result_XF_XCl_Ap_800, result_XF_XOH_Ap_800, result_XCl_XOH_Ap_800 = [[] for _ in range(3)]
result_XF_XCl_Ap_900, result_XF_XOH_Ap_900, result_XCl_XOH_Ap_900 = [[] for _ in range(3)]
result_XF_XCl_Ap_1000, result_XF_XOH_Ap_1000, result_XCl_XOH_Ap_1000 = [[] for _ in range(3)]
for i, topN_param in enumerate(topN_params_high_fluorine):
    (p0, p1, p2, p3, p4,
     p5, p6,
     p7, p8,
     p9, p10) = topN_param
    m_H2O = np.full(len(F_high_fluorine), p10)
    # m_H2O = p10 + (6.0-p10) * F_high_fluorine
    m_F = (p0 * p4 ** (p2 - 1)) * F_high_fluorine ** (p5 * (1 - p9 * (p10 * 0.01)) + p7 * p9 * (p10 * 0.01) - 1)
    m_Cl = (p1 * p4 ** (p3 - 1)) * F_high_fluorine ** (p6 * (1 - p9 * (p10 * 0.01)) + p8 * p9 * (p10 * 0.01) - 1)
    X_F = m_F / 18.9984 / 1.70; X_Cl = m_Cl / 35.453 / 1.70
    X_H2O = m_H2O / 18.01528 / 1.70
    X_OH_800 = X_OH_calc(1073.15, X_H2O); X_OH_900 = X_OH_calc(1173.15, X_H2O); X_OH_1000 = X_OH_calc(1273.15, X_H2O)
    KD_OH_F_800, KD_OH_Cl_800 = KD_fluorine(1073.15, XOH_high_fluorine, XF_high_fluorine, XCl_high_fluorine)
    KD_OH_F_900, KD_OH_Cl_900 = KD_fluorine(1173.15, XOH_high_fluorine, XF_high_fluorine, XCl_high_fluorine)
    KD_OH_F_1000, KD_OH_Cl_1000 = KD_fluorine(1273.15, XOH_high_fluorine, XF_high_fluorine, XCl_high_fluorine)
    XF_XOH_melt_800 = X_F / X_OH_800; XCl_XOH_melt_800 = X_Cl / X_OH_800
    XF_XOH_melt_900 = X_F / X_OH_900; XCl_XOH_melt_900 = X_Cl / X_OH_900
    XF_XOH_melt_1000 = X_F / X_OH_1000; XCl_XOH_melt_1000 = X_Cl / X_OH_1000
    XF_XOH_Ap_800 = XF_XOH_melt_800 / KD_OH_F_800; XCl_XOH_Ap_800 = XCl_XOH_melt_800 / KD_OH_Cl_800; XF_XCl_Ap_800 = XF_XOH_Ap_800 / XCl_XOH_Ap_800
    XF_XOH_Ap_900 = XF_XOH_melt_900 / KD_OH_F_900; XCl_XOH_Ap_900 = XCl_XOH_melt_900 / KD_OH_Cl_900; XF_XCl_Ap_900 = XF_XOH_Ap_900 / XCl_XOH_Ap_900
    XF_XOH_Ap_1000 = XF_XOH_melt_1000 / KD_OH_F_1000; XCl_XOH_Ap_1000 = XCl_XOH_melt_1000 / KD_OH_Cl_1000;XF_XCl_Ap_1000 = XF_XOH_Ap_1000 / XCl_XOH_Ap_1000
    result_XF_XOH_Ap_800.append(XF_XOH_Ap_800); result_XCl_XOH_Ap_800.append(XCl_XOH_Ap_800); result_XF_XCl_Ap_800.append(XF_XCl_Ap_800)
    result_XF_XOH_Ap_900.append(XF_XOH_Ap_900); result_XCl_XOH_Ap_900.append(XCl_XOH_Ap_900); result_XF_XCl_Ap_900.append(XF_XCl_Ap_900)
    result_XF_XOH_Ap_1000.append(XF_XOH_Ap_1000); result_XCl_XOH_Ap_1000.append(XCl_XOH_Ap_1000);result_XF_XCl_Ap_1000.append(XF_XCl_Ap_1000)

XF_XOH_Ap = {"800": result_XF_XOH_Ap_800, "900": result_XF_XOH_Ap_900, "1000": result_XF_XOH_Ap_1000}
XCl_XOH_Ap_2 = {"800": result_XCl_XOH_Ap_800, "900": result_XCl_XOH_Ap_900, "1000": result_XCl_XOH_Ap_1000}
XF_XCl_Ap = {"800": result_XF_XCl_Ap_800, "900": result_XF_XCl_Ap_900, "1000": result_XF_XCl_Ap_1000}

"""
XCl_XOH_Ap_800_low_fluorine, m_H2O_Ap_800, m_F_Ap_800, m_Cl_Ap_800 = [[] for _ in range(4)]
XCl_XOH_Ap_900_low_fluorine, m_H2O_Ap_900, m_F_Ap_900, m_Cl_Ap_900 = [[] for _ in range(4)]
XCl_XOH_Ap_1000_low_fluorine, m_H2O_Ap_1000, m_F_Ap_1000, m_Cl_Ap_1000 = [[] for _ in range(4)]
# 计算氟低于检测限时的残余熔体体积分数
(low_fluorine_cri_800_1, low_fluorine_cri_800_2,
 low_fluorine_cri_900_1, low_fluorine_cri_900_2,
 low_fluorine_cri_1000_1, low_fluorine_cri_1000_2) =  [[] for _ in range(6)]
"""
# 温度(K)、标签
temp_list = [(1073.15, "800"),(1173.15, "900"),(1273.15, "1000")]
# 预先初始化存储列表
XCl_XOH_Ap_1 = {"800": [], "900": [], "1000": []}
m_H2O_Ap = {"800": [], "900": [], "1000": []}
m_Cl_Ap = {"800": [], "900": [], "1000": []}
m_F_Ap = {"800": [], "900": [], "1000": []}
low_fluorine_cri_1 = {"800":[], "900":[], "1000":[]} # <=0.05
low_fluorine_cri_2 = {"800":[], "900":[], "1000":[]} # <=0.02
for i, topN_param in enumerate(topN_params_low_fluorine):
    (p0, p1, p2, p3, p4,
     p5, p6,
     p7, p8,
     p9, p10, p11) = topN_param
    m_F = (p0 * p4 ** (p2 - 1)) * F_low_fluorine ** (p5 * (1 - p9 * (p10 * 0.01)) + p7 * p9 * (p10 * 0.01) - 1)
    m_Cl = (p1 * p4 ** (p3 - 1)) * F_low_fluorine ** (p6 * (1 - p9 * (p10 * 0.01)) + p8 * p9 * (p10 * 0.01) - 1)
    X_F = m_F / 18.9984 / 1.70; X_Cl = m_Cl / 35.453 / 1.70
    for T_K, tag in temp_list:
        xcl_xoh_ap, m_H2O, m_Cl, m_F = calc_ap_chem(T_K, X_F, X_Cl,
                                                    XOH_low_fluorine, XF_low_fluorine, XCl_low_fluorine,
                                                    F_low_fluorine, p10, p11)
        # 存入列表
        XCl_XOH_Ap_1[tag].append(xcl_xoh_ap)
        m_H2O_Ap[tag].append(m_H2O)
        m_Cl_Ap[tag].append(m_Cl)
        m_F_Ap[tag].append(m_F)
        # 筛选低F的索引
        idx_list1 = []  # 存储这条曲线满足 <=0.05 的j
        idx_list2 = []  # 存储这条曲线满足 <=0.02 的j
        for j, f_ap in enumerate(m_F_Ap[tag][i]):
            if f_ap >= 0.05:
                idx_list1.append(j)
                break
        for j, f_ap in enumerate(m_F_Ap[tag][i]):
            if f_ap >= 0.02:
                idx_list2.append(j)
                break
        low_fluorine_cri_1[tag].append(idx_list1)
        low_fluorine_cri_2[tag].append(idx_list2)
# 子图布局：4行3列
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 16), dpi=600)
fig.subplots_adjust(wspace=0.22, hspace=0.25)
temp_tags = ["800", "900", "1000"]
# 定义4行的x轴数据、x轴标签
row_info = [
    {"x_dict": m_Cl_Ap,    "xlabel": r"m$_{Cl}$ (wt.%)"},
    {"x_dict": m_H2O_Ap,   "xlabel": r"m$_{H_2O}$ (wt.%)"},
    {"x_dict": XF_XCl_Ap,  "xlabel": r"X$_{F}$/X$_{Cl}$"},
    {"x_dict": XF_XOH_Ap,  "xlabel": r"X$_{F}$/X$_{OH}$"},
]
y_label_text = r"X$_{Cl}$/X$_{OH}$"
color_map = {"800": ["#a5d9a3", "#2e7d32"], "900": ["#ffcc99", "#e65100"], "1000": ["#f8bbd0", "#c2185b"]}
norm1 = plt.Normalize(vmin=0.0, vmax=1.0)
norm2 = plt.Normalize(vmin=0.1, vmax=0.9)
# 循环遍历：列=温度，行=4种X变量
for col_idx, tag in enumerate(temp_tags):
    for row_idx, info in enumerate(row_info):
        ax = axes[row_idx, col_idx]
        # 获取当前子图的x、y二维数组
        x_data = info["x_dict"][tag]
        if row_idx < 2:
            y_data = XCl_XOH_Ap_1[tag]
        else:
            y_data = XCl_XOH_Ap_2[tag]
        # 绘制曲线和散点（蒙特卡洛采样曲线）
        # cmap = LinearSegmentedColormap.from_list("custom", color_map[tag])
        cmap = plt.cm.jet  # 最小值蓝色，最大值红色
        sm = []
        if row_idx == 0:
            # X轴固定刻度
            xticks0 = [2, 3, 4, 5, 6]
            # xticks1 = [0, 2000, 4000, 6000, 8000, 10000]
            ax.set_xticks(xticks0)
            ax.set_xlim(2, 6.95)
        elif row_idx == 1:
            # X轴固定对数刻度
            xticks1 = [0, 0.5, 1.0, 1.5, 2.0]
            # xticks1 = [0, 2000, 4000, 6000, 8000, 10000]
            ax.set_xticks(xticks1)
            ax.set_xlim(-0.005, 2.2)
        elif row_idx == 2:
            # X轴固定对数刻度
            xticks2 = [0.001, 0.01, 0.1, 1, 10]
            # xticks1 = [0, 2000, 4000, 6000, 8000, 10000]
            ax.set_xticks(xticks2)
            ax.xaxis.set_major_formatter(LogFormatterSciNotation())
            ax.set_xlim(0.001, 50)
        elif row_idx == 3:
            # X轴固定对数刻度
            xticks3 = [0.001, 0.01, 0.1, 1]
            # xticks1 = [0, 2000, 4000, 6000, 8000, 10000]
            ax.set_xticks(xticks3)
            ax.xaxis.set_major_formatter(LogFormatterSciNotation())
            ax.set_xlim(0.0005, 1)

        # Y轴设置，也放到外面
        if row_idx < 2:
            yticks = [1, 10, 100]
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(LogFormatterSciNotation())
            ax.set_ylim(0.2, 500)
            ax.tick_params(axis='both', labelsize=12)
        else:
            yticks = [0.001, 0.01, 0.1, 1]
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(LogFormatterSciNotation())
            ax.set_ylim(0.0005, 5)
            ax.tick_params(axis='both', labelsize=12)

        for i in range(len(x_data)):
            if row_idx < 2:
                # color1 = cmap(norm1(topN_params_low_fluorine[i][-2]))
                ax.plot(x_data[i], y_data[i], color="#4169E1", linewidth=0.5)
                """
                colored_curve(ax,
                              x_data[i],
                              y_data[i],
                              F_low_fluorine,
                              cmap, norm1)
                """
                ax.scatter(col_list[row_idx][:8], col_list[4][0:8], marker="o", facecolor="none", edgecolor="#FF0000",
                           zorder=10, s=50)
                ax.scatter(col_list[row_idx][8:14], col_list[4][8:14], marker="o", facecolor="#FF0000", edgecolor="#FF0000",
                            zorder=10, s=50)#0000FF
                ax.scatter(col_list[row_idx][14:30], col_list[4][14:30], marker="o", facecolor="none", edgecolor="#0000FF",
                           zorder=10, s=50)
                ax.scatter(col_list[row_idx][30:38], col_list[4][30:38], marker="o", facecolor="#0000FF",edgecolor="#0000FF",
                           zorder=10, s=50)

                x_data_1 = np.array(x_data[i])[low_fluorine_cri_1[tag][i]]; y_data_1 = np.array(y_data[i])[low_fluorine_cri_1[tag][i]]
                x_data_2 = np.array(x_data[i])[low_fluorine_cri_2[tag][i]]; y_data_2 = np.array(y_data[i])[low_fluorine_cri_2[tag][i]]
                # ax.scatter(x_data_1, y_data_1, c="#7030A0", s=4, alpha=1.0)
                # ax.scatter(x_data_2, y_data_2, c="#F088B0", s=4, alpha=1.0)
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(LogFormatterSciNotation())
                # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm1)

            else:
                color2 = cmap(norm2(topN_params_high_fluorine[i][-2]))
                # my_marker = make_cross_ring_marker(outer_r=1.0, inner_r=0.999, cross_width=0.0)
                my_marker = make_ring_cross_gap_marker(1.0, 0.999, 0.85, 0.001)
                ax.plot(x_data[i], y_data[i], color=color2, linewidth=0.5)

                ax.errorbar(np.asarray(col_list[row_idx][:36]), np.asarray(col_list[5][:36]),
                            xerr=np.asarray(col_list[row_idx + 4][:36]), yerr=np.asarray(col_list[8][:36]),
                             fmt="o", mfc="#FFFFFF", mec="#0000FF", ecolor="#0000FF", capsize=3, capthick=0.8,
                             elinewidth=0.8, markersize=7, zorder=10)

                ax.scatter(col_list[row_idx][:36], col_list[5][:36], marker= my_marker, facecolor="#FFFFFF", edgecolor="#0000FF",
                           zorder=12, s=50)
                ax.errorbar(col_list[row_idx][36:], col_list[5][36:],
                            xerr=col_list[row_idx + 4][36:], yerr=col_list[8][36:],
                            fmt="o", mfc="#FF69B4", mec="#FF69B4", ecolor="#FF69B4", capsize=3, capthick=0.8,
                            elinewidth=0.8, markersize=7, zorder=10)
                """
                ax.scatter(col_list[row_idx][34:], col_list[5][34:], marker="o", facecolor="#FF69B4", edgecolor="#FF69B4",
                           zorder=10, s=50)
                """
                ax.set_xscale("log")
                ax.xaxis.set_major_formatter(LogFormatterSciNotation())
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(LogFormatterSciNotation())
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
            if i % 50 == 0:
                print(f"绘图 {tag} row{row_idx}: {i}/{len(x_data)}")

        ax.tick_params(axis='both', labelsize=12)

        if row_idx > 1 and col_idx == 2:
            sm.set_array([])
            # 创建内嵌坐标轴：宽度占子图60%，高度5%，位置右上角
            bbox_to_anchor = ()
            if row_idx == 2:
                bbox_to_anchor = (0.91, 0.96)
            elif row_idx == 3:
                bbox_to_anchor = (0.91, 0.72)
            axins = inset_axes(ax,
                            width=0.1,  # 色条总长度（占ax宽度比例）
                            height=1.75,  # 色条厚度
                            loc="upper right",
                            bbox_to_anchor=bbox_to_anchor,  # 第一条y位置
                            bbox_transform=ax.transAxes,
                            borderpad=0
                            )  # 位置：upper right / upper left / lower right / lower left

            # 生成横向colorbar
            cbar = fig.colorbar(sm, cax=axins, orientation="vertical")
            # ========= 设置你想要的刻度和刻度值 =========
            tick_list = [0.1, 0.3, 0.5, 0.7, 0.9]  # 你自定义的刻度位置
            cbar.set_ticks(tick_list)
            cbar.set_ticklabels(["0.1", "0.3", "0.5", "0.7", "0.9"])  # 刻度显示文字
            # 刻度放在右侧（垂直色条默认右侧）；如果想放到左边：cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.yaxis.set_ticks_position("right")
            cbar.outline.set_visible(False)  # 隐藏边框
            cbar.ax.tick_params(labelsize=8)  # 刻度字号
            # 设置色条标签（示例）
            cbar.set_label(r"$\varphi$: Fluid escape fraction", fontsize=10, labelpad=2, va="center") # 文字不旋转，水平正向
            cbar.ax.yaxis.set_label_position('left')
        # 设置坐标轴标签
        ax.set_xlabel(info["xlabel"], fontfamily="Times New Roman", fontsize=16)
        ax.set_ylabel(y_label_text, fontfamily="Times New Roman", fontsize=16)
        """
        # 顶部子图设置标题（温度）
        if row_idx == 0:
            ax.set_title(f"{tag} °C", fontfamily="Times New Roman", fontsize=12)
        """
        # 刻度字体
        ax.tick_params(axis='both', labelsize=12)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily("Times New Roman")
handles = [
    Line2D([], [], color="#4169E1", lw=1.5, label=r"Fluid escape fraction $\varphi$ = 0")
]
axes[0,2].legend(handles=handles, loc="lower right", frameon=False, fontsize=10)
axes[1,2].legend(handles=handles, loc="upper right", frameon=False, fontsize=10)

total_ax = len(axes.flat)
letters = [f'({string.ascii_lowercase[k]})' for k in range(total_ax)]
# 遍历所有子图，左上角添加标签
nrow, ncol = axes.shape
# 生成列优先的索引：先竖着一列跑完，再下一列
idx_order = []
for icol in range(ncol):
    for irow in range(nrow):
        idx_order.append(irow + icol * nrow)
for new_idx, ax_pos in enumerate(idx_order):
    irow = ax_pos % nrow
    icol = ax_pos // nrow
    ax = axes[irow, icol]
    ax.text(
        0.02, 0.98,
        letters[new_idx],
        transform=ax.transAxes,
        fontsize=16,
        fontfamily='Times New Roman',
        va='top', ha='left',
        weight='bold'
    )
# plt.tight_layout()
plt.savefig('exsolved_fluid_precipitation_curves_100_33_new.png', dpi=600, bbox_inches='tight')