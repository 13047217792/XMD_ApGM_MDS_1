import numpy as np
from Function import T_crystallization, KD, X_OH_calc, extract_csv_columns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import itertools
from matplotlib.ticker import LogFormatterSciNotation
# ========== 全局字体设置 ==========
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

F = np.linspace(0.01, 1, 100) # 记得倒转
# m_H2O_saturation = 5.0 # 熔体水饱和含量

initial_melt_H2O_min, initial_melt_H2O_max = 0.5, 3.0 # %
initial_melt_F_min, initial_melt_F_max = 0.001, 0.1 # % 0.001, 0.05
initial_melt_Cl_min, initial_melt_Cl_max = 0.01000, 0.5 # %, 0.01, 0.5
D_H2O_undersaturation_min, D_H2O_undersaturation_max = 0, 0.5
D_F_undersaturation_min, D_F_undersaturation_max = 0, 2
D_Cl_undersaturation_min, D_Cl_undersaturation_max = 0, 1
D_F_saturation_min, D_F_saturation_max = 0, 20
D_Cl_saturation_min, D_Cl_saturation_max = 20, 150
m_H2O_saturation_mix,  m_H2O_saturation_max= 4.0, 6.0 # 熔体水饱和含量

N_mc = 50000   # MC正演总次数，按需调整，5万、10万都可以
topN_params = np.empty((N_mc, 9))
for i in range(N_mc):
    initial_melt_H2O = np.random.uniform(initial_melt_H2O_min, initial_melt_H2O_max)
    initial_melt_F = np.random.uniform(initial_melt_F_min, initial_melt_F_max)
    initial_melt_Cl = np.random.uniform(initial_melt_Cl_min, initial_melt_Cl_max)
    D_F_undersaturation = np.random.uniform(D_F_undersaturation_min, D_F_undersaturation_max)
    D_Cl_undersaturation = np.random.uniform(D_Cl_undersaturation_min, D_Cl_undersaturation_max)
    D_H2O_undersaturation = np.random.uniform(D_H2O_undersaturation_min, D_H2O_undersaturation_max)
    D_F_saturation = np.random.uniform(D_F_saturation_min, D_F_saturation_max)
    D_Cl_saturation = np.random.uniform(D_Cl_saturation_min, D_Cl_saturation_max)
    m_H2O_saturation = np.random.uniform(m_H2O_saturation_mix, m_H2O_saturation_max)
    params = np.array([initial_melt_H2O, initial_melt_F, initial_melt_Cl,
                        D_H2O_undersaturation, D_F_undersaturation, D_Cl_undersaturation,
                       D_F_saturation, D_Cl_saturation, m_H2O_saturation])
    topN_params[i] = params
"""
know_params = [[2.0, 0.008, 0.2, 0, 0.4, 0.2, 0.1, 125], [3.0, 0.015, 0.35, 0, 0.4, 0.2, 0.1, 125]]
topN_params = np.vstack([know_params, topN_params])
initial_melt_H2O = np.random.uniform(initial_melt_H2O_min, initial_melt_H2O_max, size=3)
initial_melt_F = np.random.uniform(initial_melt_F_min, initial_melt_F_max, size=6)
initial_melt_Cl = np.random.uniform(initial_melt_Cl_min, initial_melt_Cl_max, size=6)
D_F_undersaturation = np.random.uniform(D_F_undersaturation_min, D_F_undersaturation_max, size=3)
D_Cl_undersaturation = np.random.uniform(D_Cl_undersaturation_min, D_Cl_undersaturation_max, size=3)
D_H2O_undersaturation = np.random.uniform(D_H2O_undersaturation_min, D_H2O_undersaturation_max, size=3)
D_F_saturation = np.random.uniform(D_F_saturation_min, D_F_saturation_max, size=3)
D_Cl_saturation = np.random.uniform(D_Cl_saturation_min, D_Cl_saturation_max, size=6)
param_combinations = itertools.product(
    initial_melt_H2O, initial_melt_F, initial_melt_Cl,
    D_H2O_undersaturation, D_F_undersaturation, D_Cl_undersaturation,
    D_F_saturation, D_Cl_saturation
)
topN_params = np.array(list(param_combinations))
"""
fp = r"F:\论文写作\Revision\GR-apatite modelling\磷灰石成分蒙特卡洛模拟\Volatile simulation without post-recrystallization.csv"
col_list = extract_csv_columns(fp)
# 测试数据
X_data = np.array([
    col_list[0],
    col_list[2]
])
Y_data = np.array([
    col_list[1],
    col_list[3]
])
sigma = np.array([col_list[4],col_list[5], col_list[6], col_list[7]])
result_XF_XCl_Ap, result_XF_XOH_Ap, result_XCl_XOH_Ap = [np.zeros((len(topN_params), len(F))) for _ in range(3)]
rmse1 = np.zeros(len(topN_params)); rmse2 = np.zeros(len(topN_params))
for i, topN_param in enumerate(topN_params):
    (p0, p1, p2,
     p3, p4, p5,
     p6, p7) = topN_param[0:8]
    m_H2O_saturation = topN_param[8]
    F_cri = (m_H2O_saturation/p0)**(1/(p3-1)) # 水饱和的临界F值
    F_index = np.ceil((F_cri-0.01)/0.01) # F_cri的索引
    for j, f in enumerate(F):
        if f >= F_cri:  # 当流体未达到饱和时
            m_h2o = p0 * f ** (p3 - 1)
            m_f = p1 * f ** (p4 - 1)
            m_cl = p2 * f ** (p5 - 1)
        else: # 当流体达到饱和时
            m_h2o = m_H2O_saturation
            m_f = (p1 * F_cri ** (p4 - 1)) * (f / F_cri) ** ((-p4 + p6) * m_H2O_saturation * 0.01 + p4 - 1)
            m_cl = (p2 * F_cri ** (p5 - 1)) * (f / F_cri) ** ((-p5 + p7) * m_H2O_saturation * 0.01 + p5 - 1)

        X_H2O = m_h2o/18.01528/1.70; X_F = m_f/18.9984/1.70; X_Cl = m_cl/35.453/1.70 # 1.70为氧化物总的摩尔质量之和
        T = T_crystallization(f, m_h2o); X_OH = X_OH_calc(T, X_H2O)
        KD_OH_F, KD_OH_Cl = KD(f, m_h2o)
        XF_XCl_melt = X_Cl/X_F; XF_XOH_melt = X_F/X_OH; XCl_XOH_melt = X_Cl/X_OH
        XF_XOH_Ap = XF_XOH_melt/KD_OH_F
        XCl_XOH_Ap = XCl_XOH_melt / KD_OH_Cl
        XF_XCl_Ap = XF_XOH_Ap/XCl_XOH_Ap
        result_XF_XCl_Ap[i, j] = XF_XCl_Ap
        result_XF_XOH_Ap[i, j] = XF_XOH_Ap
        result_XCl_XOH_Ap[i, j] = XCl_XOH_Ap

    """
    X_data_0_new = np.delete(X_data[0], [33, 36, 38, 41])
    Y_data_0_new = np.delete(Y_data[0], [33, 36, 38, 41])
    """
    X_data_0_new = np.delete(X_data[0], [36])
    dist1_sum = 0.0
    for xi, yi in zip(X_data[0], Y_data[0]):
        dx = result_XF_XCl_Ap[i,:]-xi
        dy = result_XCl_XOH_Ap[i,:]-yi
        dist_sq_array = np.sqrt(dx ** 2 + dy ** 2)
        min_dist1 = np.min(dist_sq_array)
        dist1_sum += min_dist1
    rmse1[i] = np.log10(dist1_sum/len(X_data[0]))

    #删除相对误差大于1的数据
    X_data_1_new = np.delete(X_data[1], [33, 38, 41])
    Y_data_1_new = np.delete(Y_data[1], [33, 38, 41])
    dist2_sum = 0.0
    for xi, yi in zip(X_data[1], Y_data[1]):
        dx = result_XF_XOH_Ap[i,:] - xi
        dy = result_XCl_XOH_Ap[i,:] - yi
        dist_sq_array = np.sqrt(dx ** 2 + dy ** 2)
        min_dist2 = np.min(dist_sq_array)
        dist2_sum += min_dist2
    rmse2[i] = np.log10(dist2_sum/len(X_data[1]))

rmse = rmse1 + rmse2
# 按列拼接（水平拼接）: 结果 shape (3, 4)
result = np.column_stack([topN_params, rmse1, rmse2, rmse])
# 保存为 csv（无表头、空格分隔）
np.savetxt('分阶段评估\Result_T_Composition_1.csv', result, delimiter=',', fmt='%.6f', header='H2O_initial, F_initial, Cl_initial, '
                                                                   'D_H2O_mineral/melt,D_F_mineral/melt, D_Cl_mineral/melt, '
                                                                   'D_F_fluid/melt, D_Cl_fluid/melt, m_H2O_saturation,'
                                                                   'rmse1, rmse2, rmse', comments='')
"""
XF_XCl_rmse1 = np.column_stack([result_XF_XCl_Ap, rmse1])
XCl_XOH_rmse1 = np.column_stack([result_XCl_XOH_Ap, rmse1])
np.savetxt('XF_XCl.csv', XF_XCl_rmse1, delimiter=',', fmt='%.6f', comments='')
np.savetxt('XCl_XOH.csv', XCl_XOH_rmse1, delimiter=',', fmt='%.6f', comments='')
"""
# ---------------------- 创建1行2列子图（沿X轴并排拼接） ----------------------
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 12), dpi=600)
# 单独设置子图竖直间距
fig.subplots_adjust(hspace=0.15)
cmap = plt.cm.jet   # 最小值蓝色，最大值红色
# cmap = LinearSegmentedColormap.from_list("blue_red_custom", ["#0000FF", "#FF0000"])

# ========== 第一组：基于 rmse1 从小到大排序 ==========
# 获取rmse1从小到大排序的索引
idx1 = np.argsort(rmse1)
# 使用索引同步排序其他数组
rmse1_sorted = rmse1[idx1]
result_XF_XCl_Ap_1 = result_XF_XCl_Ap[idx1]
result_XCl_XOH_Ap_1 = result_XCl_XOH_Ap[idx1]
# ========== 第二组：基于 rmse2 从小到大排序 ==========
# 获取rmse2从小到大排序的索引
idx2 = np.argsort(rmse2)
rmse2_sorted = rmse2[idx2]
result_XF_XOH_Ap_2 = result_XF_XOH_Ap[idx2]
result_XCl_XOH_Ap_2 = result_XCl_XOH_Ap[idx2]

norm_rmse1 = plt.Normalize(vmin=np.min(rmse1_sorted[:50000]), vmax=np.max(rmse1_sorted[:50000]))
norm_rmse2 = plt.Normalize(vmin=np.min(rmse2_sorted[:50000]), vmax=np.max(rmse2_sorted[:50000]))
N_curve = len(rmse1_sorted[:50000])
# ============ 第一张子图 ============
for i in range(N_curve-1, -1, -1):
    x1 = result_XF_XCl_Ap_1[i, :]
    y1 = result_XCl_XOH_Ap_1[i, :]
    color1 = cmap(norm_rmse1(rmse1_sorted[i]))
    ax1.plot(x1, y1, color=color1, linewidth=0.25)
# ============ 第二张子图 ============
for i in range(N_curve-1, -1, -1):
    x2 = result_XF_XOH_Ap_2[i, :]
    y2 = result_XCl_XOH_Ap_2[i, :]
    color2 = cmap(norm_rmse2(rmse2_sorted[i]))
    ax2.plot(x2, y2, color=color2, linewidth=0.25)
"""
# 全局统一RMSE颜色标尺
vmin_global = min(np.min(rmse1), np.min(rmse2))
vmax_global = max(np.max(rmse1), np.max(rmse2))
norm_global = plt.Normalize(vmin=vmin_global, vmax=vmax_global)
N_curve = result_XF_XCl_Ap.shape[0]
# ============ 第一张子图 ============
for i in range(N_curve):
    x1 = result_XF_XCl_Ap[i, :]
    y1 = result_XCl_XOH_Ap[i, :]
    color1 = cmap(norm_global(rmse1[i]))
    ax1.plot(x1, y1, color=color1, linewidth=1.3)
# ============ 第二张子图 ============
for i in range(N_curve):
    x2 = result_XF_XOH_Ap[i, :]
    y2 = result_XCl_XOH_Ap[i, :]
    color2 = cmap(norm_global(rmse2[i]))
    ax2.plot(x2, y2, color=color2, linewidth=1.3)
"""
# 设置双对数坐标轴
ax1.set_xscale("log")
ax1.set_yscale("log")
ax2.set_xscale("log")
ax2.set_yscale("log")

# X轴固定对数刻度
xticks1 = [0.01, 0.1, 1, 10, 100, 1000, 10000]
# xticks1 = [0, 2000, 4000, 6000, 8000, 10000]
ax1.set_xticks(xticks1)
ax1.xaxis.set_major_formatter(LogFormatterSciNotation())
ax1.set_xlim(0.005, 10000)
# Y轴固定对数刻度
yticks1 = [0.002, 0.01, 0.1, 1, 10, 30]
ax1.set_yticks(yticks1)
ax1.yaxis.set_major_formatter(LogFormatterSciNotation())
ax1.set_ylim(0.0005, 30)
ax1.tick_params(axis='both', labelsize=12)

# X轴固定对数刻度
xticks2 = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
# xticks2 = [0, 200, 400, 600, 800, 1000]
ax2.set_xticks(xticks2)
ax2.xaxis.set_major_formatter(LogFormatterSciNotation())
ax2.set_xlim(0.001, 1000)
# Y轴固定对数刻度
yticks2 = [0.001, 0.01, 0.1, 1, 10]
ax2.set_yticks(yticks2)
ax2.yaxis.set_major_formatter(LogFormatterSciNotation())
ax2.set_ylim(0.0005, 30)
ax2.tick_params(axis='both', labelsize=12)

# ====== 在ax2内部手动创建内嵌色条坐标轴 ======
# 参数 [left, bottom, width, height]，坐标为相对于整个画布的比例
# 靠近ax2右侧Y轴，放在子图内部
# cax2 = ax2.add_axes([0.90, 0.2, 0.05, 0.70]) # [x,y,width,height]

sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=norm_rmse1)
sm1.set_array([])
# 创建内嵌坐标轴：宽度占子图60%，高度5%，位置右上角
axins1 = inset_axes(ax1,
                    width=2.0,  # 色条总长度（占ax宽度比例）
                    height=0.1,  # 色条厚度
                    loc="upper right",
                    bbox_to_anchor=(0.89, 0.95),  # 第一条y位置
                    bbox_transform=ax1.transAxes,
                    borderpad=0
                    )  # 位置：upper right / upper left / lower right / lower left
# 生成横向colorbar
cbar1 = fig.colorbar(sm1, cax=axins1, orientation="horizontal")
cbar1.ax.xaxis.set_ticks_position("bottom")  # 刻度放在色条下方，论文图好看
cbar1.ax.set_xticks([])
cbar1.outline.set_visible(False)
# 设置色条标签（示例）
cbar1.set_label("log RMSE", fontsize=10)
cbar1.ax.xaxis.set_label_position('top')

sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm_rmse2)
sm2.set_array([])
axins2 = inset_axes(ax2,
                    width=2.0,  # 色条总长度（占ax宽度比例）
                    height=0.1,  # 色条厚度
                    loc="upper right",
                    bbox_to_anchor=(0.89, 0.95),  # 第一条y位置
                    bbox_transform=ax2.transAxes,
                    borderpad=0
                    )  # 位置：upper right / upper left / lower right / lower left
cbar2 = fig.colorbar(sm2, cax=axins2, orientation="horizontal")
cbar2.ax.xaxis.set_ticks_position("bottom")  # 刻度放在色条下方，论文图好看
cbar2.ax.set_xticks([])
cbar2.outline.set_visible(False)
# 设置色条标签（示例）
cbar2.set_label("log RMSE", fontsize=10)
cbar2.ax.xaxis.set_label_position('top')

ax1.set_xlabel(r"X$_\mathrm{F}$/X$_\mathrm{Cl}$", fontsize=16)
ax1.set_ylabel(r"X$_\mathrm{Cl}$/X$_\mathrm{OH}$", fontsize=16)
ax2.set_xlabel(r"X$_\mathrm{F}$/X$_\mathrm{OH}$", fontsize=16)
ax2.set_ylabel(r"X$_\mathrm{Cl}$/X$_\mathrm{OH}$", fontsize=16)

# 第0组E-DS前10个点实心方形,≥10所有点空心方形（仅边框），偏深棕黄
ax1.errorbar(X_data[0][:13], Y_data[0][:13],
             xerr=sigma[0][:13],yerr=sigma[1][:13],
             fmt="s",mfc="#C89038",mec="#C89038",ecolor="#C89038",capsize=3,capthick=0.8,elinewidth=0.8,markersize=7,zorder=10)
ax1.errorbar(X_data[0][13:31], Y_data[0][13:31],
            xerr=sigma[0][13:31],yerr=sigma[1][13:31],
            fmt="s",mfc="none",mec="#C89038",ecolor="#C89038",capsize=3,capthick=0.8,elinewidth=0.8,markersize=7,zorder=10)
ax1.scatter(
    X_data[0][13:31], Y_data[0][13:31],
    marker="s", s=49, facecolor="#FFFFFF", edgecolor="#C89038", zorder=12
)
# ======== 第1组E-PTS橙色 #FF8822，前10实心菱形，后面空心菱形 ========
ax1.errorbar(X_data[0][31:42], Y_data[0][31:42],
            xerr=sigma[0][31:42],yerr=sigma[1][31:42],
            fmt="D",mfc="#FF8822",mec="#FF8822",ecolor="#FF8822",capsize=3,capthick=0.8,elinewidth=0.8,markersize=7,zorder=10)
ax1.errorbar(X_data[0][42:69], Y_data[0][42:69],
            xerr=sigma[0][42:69],yerr=sigma[1][42:69],
            fmt="D",mfc="none",mec="#FF8822",ecolor="#FF8822",capsize=3,capthick=0.8,elinewidth=0.8,markersize=7,zorder=10)
ax1.scatter(
    X_data[0][42:69], Y_data[0][42:69],
    marker="D", s=49, facecolor="#FFFFFF", edgecolor="#FF8822", zorder=12
)
# ======== 第2组W-DS绿色 #FF8822 ========
ax1.errorbar(X_data[0][69:], Y_data[0][69:],
            xerr=sigma[0][69:], yerr=sigma[1][69:],
            fmt="s",mfc="#22aa22",mec="#22aa22",ecolor="#22aa22",capsize=3,capthick=0.8,elinewidth=0.8,markersize=7,zorder=10)


ax2.errorbar(X_data[1][:13], Y_data[1][:13],
            xerr=sigma[2][:13],yerr=sigma[3][:13],
            fmt="s",mfc="#C89038",mec="#C89038",ecolor="#C89038",capsize=3,capthick=0.8,elinewidth=0.8,markersize=7,zorder=10)
ax2.errorbar(X_data[1][13:31], Y_data[1][13:31],
            xerr=sigma[2][13:31],yerr=sigma[3][13:31],
            fmt="s",mfc="none",mec="#C89038",ecolor="#C89038",capsize=3,capthick=0.8,elinewidth=0.8,markersize=7,zorder=10)
ax2.scatter(
    X_data[1][13:31], Y_data[1][13:31],
    marker="s", s=49, facecolor="#FFFFFF", edgecolor="#C89038", zorder=12
)
ax2.errorbar(X_data[1][31:42], Y_data[1][31:42],
            xerr=sigma[2][31:42],yerr=sigma[3][31:42],
            fmt="D",mfc="#FF8822",mec="#FF8822",ecolor="#FF8822",capsize=3,capthick=0.8,elinewidth=0.8,markersize=7,zorder=10)
ax2.errorbar(X_data[1][42:69], Y_data[1][42:69],
            xerr=sigma[2][42:69],yerr=sigma[3][42:69],
            fmt="D",mfc="none",mec="#FF8822",ecolor="#FF8822",capsize=3,capthick=0.8,elinewidth=0.8,markersize=7,zorder=10)
ax2.scatter(
    X_data[1][42:69], Y_data[1][42:69],
    marker="D", s=49, facecolor="#FFFFFF", edgecolor="#FF8822", zorder=12
)
ax2.errorbar(X_data[1][69:], Y_data[1][69:],
            xerr=sigma[2][69:], yerr=sigma[3][69:],
            fmt="s",mfc="#22aa22",mec="#22aa22",ecolor="#22aa22",capsize=3,capthick=0.8,elinewidth=0.8,markersize=7,zorder=10)

# plt.tight_layout()
plt.savefig('分阶段评估\monte_carlo_curves_T_Composition_1.png', dpi=600, bbox_inches='tight')
# plt.savefig('cartesian_grid_sweep_curves.png', dpi=600, bbox_inches='tight')
# plt.show()