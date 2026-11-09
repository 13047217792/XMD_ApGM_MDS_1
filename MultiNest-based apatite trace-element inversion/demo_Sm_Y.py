import numpy as np
import matplotlib
matplotlib.use('Agg') # 非交互式后端，不依赖本地端口，只保存图片
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ["DejaVu Serif"]
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
import pymultinest
from Function import extract_csv_columns, make_ring_cross_gap_marker
import Inversion_Sm_Y as mi
"""
# Eastern_No_NTS
unknowns = [{'Sm_parent':['uni',[2.0,5.0]], 'Y_parent':['uni',[15,25]],
             'Sm_wallrock':['uni',[2.0,8.0]], 'Y_wallrock':['uni',[20,40]],
             'DSm_mineral/melt_0':['uni',[0.028,0.3]],'DY_mineral/melt_0':['uni',[0.033,0.2]], 'r':['uni',[0.0,0.4]]}, # [0.001, 0.3], [0.003, 1.0]
            {'DSm_mineral/melt_1':['uni',[0.028,0.67]], 'DY_mineral/melt_1':['uni',[0.033,1.27]], 'F0':['uni',[0.7,0.9]]}, # [0.001, 0.6], [0.003, 1.0]
            {'DSm_mineral/melt_2':['uni',[0.028,0.445]],'DY_mineral/melt_2':['uni',[0.033,1.27]], 'F1':['uni',[0.3,0.6]]}, # [0.05, 1.0], [0.01, 1.0] # DSm有进一步约束
            {'DSm_mineral/melt_3':['uni',[0.002,3.0]],'DY_mineral/melt_3':['uni',[0.04,4.0]], 'F2':['uni',[0.3,0.6]]},
            {'DSm_mineral/melt_4':['uni',[0.002,8.0]],'DY_mineral/melt_4':['uni',[0.04,3.0]], 'F3':['uni',[0.1,0.3]]}, #对DY进行了修改
            {'DSm_ap/melt':['uni',[25,65]],'DY_ap/melt':['uni',[15,40]]}]

# Western_No_NTS
unknowns = [{'Sm_parent':['uni',[2.0,5.0]], 'Y_parent':['uni',[15,25]],
             'Sm_wallrock':['uni',[2.0,10.0]], 'Y_wallrock':['uni',[20,40]],
             'DSm_mineral/melt_0':['uni',[0.028,0.3]],'DY_mineral/melt_0':['uni',[0.033,0.2]], 'r': ['uni',[0,0.4]]},
            {'DSm_mineral/melt_1': ['uni', [0.028, 0.67]], 'DY_mineral/melt_1': ['uni', [0.033, 1.27]], 'F0': ['uni', [0.7, 0.9]]},  # [0.001, 0.6], [0.003, 1.0]
            {'DSm_mineral/melt_2': ['uni', [0.002, 3.0]], 'DY_mineral/melt_2': ['uni', [0.04, 4.0]], 'F1': ['uni', [0.3, 0.6]]},
            {'DSm_ap/melt':['uni',[25,60]],'DY_ap/melt':['uni',[13,37]]}]


# Eastern_NTS
unknowns = [{'Sm_parent':['uni',[2.0,5.0]], 'Y_parent':['uni',[15,25]],
             'Sm_wallrock':['uni',[2.0,10.0]], 'Y_wallrock':['uni',[20,40]],
             'DSm_mineral/melt_0':['uni',[0.028,0.3]],'DY_mineral/melt_0':['uni',[0.033,0.2]], 'r': ['uni',[0,0.4]]},
            {'DSm_mineral/melt_1': ['uni', [0.028, 0.67]], 'DY_mineral/melt_1': ['uni', [0.033, 1.27]], 'F0': ['uni', [0.7, 0.9]]},  # [0.001, 0.6], [0.003, 1.0]
            {'DSm_mineral/melt_2': ['uni', [0.002, 3.0]], 'DY_mineral/melt_2': ['uni', [0.04, 4.0]], 'F1': ['uni', [0.3, 0.6]]},
            {'DSm_ap/melt':['uni',[50,130]],'DY_ap/melt':['uni',[30,80]]}] # 'DSr_mineral/melt_1': ['uni', [0.001, 0.1]]基于Eastern_No_NTS反演结果进行调整
"""
# Western_NTS
unknowns_Cu_NTS = [{'Sm_parent':['uni',[2.0,5.0]], 'Y_parent':['uni',[15,25]],
             'Sm_wallrock':['uni',[2.0,10.0]], 'Y_wallrock':['uni',[20,40]],
             'DSm_mineral/melt_0':['uni',[0.028,0.3]],'DY_mineral/melt_0':['uni',[0.033,0.2]], 'r': ['uni',[0,0.4]]},
            {'DSm_mineral/melt_CuNTS_1': ['uni', [0.028, 0.67]], 'DY_mineral/melt_CuNTS_1': ['uni', [0.033, 1.27]],'F_CuNTS_0': ['uni', [0.5, 0.7]]},
            {'DSm_mineral/melt_CuNTS_2': ['uni', [0.002, 3.0]], 'DY_mineral/melt_CuNTS_2': ['uni', [0.04, 4.0]], 'F_CuNTS_1': ['uni', [0.3, 0.6]]},
            {'DSm_ap/melt_CuNTS':['uni',[40,90]],'DY_ap/melt_CuNTS':['uni',[25,65]]}]
unknowns = [{'Sm_parent':['uni',[2.0,5.0]], 'Y_parent':['uni',[15,25]],
             'Sm_wallrock':['uni',[2.0,10.0]], 'Y_wallrock':['uni',[20,40]],
             'DSm_mineral/melt_0':['uni',[0.028,0.3]],'DY_mineral/melt_0':['uni',[0.033,0.2]], 'r': ['uni',[0,0.4]]},
            {'DSm_mineral/melt_NTS_1': ['uni', [0.028, 0.67]], 'DY_mineral/melt_NTS_1': ['uni', [0.033, 1.27]], 'F_NTS_0': ['uni', [0.7, 0.9]]}, # [0.001, 0.6], [0.003, 1.0]
            {'DSm_mineral/melt_NTS_2': ['uni', [0.002, 3.0]], 'DY_mineral/melt_NTS_2': ['uni', [0.04, 4.0]], 'F_NTS_1': ['uni', [0.3, 0.6]]},
            {'DSm_ap/melt_NTS':['uni',[50,130]],'DY_ap/melt_NTS':['uni',[30,80]]}] # 'DSr_mineral/melt_1': ['uni', [0.001, 0.1]]基于Eastern_No_NTS反演结果进行调整

#==================== 1. MultiNest 读取结果 (原有代码) ====================
knowns = ''
fp = "/mnt/f/论文写作/Revision/GR-apatite modelling/Inversion_Sm_Y/Sm_Y.csv"
col_list = extract_csv_columns(fp)
# Create the inversion object
inv = mi.inversion(col_list,knowns,unknowns_Cu_NTS,type='Cu_NTS', name='Western') # 改变磷灰石类型
# Run the inversion
inv.run_multinest()
n_params = 0       # 根据你实际参数数量修改
for d in unknowns:
    n_params += len(d)
# OUTPUT_PREFIX = r"/mnt/f/论文写作/Revision/GR-apatite modelling/磷灰石成分蒙特卡洛模拟/Eastern/Eastern_No_NTS"
OUTPUT_PREFIX = inv.name + '/' + inv.name + '_' + inv.type
a = pymultinest.Analyzer(n_params, outputfiles_basename=OUTPUT_PREFIX)
samples = a.get_data()
weights = samples[:,0]
logL_vals = samples[:, 1]
param_samples = samples[:, 2:]
# 按对数似然从大到小排序
sort_idx = np.argsort(-weights)
sorted_params = param_samples[sort_idx, :]
TOP_N = 300 # 取前N个最优样本，自行修改
topN_params = sorted_params[:TOP_N, :]
# sorted_params 来自你的代码，形状应为 (n_samples, 21)

inv1 = mi.inversion(col_list,knowns,unknowns,type='NTS', name='Western')
# Run the inversion
inv1.run_multinest()
# inv1.run_multinest()
n_params1 = 0
for d in unknowns:
    n_params1 += len(d)
# OUTPUT_PREFIX = r"/mnt/f/论文写作/Revision/GR-apatite modelling/磷灰石成分蒙特卡洛模拟/Eastern/Eastern_No_NTS"
OUTPUT_PREFIX1 = inv1.name + '/' + inv1.name + '_' + inv1.type
a1 = pymultinest.Analyzer(n_params1, outputfiles_basename=OUTPUT_PREFIX1)
samples1 = a1.get_data()
weights1 = samples1[:,0]
logL_vals1 = samples1[:, 1]
param_samples1 = samples1[:, 2:]
# 按对数似然从大到小排序
sort_idx1 = np.argsort(-weights1)
sorted_params1 = param_samples1[sort_idx1, :]
topN_params1 = sorted_params1[:TOP_N, :]
topN_params = np.hstack([topN_params, topN_params1])

if inv.type == 'No_NTS' and inv.name == 'Eastern':
    column_names = [
        'Sm_parent', 'Y_parent', 'Sm_wallrock', 'Y_wallrock',
        'DSm_mineral/melt_0', 'DY_mineral/melt_0', 'r',
        'DSm_mineral/melt_1', 'DY_mineral/melt_1', 'F0',
        'DSm_mineral/melt_2', 'DY_mineral/melt_2', 'F1',
        'DSm_mineral/melt_3', 'DY_mineral/melt_3', 'F2',
        'DSm_mineral/melt_4', 'DY_mineral/melt_4', 'F3',
        'DSm_ap/melt', 'DY_ap/melt']
    path = r"/mnt/f/论文写作/Revision/GR-apatite modelling/Inversion_Sm_Y/Eastern/Eastern_No_NTS_Sm_Y.xlsx"
elif (inv.type == 'No_NTS' and inv.name == 'Western') or (inv.type == 'NTS' and inv.name == 'Eastern'):
    column_names = ['Sm_parent', 'Y_parent', 'Sm_wallrock', 'Y_wallrock',
        'DSm_mineral/melt_0', 'DY_mineral/melt_0', 'r',
        'DSm_mineral/melt_1', 'DY_mineral/melt_1', 'F0',
        'DSm_mineral/melt_2', 'DY_mineral/melt_2', 'F1',
        'DSm_ap/melt', 'DY_ap/melt']
    if inv.type == 'No_NTS' and inv.name == 'Western':
        path = r"/mnt/f/论文写作/Revision/GR-apatite modelling/Inversion_Sm_Y/Western/Western_No_NTS_Sm_Y.xlsx"
    else:
        path = r"/mnt/f/论文写作/Revision/GR-apatite modelling/Inversion_Sm_Y/Eastern/Eastern_NTS_Sm_Y.xlsx"
else:
    column_names = ['Sm_parent', 'Y_parent', 'Sm_wallrock', 'Y_wallrock',
                    'DSm_mineral/melt_0', 'DY_mineral/melt_0', 'r',
                    'DSm_mineral/melt_CuNTS_1', 'DY_mineral/melt_CuNTS_1', 'F_CuNTS_0',
                    'DSm_mineral/melt_CuNTS_2', 'DY_mineral/melt_CuNTS_2', 'F_PTS_CuNTS_1',
                    'DSmap/melt_CuNTS', 'DY_ap/melt_CuNTS',
                    'Sm_parent', 'Y_parent', 'Sm_wallrock', 'Y_wallrock',
                    'DSm_mineral/melt_0', 'DY_mineral/melt_0', 'r',
                    'DSm_mineral/melt_NTS_1', 'DY_mineral/melt_NTS_1', 'F_NTS_0',
                    'DSm_mineral/melt_NTS_2', 'DY_mineral/melt_NTS_2', 'F_NTS_1',
                    'DSm_ap/melt_NTS', 'DY_ap/melt_NTS'
                    ]
    path = r"/mnt/f/论文写作/Revision/GR-apatite modelling/Inversion_Sm_Y/Western/Western_NTS_Sm_Y.xlsx"
# 断言：确认列数匹配
assert topN_params.shape[1] == len(column_names), \
    f"列数不匹配: 矩阵有 {topN_params.shape[1]} 列, 表头有 {len(column_names)} 列"
# 转成 DataFrame 并保存
df = pd.DataFrame(topN_params, columns=column_names)
df.to_excel(path, index=False)
print(f"已保存，形状: {df.shape}, 行数: {df.shape[0]}, 列数: {df.shape[1]}")

#==================== 2. 实测数据 ====================
# col_list = np.loadtxt("your_data.txt")
Y_data_1, Y_data_2, Y_data_3, Y_data_4, Y_data_5, Y_data_6, Y_data_7 = (col_list[0], col_list[2], col_list[4],
                                                                        col_list[6], col_list[8], col_list[10], col_list[12])
Sm_data_1, Sm_data_2, Sm_data_3, Sm_data_4, Sm_data_5, Sm_data_6, Sm_data_7 = (col_list[1], col_list[3], col_list[5],
                                                                        col_list[7], col_list[9], col_list[11], col_list[13])
sigma_PTS_Y_1, sigma_PTS_Y_2, sigma_NTS_Y = (col_list[14], col_list[16], col_list[18])
sigma_PTS_Sm_1, sigma_PTS_Sm_2, sigma_NTS_Sm = (col_list[15], col_list[17], col_list[19])
#==================== 3. 配置参数 ====================

F = np.linspace(0.01, 1, 100)
eps = 1e-4

if inv.type == "No_NTS" and inv.name == "Eastern":

    # --------------------------
    # 【第一步：预计算所有RMSE，获取全局误差范围】
    # --------------------------
    rmse1_list = []
    rmse2_list = []
    rmse3_list = []

    for topN_param in topN_params:
        (p19, p20) = topN_param[-2:]
        (p0, p1, p2, p3, p4, p5, p6,
         p7, p8, p9,
         p10, p11, p12) = topN_param[0:13]
        Sm0 = p0 * p9 ** (p4 / (1 - p6) - 1) + p6 / (p6 + p4 - 1) * p2 * (1 - p9 ** (p4 / (1 - p6) - 1))
        Y0 = p1 * p9 ** (p5 / (1 - p6) - 1) + p6 / (p6 + p5 - 1) * p3 * (1 - p9 ** (p5 / (1 - p6) - 1))
        Sm1 = Sm0 * (p12 ** (p7 - 1.0)); Y1 = Y0 * (p12 ** (p8 - 1.0))  # 阶段b的初始值

        # point 1 rmse
        F_1 = (Y_data_1 / (Y1 * p20)) ** (1.0/(p11 - 1 + eps))
        pred_Sm1 = (Sm1 * p19) * (F_1 ** (p10 - 1))
        rmse1 = np.sqrt(np.mean((pred_Sm1 - Sm_data_1)**2))
        rmse1_list.append(rmse1)

        # point 2 rmse
        p13, p14, p15 = topN_param[13:16]
        Sm2 = Sm1 * (p15 ** (p10 - 1))
        Y2  = Y1  * (p15 ** (p11 - 1))
        F_2 = (Y_data_2 / (Y2 * p20)) ** (1.0/(p14 - 1 + eps))
        pred_Sm2 = (Sm2 * p19) * (F_2 ** (p13 - 1))
        rmse2 = np.sqrt(np.mean((pred_Sm2 - Sm_data_2)**2))
        rmse2_list.append(rmse2)

        # point3 rmse
        p16, p17, p18 = topN_param[16:19]
        Sm3 = Sm1 * (p18 ** (p10 - 1))
        Y3  = Y1  * (p18 ** (p11 - 1))
        F_3 = (Y_data_3 / (Y3 * p20)) ** (1.0/(p17 - 1 + eps))
        pred_Sm3 = (Sm3 * p19) * (F_3 ** (p16 - 1))
        rmse3 = np.sqrt(np.mean((pred_Sm3 - Sm_data_3)**2))
        rmse3_list.append(rmse3)

    rmse1_arr = np.array(rmse1_list)
    rmse2_arr = np.array(rmse2_list)
    rmse3_arr = np.array(rmse3_list)

    # 全局归一化（所有轨迹共用一套标尺）
    norm1 = plt.Normalize(vmin=np.min(rmse1_arr), vmax=np.max(rmse1_arr))
    norm2 = plt.Normalize(vmin=np.min(rmse2_arr), vmax=np.max(rmse2_arr))
    norm3 = plt.Normalize(vmin=np.min(rmse3_arr), vmax=np.max(rmse3_arr))
    """
    vmin_global = min(np.min(rmse2_arr), np.min(rmse3_arr))
    vmax_global = max(np.max(rmse2_arr), np.max(rmse3_arr))
    norm2 = plt.Normalize(vmin=vmin_global, vmax=vmax_global)
    """

    # cmap1 = plt.colormaps["YlOrBr"]
    # cmap2 = plt.colormaps["Oranges"]
    # cmap3 = plt.colormaps["Oranges"]
    # 自建黄褐色渐变colormap：浅黄褐色→深黄褐色
    cmap1 = LinearSegmentedColormap.from_list("cmap1", ["#f8eedb", "#704c2b"])
    cmap2 = LinearSegmentedColormap.from_list("orange_custom", ["#ffd9b3", "#e67020"])
    cmap3 = LinearSegmentedColormap.from_list("orange_custom", ["#ffd9b3", "#e67020"])

    # --------------------------
    # 【第二步：绘图循环】
    # --------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    for idx, topN_param in enumerate(topN_params):
        (p19, p20) = topN_param[-2:]
        (p0, p1, p2, p3, p4, p5, p6,
        p7, p8, p9) = topN_param[0:10]
        # 根曲线 Sr0‑Y0
        Sm_0 = p0 * p9 ** (p4 / (1 - p6) - 1) + p6 / (p6 + p4 - 1) * p2 * (1 - p9 ** (p4 / (1 - p6) - 1))
        Y_0 = p1 * p9 ** (p5 / (1 - p6) - 1) + p6 / (p6 + p5 - 1) * p3 * (1 - p9 ** (p5 / (1 - p6) - 1))
        Sm0 = Sm_0 * (F ** (p7 - 1)) * p19
        Y0  = Y_0 * (F ** (p8 - 1)) * p20


        # ======演化分支 1 ======
        (p10, p11, p12) = topN_param[10:13]
        Sm_1 = Sm_0 * (p12 ** (p7 - 1))
        Y_1 = Y_0 * (p12 ** (p8 - 1))
        Sm1 = Sm_1 * (F ** (p10 - 1)) * p19
        Y1 = Y_1 * (F ** (p11 - 1)) * p20
        # 获取当前轨迹对应的单RMSE值，整条线取统一颜色
        c1 = cmap1(norm1(rmse1_arr[idx]))
        ax.plot(Y1, Sm1, color=c1, lw=0.7)

        #======演化分支 2 ======
        p13, p14, p15 = topN_param[13:16]
        Sm_2 = Sm_1 * (p15 ** (p10 - 1))
        Y_2  = Y_1  * (p15 ** (p11 - 1))
        Sm2  = Sm_2 * (F ** (p13 - 1)) * p19
        Y2   = Y_2  * (F ** (p14 - 1)) * p20
        c2 = cmap2(norm2(rmse2_arr[idx]))
        ax.plot(Y2, Sm2, color=c2, lw=0.7)

        #======演化分支 3 ======
        p16, p17, p18 = topN_param[16:19]
        Sm_3 = Sm_1 * (p18 ** (p10 - 1))
        Y_3  = Y_1  * (p18 ** (p11 - 1))
        Sm3  = Sm_3 * (F ** (p16 - 1)) * p19
        Y3   = Y_3  * (F ** (p17 - 1)) * p20
        c3 = cmap3(norm3(rmse3_arr[idx]))
        ax.plot(Y3, Sm3, color=c3, lw=0.7, linestyle='--')

        # 根曲线 Sr0‑Y0 放在最后
        ax.plot(Y0, Sm0, color="k", lw=0.4, alpha=0.25)

    # 实测散点
    # ======== 第0组 Y_data[0], SrY_data[0]：棕黄色 #D4AF37，前10实心方形，后面空心方形 ========
    # 前10个点：实心方形
    ax.scatter(Y_data_1[:10], Sm_data_1[:10],
               marker="s",
               facecolor="#D4AF37",
               edgecolor="#D4AF37",
               zorder=12, s=50
               ) # label = "E-DS Ap, Duan et al. (2026)"
    # 索引≥10所有点：空心方形（仅边框）
    ax.scatter(Y_data_1[10:], Sm_data_1[10:],
               marker="s",
               facecolor="none",
               edgecolor="#D4AF37",
               zorder=12, s=50
               ) # label = "E-DS Ap, Liu et al. (2021)"
    # ======== 第1组 Y_data[1], SrY_data[1]：橙色 #FF8822，前10实心菱形，后面空心菱形 ========
    ax.errorbar(Y_data_2[:5], Sm_data_2[:5],
                xerr=sigma_PTS_Y_2[:5],
                yerr=sigma_PTS_Sm_2[:5],
                fmt="D",
                mfc="#FF8822",
                mec="#FF8822",
                ecolor="#FF8822",
                capsize=3,
                capthick=0.8,  # cap两端小横线的粗细
                elinewidth=0.8,  # 误差棒竖/横线主干粗细，常用 0.6~1.2
                markersize=7,
                zorder=12
                )  # label = "E-PTS Ap, Duan et al. (2026)"
    ax.scatter(Y_data_2[5:], Sm_data_2[5:],
                marker="D",
                facecolor="none",
                edgecolor="#FF8822",
                zorder=12, s=50
                ) # label = "E-PTS Ap, Li (2024)"
    # ======== 第2组 Y_data[2], SrY_data[2]：橙色，全部实心菱形 ========
    ax.errorbar(Y_data_3, Sm_data_3,
                xerr=sigma_PTS_Y_2,
                yerr=sigma_PTS_Sm_2,
                fmt="D",
                mfc="#FF8822",
                mec="#FF8822",
                ecolor="#FF8822",
                capsize=3,
                capthick=0.8,  # cap两端小横线的粗细
                elinewidth=0.8,  # 误差棒竖/横线主干粗细，常用 0.6~1.2
                markersize=7,
                zorder=12)

    # ax.scatter(Y_data, SrY_data, marker="o", c="black", zorder=10, s=35, label="Observed")
    ax.set_xlabel("Y (ppm)", fontsize=18)
    ax.set_ylabel("Sm (ppm)", fontsize=18)
    # ax.legend()
    # X线性，Y对数
    ax.set_yscale('log')
    # 定义主刻度
    xticks = [0, 500, 1000, 1500, 2000, 2500]
    ax.set_xticks(xticks)
    ax.set_xlim(-20, 2500)
    # Y轴固定对数刻度
    yticks = [10, 100]
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(10, 800)
    ax.tick_params(axis='both', labelsize=12, width=1.0, length=4)

    # --------------------------
    # 添加颜色条（可选，展示RMSE标尺）
    # --------------------------
    sm1 = plt.cm.ScalarMappable(norm=norm1, cmap=cmap1)
    sm1.set_array([])
    """
    cbar1 = fig.colorbar(sm1, ax=ax, shrink=0.7, pad=0.02)
    cbar1.set_label('RMSE', fontsize=10)
    """
    # 创建内嵌坐标轴：宽度占子图60%，高度5%，位置右上角
    axins1 = inset_axes(ax,
                        width=2.0,  # 色条总长度（占ax宽度比例）
                        height=0.1,  # 色条厚度
                        loc="upper right",
                        bbox_to_anchor=(0.895, 0.11),  # 第一条y位置
                        bbox_transform=ax.transAxes,
                        borderpad=0
                        )  # 位置：upper right / upper left / lower right / lower left
    # 生成横向colorbar
    cbar1 = fig.colorbar(sm1, cax=axins1, orientation="horizontal")
    cbar1.ax.xaxis.set_ticks_position("bottom")  # 刻度放在色条下方，论文图好看
    cbar1.ax.set_xticks([])
    cbar1.outline.set_visible(False)
    # 设置色条标签（示例）
    cbar1.set_label("RMSE", fontsize=10)
    cbar1.ax.xaxis.set_label_position('top')
    # cbar1.ax.tick_params(labelsize=8)

    sm2 = plt.cm.ScalarMappable(norm=norm2, cmap=cmap2)
    sm2.set_array([])
    # cbar2 = fig.colorbar(sm2, ax=ax, shrink=0.7, pad=0.12)
    axins2 = inset_axes(ax,
                    width=2.0,
                    height=0.1,
                    loc="upper right",
                    bbox_to_anchor=(0.895, 0.085),
                    bbox_transform=ax.transAxes,
                    borderpad=0)
    cbar2 = fig.colorbar(sm2, cax=axins2, orientation="horizontal")
    cbar2.ax.xaxis.set_ticks_position("bottom")
    # 关闭刻度+刻度数字
    cbar2.ax.set_xticks([])
    cbar2.outline.set_visible(False)
    # cbar2.set_label("RMSE2", fontsize=8)
    # cbar2.ax.tick_params(labelsize=7)
    """
    sm3 = plt.cm.ScalarMappable(norm=norm3, cmap=cmap3)
    sm3.set_array([])
    cbar3 = fig.colorbar(sm3, ax=ax, shrink=0.7, pad=0.22)
    """
    # plt.tight_layout()
    # plt.show()
    plt.savefig("SrY_Y_Eastern_No_NTS.png", dpi=600, bbox_inches="tight")
    plt.close()  # 释放内存，循环绘图必须加，避免内存泄漏

elif (inv.type == "No_NTS" and inv.name == "Western") or (inv.type == "NTS" and inv.name == "Eastern"):
    rmse1_list = []
    for topN_param in topN_params:
        (p0, p1, p2, p3, p4, p5, p6,
         p7, p8, p9,
         p10, p11, p12,
         p13, p14) = topN_param
        Sm0 = p0 * p9 ** (p4 / (1 - p6) - 1) + p6 / (p6 + p4 - 1) * p2 * (1 - p9 ** (p4 / (1 - p6) - 1))
        Y0 = p1 * p9 ** (p5 / (1 - p6) - 1) + p6 / (p6 + p5 - 1) * p3 * (1 - p9 ** (p5 / (1 - p6) - 1))
        Sm1 = Sm0 * (p12 ** (p7 - 1.0)); Y1 = Y0 * (p12 ** (p8 - 1.0))  # 阶段b的初始值
        # point 1 rmse
        if inv.type == "No_NTS" and inv.name == "Western":
            F_1 = (Y_data_4 / (Y1 * p14)) ** (1.0/(p11 - 1 + eps))
            pred_Sm1 = (Sm1 * p13) * (F_1 ** (p10 - 1))
            rmse1 = np.sqrt(np.mean((pred_Sm1 - Sm_data_4)**2))
        else:
            F_1 = (Y_data_5 / (Y1 * p14)) ** (1.0 / (p11 - 1 + eps))
            pred_Sm1 = (Sm1 * p13) * (F_1 ** (p10 - 1))
            rmse1 = np.sqrt(np.mean((pred_Sm1 - Sm_data_5) ** 2))
        rmse1_list.append(rmse1)
    rmse1_arr = np.array(rmse1_list)
    # 全局归一化（所有轨迹共用一套标尺）
    norm1 = plt.Normalize(vmin=np.min(rmse1_arr), vmax=np.max(rmse1_arr))
    if inv.type == "No_NTS" and inv.name == "Western":
        cmap1 = LinearSegmentedColormap.from_list("green_cmap", ["#e6f4ea", "#205935"])
        # cmap1 = plt.colormaps["Greens"]
    else:
        # cmap1 = plt.colormaps["Reds"]
        cmap1 = LinearSegmentedColormap.from_list("red_cmap", ["#fce4e4", "#701818"])
    fig, ax = plt.subplots(figsize=(8, 8))
    for idx, topN_param in enumerate(topN_params):
        (p0, p1, p2, p3, p4, p5, p6,
         p7, p8, p9,
         p10, p11, p12,
         p13, p14) = topN_param
        # 根曲线 Sr0‑Y0
        Sm_0 = p0 * p9 ** (p4 / (1 - p6) - 1) + p6 / (p6 + p4 - 1) * p2 * (1 - p9 ** (p4 / (1 - p6) - 1))
        Y_0 = p1 * p9 ** (p5 / (1 - p6) - 1) + p6 / (p6 + p5 - 1) * p3 * (1 - p9 ** (p5 / (1 - p6) - 1))
        Sm0 = Sm_0 * (F ** (p7 - 1)) * p13
        Y0 = Y_0 * (F ** (p8 - 1)) * p14
        ax.plot(Y0, Sm0, color="k", lw=0.4, alpha=0.25)
        # ======演化分支 1 ======
        Sm_1 = Sm_0 * (p12 ** (p7 - 1))
        Y_1 = Y_0 * (p12 ** (p8 - 1))
        Sm1 = Sm_1 * (F ** (p10 - 1)) * p13
        Y1 = Y_1 * (F ** (p11 - 1)) * p14
        # 获取当前轨迹对应的单RMSE值，整条线取统一颜色
        c1 = cmap1(norm1(rmse1_arr[idx]))
        ax.plot(Y1, Sm1, color=c1, lw=0.7)
    # 实测数据点
    if inv.type == "No_NTS" and inv.name == "Western":
        ax.scatter(Y_data_4, Sm_data_4,
                   marker="s",
                   facecolor="#33bb55",
                   edgecolor="#33bb55",
                   zorder=12, s=50)
    else:
        ax.scatter(Y_data_5[10:], Sm_data_5[10:],
                   marker="o",
                   facecolor="none",
                   edgecolor="#d62828",
                   zorder=10, s=50)
        """
        ax.scatter(Y_data_5[:10], Sm_data_5[:10],
                   marker="o",
                   facecolor="#d62828",
                   edgecolor="#d62828",
                   zorder=10, s=50)
        """
        ax.errorbar(Y_data_5[:10], Sm_data_5[:10],
                    xerr=sigma_NTS_Y[:10],
                    yerr=sigma_NTS_Sm[:10],
                    fmt="o",
                    mfc="#d62828",
                    mec="#d62828",
                    ecolor="#d62828",
                    capsize=3,
                    capthick=0.8,  # cap两端小横线的粗细
                    elinewidth=0.8,  # 误差棒竖/横线主干粗细，常用 0.6~1.2
                    markersize=7,
                    zorder=12)
        # ax.scatter(Y_data, SrY_data, marker="o", c="black", zorder=10, s=35, label="Observed")
    ax.set_xlabel("Y (ppm)", fontsize=18)
    ax.set_ylabel("Sm (ppm)", fontsize=18)
    # ax.legend()
    # X线性，Y对数
    ax.set_yscale('log')
    # 定义主刻度
    if inv.type == "No_NTS" and inv.name == "Western":
        xticks = [-50, 0, 200, 400, 600, 800]
        ax.set_xticks(xticks)
        ax.set_xlim(-50, 800)
        yticks = [1, 10, 100, 1000]
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(1, 1000)
    else:
        xticks = [0, 500, 1000, 1500, 2000]
        ax.set_xticks(xticks)
        ax.set_xlim(-30, 2000)
        yticks = [10, 100, 1000]
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(5, 4000)
    # Y轴固定对数刻度
    ax.tick_params(axis='both', labelsize=12, width=1.0, length=4)

    # 添加颜色条（可选，展示RMSE标尺）
    sm1 = plt.cm.ScalarMappable(norm=norm1, cmap=cmap1)
    sm1.set_array([])
    # 创建内嵌坐标轴：宽度占子图60%，高度5%，位置右上角
    axins1 = inset_axes(ax,
                        width=2.0,  # 色条总长度（占ax宽度比例）
                        height=0.1,  # 色条厚度
                        loc="upper right",
                        bbox_to_anchor=(0.895, 0.11),  # 第一条y位置
                        bbox_transform=ax.transAxes,
                        borderpad=0
                        )  # 位置：upper right / upper left / lower right / lower left
    # 生成横向colorbar
    cbar1 = fig.colorbar(sm1, cax=axins1, orientation="horizontal")
    cbar1.ax.xaxis.set_ticks_position("bottom")  # 刻度放在色条下方，论文图好看
    cbar1.ax.set_xticks([])
    cbar1.outline.set_visible(False)
    # 设置色条标签（示例）
    cbar1.set_label("RMSE", fontsize=10)
    cbar1.ax.xaxis.set_label_position('top')
    # cbar1 = fig.colorbar(sm1, ax=ax, shrink=0.7, pad=0.02)
    # cbar1.set_label('RMSE‑Trajectory 1', fontsize=10)
    # plt.tight_layout()
    # plt.show()
    if inv.type == "No_NTS" and inv.name == "Western":
        plt.savefig("SrY_Y_Western_No_NTS.png", dpi=600, bbox_inches="tight")
    else:
        plt.savefig("SrY_Y_Eastern_NTS.png", dpi=600, bbox_inches="tight")
    plt.close()  # 释放内存，循环绘图必须加，避免内存泄漏
else:
    rmse1_list = []
    rmse2_list = []
    for topN_param in topN_params:
        (p13, p14) = topN_param[13:15]
        (p0, p1, p2, p3, p4, p5, p6,
         p7, p8, p9,
         p10, p11, p12) = topN_param[0:13]
        # point 1 rmse
        Sm_0_PTS = p0 * p9 ** (p4 / (1 - p6) - 1) + p6 / (p6 + p4 - 1) * p2 * (1 - p9 ** (p4 / (1 - p6) - 1))
        Y_0_PTS = p1 * p9 ** (p5 / (1 - p6) - 1) + p6 / (p6 + p5 - 1) * p3 * (1 - p9 ** (p5 / (1 - p6) - 1))
        Sm0_PTS = Sm_0_PTS * (p12 ** (p7 - 1.0)); Y0_PTS = Y_0_PTS * (p12 ** (p8 - 1.0))  # 阶段b的初始值
        F_PTS = (Y_data_6 / (Y0_PTS * p14)) ** (1.0 / (p11 - 1 + eps))
        pred_Sm_PTS = (Sm0_PTS * p13) * (F_PTS ** (p10 - 1))
        rmse1 = np.sqrt(np.mean((pred_Sm_PTS - Sm_data_6) ** 2))
        rmse1_list.append(rmse1)
        # point 2 rmse
        topN_param1 = topN_param[15:]
        (q13, q14) = topN_param1[-2:]
        (q0, q1, q2, q3, q4, q5, q6,
         q7, q8, q9,
         q10, q11, q12) = topN_param1[0:13]
        Sm_0_NTS = q0 * q9 ** (q4 / (1 - q6) - 1) + q6 / (q6 + q4 - 1) * q2 * (1 - q9 ** (q4 / (1 - q6) - 1))
        Y_0_NTS = q1 * q9 ** (q5 / (1 - q6) - 1) + q6 / (q6 + q5 - 1) * q3 * (1 - q9 ** (q5 / (1 - q6) - 1))
        Sm0_NTS = Sm_0_NTS * (q12 ** (q7 - 1.0)); Y0_NTS = Y_0_NTS * (q12 ** (q8 - 1.0))  # 阶段b的初始值
        F_NTS = (Y_data_7 / (Y0_NTS * q14)) ** (1.0 / (q11 - 1 + eps))
        pred_Sm_NTS = (Sm0_NTS * q13) * (F_NTS ** (q10 - 1))
        rmse2 = np.sqrt(np.mean((pred_Sm_NTS - Sm_data_7) ** 2))
        rmse2_list.append(rmse2)
    rmse1_arr = np.array(rmse1_list)
    rmse2_arr = np.array(rmse2_list)
    norm1 = plt.Normalize(vmin=np.min(rmse1_arr), vmax=np.max(rmse1_arr))
    norm2 = plt.Normalize(vmin=np.min(rmse2_arr), vmax=np.max(rmse2_arr))
    cmap1 = LinearSegmentedColormap.from_list("pink_custom", ["#fff0f3", "#c94c75"])
    cmap2 = LinearSegmentedColormap.from_list("blue_custom", ["#f0f7ff", "#1f5798"])

    fig, ax = plt.subplots(figsize=(8, 8))
    for idx, topN_param in enumerate(topN_params):
        (p0, p1, p2, p3, p4, p5, p6,
         p7, p8, p9,
         p10, p11, p12,
         p13, p14) = topN_param[0:15]
        # 根曲线 Sr0_0‑Y0_0
        Sm_0_PTS = p0 * p9 ** (p4 / (1 - p6) - 1) + p6 / (p6 + p4 - 1) * p2 * (1 - p9 ** (p4 / (1 - p6) - 1))
        Y_0_PTS = p1 * p9 ** (p5 / (1 - p6) - 1) + p6 / (p6 + p5 - 1) * p3 * (1 - p9 ** (p5 / (1 - p6) - 1))
        Sm0_PTS = Sm_0_PTS * (F ** (p7 - 1.0)) * p13
        Y0_PTS = Y_0_PTS * (F ** (p8 - 1.0)) * p14
        ax.plot(Y0_PTS, Sm0_PTS, color="k", lw=0.4, alpha=0.25)
        #======演化分支 1 ======
        Sm_1_PTS = Sm_0_PTS * (p12 ** (p7 - 1))
        Y_1_PTS  = Y_0_PTS * (p12 ** (p8 - 1))
        Sm1_PTS  = Sm_1_PTS * (F ** (p10 - 1)) * p13
        Y1_PTS   = Y_1_PTS  * (F ** (p11 - 1)) * p14
        # 获取当前轨迹对应的单RMSE值，整条线取统一颜色
        c1 = cmap1(norm1(rmse1_arr[idx]))
        ax.plot(Y1_PTS, Sm1_PTS, color=c1, lw=0.7)
       # W-NTS
        # 根曲线 Sr0_1‑Y0_1
        (q0, q1, q2, q3, q4, q5, q6,
         q7, q8, q9,
         q10, q11, q12,
         q13, q14) = topN_param[15:]
        Sm_0_NTS = q0 * q9 ** (q4 / (1 - q6) - 1) + q6 / (q6 + q4 - 1) * q2 * (1 - q9 ** (q4 / (1 - q6) - 1))
        Y_0_NTS = q1 * q9 ** (q5 / (1 - q6) - 1) + q6 / (q6 + q5 - 1) * q3 * (1 - q9 ** (q5 / (1 - q6) - 1))
        Sm0_NTS = Sm_0_NTS * (F ** (q7 - 1.0)) * q13
        Y0_NTS = Y_0_NTS * (F ** (q8 - 1.0)) * p14
        ax.plot(Y0_NTS, Sm0_NTS, color="k", lw=0.4, alpha=0.25)
        # ======演化分支 2 ======
        Sm_1_NTS = Sm_0_NTS * (q12 ** (q7 - 1))
        Y_1_NTS = Y_0_NTS * (q12 ** (q8 - 1))
        Sm1_NTS = Sm_1_NTS * (F ** (q10 - 1)) * q13
        Y1_NTS = Y_1_NTS * (F ** (q11 - 1)) * q14
        # 获取当前轨迹对应的单RMSE值，整条线取统一颜色
        c2 = cmap2(norm2(rmse2_arr[idx]))
        ax.plot(Y1_NTS, Sm1_NTS, color=c2, lw=0.7)

    #实测数据
    ax.scatter(Y_data_6, Sm_data_6,
               marker="o",
               facecolor="#e86a92",
               edgecolor="#e86a92",
               zorder=10, s=50)
    ax.scatter(Y_data_7[:10], Sm_data_7[:10],
               marker="o",
               facecolor="#2171b5",
               edgecolor="#2171b5",
               zorder=10, s=50)
    my_marker = make_ring_cross_gap_marker(2.0, 1.999, 1.85, 0.001)
    ax.scatter(Y_data_7[10:], Sm_data_7[10:],
               marker=my_marker,
               facecolor="#FFFFFF",
               edgecolor="#2171b5",
               zorder=10, s=50)
    ax.set_xlabel("Y (ppm)", fontsize=18)
    ax.set_ylabel("Sm (ppm)", fontsize=18)
    # ax.legend()
    # X线性，Y对数
    ax.set_yscale('log')
    xticks = [0, 500, 1000, 1500, 2000]
    ax.set_xticks(xticks)
    ax.set_xlim(-50, 2100)
    yticks = [1, 10, 100, 1000]
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(1, 5000)
    ax.tick_params(axis='both', labelsize=12, width=1.0, length=4)

    sm1 = plt.cm.ScalarMappable(norm=norm1, cmap=cmap1)
    sm1.set_array([])
    # 创建内嵌坐标轴：宽度占子图60%，高度5%，位置右上角
    axins1 = inset_axes(ax,
                        width=2.0,  # 色条总长度（占ax宽度比例）
                        height=0.1,  # 色条厚度
                        loc="upper right",
                        bbox_to_anchor=(0.895, 0.11),  # 第一条y位置
                        bbox_transform=ax.transAxes,
                        borderpad=0
                        )  # 位置：upper right / upper left / lower right / lower left
    # 生成横向colorbar
    cbar1 = fig.colorbar(sm1, cax=axins1, orientation="horizontal")
    cbar1.ax.xaxis.set_ticks_position("bottom")  # 刻度放在色条下方，论文图好看
    cbar1.ax.set_xticks([])
    cbar1.outline.set_visible(False)
    # 设置色条标签（示例）
    cbar1.set_label("RMSE", fontsize=10)
    cbar1.ax.xaxis.set_label_position('top')
    # cbar1.ax.tick_params(labelsize=8)
    sm2 = plt.cm.ScalarMappable(norm=norm2, cmap=cmap2)
    sm2.set_array([])
    # cbar2 = fig.colorbar(sm2, ax=ax, shrink=0.7, pad=0.12)
    axins2 = inset_axes(ax,
                        width=2.0,
                        height=0.1,
                        loc="upper right",
                        bbox_to_anchor=(0.895, 0.085),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
    cbar2 = fig.colorbar(sm2, cax=axins2, orientation="horizontal")
    cbar2.ax.xaxis.set_ticks_position("bottom")
    # 关闭刻度+刻度数字
    cbar2.ax.set_xticks([])
    cbar2.outline.set_visible(False)
    plt.savefig("SrY_Y_Western_NTS.png", dpi=600, bbox_inches="tight")
    plt.close()  # 释放内存，循环绘图必须加，避免内存泄漏