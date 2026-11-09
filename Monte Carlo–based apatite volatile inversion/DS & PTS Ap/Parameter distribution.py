import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde
import string
import matplotlib.patches as mpatches
# =====================【这里修改成你自己Excel完整路径！】=====================
file_K1 = r"F:\论文写作\Revision\GR-apatite modelling\磷灰石成分蒙特卡洛模拟\分阶段评估\Result_Composition_1.csv"
file_K2 = r"F:\论文写作\Revision\GR-apatite modelling\磷灰石成分蒙特卡洛模拟\分阶段评估\Result_T_1.csv"
file_K3 = r"F:\论文写作\Revision\GR-apatite modelling\磷灰石成分蒙特卡洛模拟\分阶段评估\Result_T_Composition_1.csv"
d_labels = ["H2O_initial", "F_initial", "Cl_initial",
            "D_H2O_mineral/melt", "D_F_mineral/melt", "D_Cl_mineral/melt",
            "D_F_fluid/melt", "D_Cl_fluid/melt", "m_H2O_saturation",
            "rmse1", "rmse2", "rmse"]
tag2latex = {
    "H2O_initial": r"$H_2O_{ini}$",
    "F_initial": r"$F_{ini}$",
    "Cl_initial": r"$Cl_{ini}$",
    "D_H2O_mineral/melt": r"$D^{min/mel}_{H_2O}$",
    "D_F_mineral/melt": r"$D^{min/mel}_{F}$",
    "D_Cl_mineral/melt": r"$D^{min/mel}_{Cl}$",
    "D_F_fluid/melt": r"$D^{flu/mel}_{F}$",
    "D_Cl_fluid/melt": r"$D^{flu/mel}_{Cl}$",
    "m_H2O_saturation": r"$m_{H_2O}^{sat}$",
    "rmse1": r"$log\ RMSE1$",
    "rmse2": r"$log\ RMSE2$",
    "rmse": r"$log\ (RMSE1+RMSE2)$"
    # "rmse2": r"$\mathrm{RMSE_2}$"
}
dataset_names = ["K1", "K2", "K3"]
color_map = ["#2255aa", "#22bbbb", "#228844"]
# 绘图全局字体（期刊Arial）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 10),squeeze=False)
# 读取3个Excel
df_K1   = pd.read_csv(file_K1)
df_K2   = pd.read_csv(file_K2)
df_K3 = pd.read_csv(file_K3)
# 定义筛选函数：按rmse1升序，取最小的前1000行
def filter_lowest_rmse(df, n=1000):
    df_sorted = df.sort_values(by="rmse", ascending=True)
    df_sub = df_sorted.head(n).copy()
    return df_sub

df_K1    = filter_lowest_rmse(df_K1, 1000)
df_K2    = filter_lowest_rmse(df_K2, 1000)
df_K3  = filter_lowest_rmse(df_K3, 1000)
# 保存，index=False 不输出pandas行号
df_K1.to_csv(r"分阶段评估\Result_T_best_1.csv", index=False)
df_K2.to_csv(r"分阶段评估\Result_Composition_best_1.csv", index=False)
df_K3.to_csv(r"分阶段评估\Result_T_Composition_best_1.csv", index=False)

data_collection = {
    "K1": df_K1,
    "K2":   df_K2,
    "K3": df_K3,
}
def get_paramenter_colname(d_label):
    return f"{d_label}"

for i, d_tag in enumerate(d_labels):
    divisor = 4 # 一行4个图
    m, n = divmod(i, divisor)
    ax = axes[m, n]
    stacks = []
    for ds_index, ds_name in enumerate(dataset_names):
        df = data_collection[ds_name]
        col = get_paramenter_colname(d_tag)
        if col not in df.columns:
            continue  # 该表无此元素分配系数列，跳过
        data = df[col].dropna()
        if len(data) < 3:
            continue
        stacks.append((ds_name, ds_index, data))

    if not stacks:
        ax.set_visible(False)  # 完全无数据，隐藏子图
        continue

    # 统一bins：把3套数据合并取共同范围，保证堆叠对齐
    all_data = np.concatenate([d for _, _, d in stacks])
    x_data_min = all_data.min()
    x_data_max = all_data.max()
    # 【关键】临时获取matplotlib自动推荐的X刻度
    # 先设置x范围，让matplotlib自动生成最优刻度
    ax.set_xlim(x_data_min, x_data_max)
    # 触发matplotlib刻度自动计算
    ax.figure.canvas.draw()
    auto_ticks = ax.get_xticks()
    tick_spacing = np.diff(auto_ticks)[0]  # matplotlib自动给出的主刻度间隔
    # 主刻度之间切分成2个bin（n=2，你可以改成4，分得更细）
    n_bin_per_tick = 8
    bin_width = tick_spacing / n_bin_per_tick
    # 将bin边界对齐到tick网格
    xmin_bin = np.floor(x_data_min / bin_width) * bin_width
    xmax_bin = np.ceil(x_data_max / bin_width) * bin_width
    bins = np.arange(xmin_bin, xmax_bin + bin_width, bin_width)
    # 让X轴刻度继续使用matplotlib自动推荐的刻度，不覆盖！
    ax.set_xticks(auto_ticks)
    bs = (auto_ticks.max() - auto_ticks.min()) * 0.1
    ax.set_xlim(x_data_min - bs, x_data_max + bs)

    # 逐层堆叠：bottom 累计
    cumulative_bottom = np.zeros(len(bins) - 1)
    ax2 = ax.twinx()
    for ds_name, ds_index, data in stacks:
        counts, _ = np.histogram(data, bins=bins)
        ax.bar(bins[:-1], counts,
               width=np.diff(bins),  # 每根bar宽度 = bin宽度
               bottom=cumulative_bottom,  # 从累计底部叠上去
               align='edge', alpha=0.75,
               color=color_map[ds_index],
               label=ds_name, edgecolor='black', linewidth=0.3)
        cumulative_bottom += counts  # 更新累计高度
        # KDE核密度曲线
        kde = gaussian_kde(data)
        xx = np.linspace(np.floor(bins.min()), np.ceil(bins.max()), 1000)
        ax2.plot(xx, kde(xx), color=color_map[ds_index], lw=1.25)
        average_data = np.mean(data)
        ax.axvline(x=average_data, linestyle='--', color=color_map[ds_index], linewidth=1.25,
                   zorder=4)  # 层级，放在直方图之上
    total_height = cumulative_bottom.max()

    if i < 3 or i == 8:
        xlabel_text = fr"{tag2latex[d_tag]} (wt.%)"
    else:
        xlabel_text = fr"{tag2latex[d_tag]}"
    ax.set_xlabel(xlabel_text, fontfamily='Arial', fontsize=12)
    # Y轴标签只放在第一列
    if n == 0:
        ax.set_ylabel("Count", fontsize=12)
    elif n == 3:
        ax2.set_ylabel("Probability density", fontsize=12)
    ax.tick_params(axis='both', labelsize=10, direction='in', top=False, right=False)
    ax2.tick_params(axis='y', labelsize=10, direction='in', right=True)

total_ax = len(axes.flat)
letters = [f'({string.ascii_lowercase[k]})' for k in range(total_ax)]

# 遍历所有子图，左上角添加标签
for i, ax in enumerate(axes.flat):
    ax.text(
        0.90, 0.98,  # axes fraction坐标：x=0.02靠左，y=0.98靠上
        letters[i],
        transform=ax.transAxes,
        fontsize=14,
        fontfamily='Times New Roman',  # 和你全文字体统一，换成'Times New Roman'也可以
        va='top', ha='left',
        weight='bold'  # 加粗，期刊图常用；不需要加粗就删掉这一行
        )
"""
# 手动创建图例色块（和你直方图颜色一一对应）
patch1 = mpatches.Patch(color="#2255aa", label="K1")
patch2 = mpatches.Patch(color="#22bbbb", label="K2")
patch3 = mpatches.Patch(color="#228844", label="K3")
# 选中右下角最后一个子图 axs[2,3]
ax_empty = axes[0, 0]
# 隐藏坐标轴、边框、刻度
ax_empty.axis('off')
# 在这个空白轴的位置添加图例
ax_empty.legend(
    handles=[patch1, patch2, patch3],
    loc="upper left",          # 图例放在这个axes正中间
    fontsize=18,
    frameon=False,         # 去掉图例边框，期刊常用
    bbox_to_anchor=(0.135, 0.96),
    # handletextpad=0.4,  # 图例条目和文字距离，可选，让图例更紧凑
    # ncol=1              # 改成ncol=2可以横向两列排布图例
)
"""
# 图例
patch1 = mpatches.Patch(color="#2255aa", label="K of Eq S")
patch2 = mpatches.Patch(color="#22bbbb", label="K of Eq S")
patch3 = mpatches.Patch(color="#228844", label="K of Eq S")
ax_j = axes[0, 0]
ax_j.legend(
    handles=[patch1, patch2, patch3],
    loc="upper left",
    fontsize=12,
    frameon=False,
    # bbox_to_anchor=(0.98, 0.98)
)

# ========= 3、关键：调边距与子图横向间距，wspace加大！ =========
fig.subplots_adjust(left=0.00, right=1.0, top=1.0, bottom=0,
                    hspace=0.175, wspace=0.3)
# 输出保存图片
# plt.tight_layout(rect=[0, 0, 1, 1])
save_path = "分阶段评估\parameter_best_1.png"
plt.savefig(save_path, dpi=600, bbox_inches="tight")
# plt.show()
print(f"图像保存路径：{Path(save_path).absolute()}")

