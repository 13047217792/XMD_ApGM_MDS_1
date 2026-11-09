import numpy as np
import pandas as pd
import csv
from matplotlib.collections import LineCollection
import matplotlib.path as mpath
def extract_csv_columns(file_path):
    """
    提取 CSV 各列数据，返回列表的列表：[[列1数据], [列2数据], ...]

    Parameters
    ----------
    file_path : str
        CSV 文件路径。
    Returns
    -------
    list
        每个子列表对应一列的全部数据（字符串）。
    """
    with open(file_path, mode='r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return []

    # 跳过第0行表头，只读取数据行
    data_rows = rows[1:]
    n_cols = len(data_rows[0])
    columns = []
    for col_idx in range(n_cols):
        col_data = []
        for row in data_rows:
            try:
                cell = row[col_idx].strip()
            except IndexError:
                # 这一行不够长，没有该单元格，视为空
                cell = ""

            if cell:  # cell非空字符串才执行
                col_data.append(float(cell))
        columns.append(col_data)
    return columns

def build_combinations_two(Sr0, Y0, D_Sr, D_Y):
    """
    生成笛卡尔组合：Sr0[i],Y0[i] × D_Sr[j] × D_Y[k]
    返回 (n_comb, 4) array: [sr, y, dsr, dy]
    """
    Sr0 = np.asarray(Sr0)
    Y0  = np.asarray(Y0)
    D_Sr= np.asarray(D_Sr)
    D_Y = np.asarray(D_Y)

    comb_list = []
    for i in range(len(Sr0)):
        sr = Sr0[i]
        y  = Y0[i]
        for dsr in D_Sr:
            for dy in D_Y:
                comb_list.append([sr, y, dsr, dy])
    return np.array(comb_list)
def cartesian_concat(A:np.ndarray, B:np.ndarray) -> np.ndarray:
    """
    笛卡尔积组合：A(N,na) 与 B(M,nb)
    返回 (N*M, na+nb), 每一行 = A[i] 拼接 B[j]
    """
    N, na = A.shape
    M, nb = B.shape

    # A每一行重复M次
    A_expanded = np.repeat(A, M, axis=0)
    # B整体平铺，循环N次
    B_tiled = np.tile(B, (N, 1))

    out = np.hstack([A_expanded, B_tiled])
    return out
def T_crystallization(f, mH2O):
    mH2O_0 = 2.0; mH2O_1 = 3.0
    T0 = 6103.378 * f ** 4 - 11602.36 * f ** 3 + 6635.159 * f ** 2 - 390.705 * f + 621.937 + 273.15  # 单位为K
    T1 = 5755.415 * f ** 4 - 11194.252 * f ** 3 + 6771.707 * f ** 2 - 616.222 * f + 630.821 + 273.15  # 单位为K
    T = T0 + (T1 - T0) / (mH2O_1 - mH2O_0) * (mH2O - mH2O_0)
    return T
def KD (f, mH2O):
    """
    求解挥发份再磷灰石与熔体之间的交换系数
    :param f:
    :param mH2O:
    :return:
    """
    mH2O_0 = 2.0; mH2O_1 = 3.0
    T0 = 6103.378 * f**4 - 11602.36 * f**3 + 6635.159 * f**2 - 390.705 * f + 621.937 + 273.15 # 单位为K
    T1 = 5755.415 * f**4 - 11194.252 * f**3 + 6771.707 * f**2 - 616.222 * f + 630.821 + 273.15 # 单位为K
    T = T0 + (T1-T0)/(mH2O_1-mH2O_0)*(mH2O-mH2O_0)
    KD_OH_F_W_DS = np.exp(-1 / (T * 8.314) * (94600 - 40 * T + 1000 * (7 * (0.045-0.6587) - 11 * 0.2963))) # Average: XF=0.045, XCl=0.2963, XOH=0.6587
    KD_OH_Cl_W_DS = np.exp(-1 / (T * 8.314) * (72900-34*T+1000*(5*(0.2963-0.6587)-10*0.045))) # Average:
    KD_OH_F_E_DS = np.exp(-1 / (T * 8.314) * (94600 - 40 * T + 1000 * (7*(0.256-0.5324)-11*0.21145))) # Average: XF=0.256, XCl=0.21145, XOH=0.5324
    KD_OH_Cl_E_DS = np.exp(-1 / (T * 8.314) * (72900-34*T+1000*(5*(0.21145-0.53255)-10*0.256))) # Average:
    KD_OH_F_E_PTS = np.exp(-1 / (T * 8.314) * (94600 - 40 * T + 1000 * (7*(0.81763-0.17423)-11*0.00814)))  # Average: XF=0.81763, XCl=0.00814, XOH=0.17243
    KD_OH_Cl_E_PTS = np.exp(-1 / (T * 8.314) * (72900-34*T+1000*(5*(0.00814-0.17423)-10*0.81763))) # Average:
    """
    if T >= (1000+273.15):
        KD_OH_F = (KD_OH_F_W_DS+KD_OH_F_E_DS)/2
        KD_OH_Cl = (KD_OH_Cl_W_DS+KD_OH_Cl_E_DS)/2
    elif (900+273.15) <= T < (1000+273.15):
        T_seg = (T-1000-273.15)/(900-1000)
        KD_OH_F = KD_OH_F_E_PTS * T_seg +(KD_OH_F_W_DS + KD_OH_F_E_DS) / 2 * (1-T_seg)
        KD_OH_Cl = KD_OH_Cl_E_PTS * T_seg + (KD_OH_Cl_W_DS + KD_OH_Cl_E_DS) / 2 * (1 - T_seg)
    else:
        KD_OH_F = KD_OH_F_E_PTS
        KD_OH_Cl = KD_OH_Cl_E_PTS
    """
    
    KD_OH_F = (KD_OH_F_W_DS + KD_OH_F_E_DS) / 2
    KD_OH_Cl = (KD_OH_Cl_W_DS + KD_OH_Cl_E_DS) / 2
    """
    KD_OH_F = KD_OH_F_E_PTS
    KD_OH_Cl = KD_OH_Cl_E_PTS
    
    
    KD_OH_F = 10 ** (1.259-3853.61 / T)
    KD_OH_Cl =10 ** (1.0729-2938 / T)
    """

    return KD_OH_F, KD_OH_Cl

def X_OH_calc(T, X_H2O):
   #  lnK1 =(-13.869 + 0.0002474 * P) + (3890.3 - 0.3948 * P) / T
   # K1 = np.exp(lnK1)
   K2 = 6.53 * np.exp(-3100 / T)
   Term1 = (X_H2O**2)*8; Term2 = 8*X_H2O; Term3 = K2*(1-2* X_H2O); Term4 = (K2*(1-2*X_H2O))**2; Term5 = 16*K2*X_H2O*(1-X_H2O)
   XH2Om = Term1/(Term2+Term3+((Term4+Term5)**0.5))
   XOH = (X_H2O-XH2Om)*2
   return XOH
def KD_fluorine(T, XOH, XF, XCl): # T的单位为K
    KD_OH_F = np.exp(-1 / (T * 8.314) * (94600 - 40 * T + 1000 * (7 * (XF - XOH) - 11 * XCl)))  # Average: XF=0.256, XCl=0.21145, XOH=0.5324
    KD_OH_Cl = np.exp(-1 / (T * 8.314) * (72900 - 34 * T + 1000 * (5 * (XCl - XOH) - 10 * XF)))  # Average:
    return KD_OH_F, KD_OH_Cl


def calc_ap_chem(T_K, X_F, X_Cl, XOH_low_fluorine, XF_low_fluorine, XCl_low_fluorine, F_low_fluorine, mH2O, D_H2O_fluid):
    # 计算分配系数
    # KD_OH_F, KD_OH_Cl = KD_fluorine(T_K, XOH_low_fluorine, XF_low_fluorine, XCl_low_fluorine)

    KD_OH_F = 10 ** (1.259 - 3853.61 / T_K)
    KD_OH_Cl = 10 ** (1.0729 - 2938 / T_K)

    # x_h2o = 5.0 / 18 / 1.70
    # m_H2O = np.full(len(F_low_fluorine), mH2O)
    # m_H2O = mH2O + (6.0 - mH2O) * F_low_fluorine
    m_H2O = mH2O * F_low_fluorine ** (D_H2O_fluid - 1)
    X_H2O = m_H2O / 18.01528 / 1.7
    # X_OH_800 = X_OH_calc(1073.15, X_H2O); X_OH_900 = X_OH_calc(1173.15, X_H2O); X_OH_1000 = X_OH_calc(1273.15, X_H2O)
    X_OH_T = X_OH_calc(T_K, X_H2O = X_H2O)
    # 熔体比值
    XF_XOH_melt = X_F / X_OH_T
    XCl_XOH_melt = X_Cl / X_OH_T

    # 磷灰石比值
    XF_XOH_Ap = XF_XOH_melt / KD_OH_F
    XCl_XOH_Ap = XCl_XOH_melt / KD_OH_Cl

    # 磷灰石摩尔分数 XF+XCl+XOH = 1
    denom = 1 + XF_XOH_Ap + XCl_XOH_Ap
    XOH_Ap = 1 / denom
    XCl_Ap = XCl_XOH_Ap / denom
    XF_Ap = XF_XOH_Ap / denom

    # 磷灰石分子式质量分母 Ca5(PO4)3(F,Cl,OH)
    mol_Ca = 5 * 40.078
    mol_P = 3 * 30.974
    mol_O = 12 * 15.999
    mol_denominator = mol_Ca + mol_P + mol_O + XF_Ap * 18.998 + XCl_Ap * 35.45 + XOH_Ap * 17.007

    # 质量百分数 wt%
    m_H2O = XOH_Ap * 18.007 / mol_denominator * 100
    m_Cl = XCl_Ap * 35.45 / mol_denominator * 100
    m_F = XF_Ap * 18.998 / mol_denominator * 100

    return XCl_XOH_Ap, m_H2O, m_Cl, m_F

def colored_curve(ax, x, y, c, cmap, norm):
    """
    用 LineCollection 画一条沿程变色曲线。
    x, y, c 均为一维数组，长度相同；第 j 段颜色由 c[j] 决定。
    """
    # 把 (x[j], y[j]) 和 (x[j+1], y[j+1]) 组成线段
    points = np.column_stack([x, y])
    segments = np.array([points[j:j+2] for j in range(len(points) - 1)])

    # 每一段的颜色值：取线段两端 c 的平均（也可以直接用 c[j]）
    seg_colors = (c[:-1] + c[1:]) / 2.0

    lc = LineCollection(segments, cmap=cmap, norm=norm,
                        linewidths=0.25)
    lc.set_array(seg_colors)
    ax.add_collection(lc)
    return lc

def extract_csv(fp1):
    df = pd.read_csv(fp1)
    # 读取三列
    H2O_initial = df["H2O_initial"].values
    D_H2O = df["D_H2O_mineral/melt"].values
    m_H2O_sat = df["m_H2O_saturation"].values

    F_cri = np.full_like(H2O_initial, np.nan)
    mask = (D_H2O != 1) & (H2O_initial != 0)
    F_cri[mask] = (m_H2O_sat[mask] / H2O_initial[mask]) ** (1.0 / (D_H2O[mask] - 1.0))

    # ✅ 核心：用F_cri直接覆盖替换rmse列
    df["F_cri"] = F_cri
    return df

def make_cross_ring_marker(outer_r=1.0, inner_r=0.999, cross_width=0.00001):
    """
    生成【圆环+十字，4个扇形镂空】自定义marker，用于ax.scatter

    Parameters
    ----------
    outer_r : 外圆半径
    inner_r : 内圆半径（圆环挖空大小）
    cross_width : 中心十字条带宽度

    Returns
    -------
    custom_marker : matplotlib.path.Path 对象，直接传给scatter的marker参数
    """
    Path = mpath.Path

    # 1. 外轮廓大圆
    theta = np.linspace(0, 2 * np.pi, 100)
    outer_x = outer_r * np.cos(theta)
    outer_y = outer_r * np.sin(theta)
    outer_verts = np.column_stack([outer_x, outer_y])

    # 2. 内圆（反向顶点，用于挖空）
    inner_x = inner_r * np.cos(theta[::-1])
    inner_y = inner_r * np.sin(theta[::-1])
    inner_verts = np.column_stack([inner_x, inner_y])

    # 3. 水平十字矩形
    h_verts = np.array([
        [-outer_r, -cross_width / 2],
        [outer_r, -cross_width / 2],
        [outer_r, cross_width / 2],
        [-outer_r, cross_width / 2],
        [-outer_r, -cross_width / 2]
    ])
    # 4. 竖直十字矩形
    v_verts = np.array([
        [-cross_width / 2, -outer_r],
        [cross_width / 2, -outer_r],
        [cross_width / 2, outer_r],
        [-cross_width / 2, outer_r],
        [-cross_width / 2, -outer_r]
    ])

    # 合并顶点与绘图指令
    verts = np.concatenate([outer_verts, inner_verts, h_verts, v_verts])
    codes = np.full(len(verts), Path.LINETO)
    codes[0] = Path.MOVETO
    codes[len(outer_verts)] = Path.MOVETO
    codes[len(outer_verts) + len(inner_verts)] = Path.MOVETO
    codes[len(outer_verts) + len(inner_verts) + len(h_verts)] = Path.MOVETO

    custom_marker = Path(verts, codes)
    return custom_marker
def make_ring_with_center_dot_marker(outer_r=1.0, inner_r=0.7, dot_r=0.08):
    """
    生成【圆环 + 中心实心圆点】自定义marker，用于ax.scatter
    Parameters
    ----------
    outer_r : 外环外半径
    inner_r : 外环内半径（控制环的粗细）
    dot_r   : 中心小圆点半径
    Returns
    -------
    custom_marker : matplotlib.path.Path 对象
    """
    Path = mpath.Path

    # -------- 1. 外环：外大圆 + 反向内圆（奇偶规则挖空成环）
    theta = np.linspace(0, 2 * np.pi, 100)
    outer_x = outer_r * np.cos(theta)
    outer_y = outer_r * np.sin(theta)
    outer_verts = np.column_stack([outer_x, outer_y])

    inner_x = inner_r * np.cos(theta[::-1])
    inner_y = inner_r * np.sin(theta[::-1])
    inner_verts = np.column_stack([inner_x, inner_y])

    # -------- 2. 中心小圆点
    theta_dot = np.linspace(0, 2 * np.pi, 50)
    dot_x = dot_r * np.cos(theta_dot)
    dot_y = dot_r * np.sin(theta_dot)
    dot_verts = np.column_stack([dot_x, dot_y])

    # 合并三段子路径
    verts = np.concatenate([outer_verts, inner_verts, dot_verts])
    codes = np.full(len(verts), Path.LINETO)

    # 每个子路径起点设置 MOVETO
    codes[0] = Path.MOVETO
    codes[len(outer_verts)] = Path.MOVETO
    codes[len(outer_verts)+len(inner_verts)] = Path.MOVETO

    custom_marker = Path(verts, codes)
    return custom_marker

def make_ring_cross_gap_marker(outer_r=1.0, inner_r=0.7, cross_len=0.85, lw_scale=0.04):
    """
    marker结构：底层白色实心圆基底 + 上层圆环轮廓 + 上层十字细线
    自带填充底板，后绘制marker自动遮挡前面marker，和原生marker="o"行为一致
    Parameters
    ----------
    outer_r : 外圈半径
    inner_r : 内环半径
    cross_len : 十字线半长
    lw_scale : 十字/圆环线条宽度（Path里面用窄矩形模拟线条）
    """
    Path = mpath.Path
    verts = []
    codes = []

    # ========= 第一层：底层白色实心大圆（负责遮挡！核心！）=========
    theta = np.linspace(0, 2 * np.pi, 100)
    base_x = outer_r * np.cos(theta)
    base_y = outer_r * np.sin(theta)
    base_verts = np.column_stack([base_x, base_y])
    base_codes = np.full(len(base_verts), Path.LINETO)
    base_codes[0] = Path.MOVETO
    verts.append(base_verts)
    codes.append(base_codes)

    # ========= 第二层：外环圆环（外圆+反向内圆）=========
    outer_verts = np.column_stack([outer_r * np.cos(theta), outer_r * np.sin(theta)])
    inner_verts = np.column_stack([inner_r * np.cos(theta[::-1]), inner_r * np.sin(theta[::-1])])
    verts.append(outer_verts)
    codes.append(np.full(len(outer_verts), Path.LINETO))
    codes[-1][0] = Path.MOVETO
    verts.append(inner_verts)
    codes.append(np.full(len(inner_verts), Path.LINETO))
    codes[-1][0] = Path.MOVETO

    # ========= 第三层：水平十字（窄矩形模拟线条）=========
    h_verts = np.array([
        [-cross_len, -lw_scale/2],
        [cross_len, -lw_scale/2],
        [cross_len, lw_scale/2],
        [-cross_len, lw_scale/2],
        [-cross_len, -lw_scale/2]
    ])
    verts.append(h_verts)
    h_code = np.full(len(h_verts), Path.LINETO)
    h_code[0] = Path.MOVETO
    codes.append(h_code)

    # ========= 第四层：竖直十字 =========
    v_verts = np.array([
        [-lw_scale/2, -cross_len],
        [lw_scale/2, -cross_len],
        [lw_scale/2, cross_len],
        [-lw_scale/2, cross_len],
        [-lw_scale/2, -cross_len]
    ])
    verts.append(v_verts)
    v_code = np.full(len(v_verts), Path.LINETO)
    v_code[0] = Path.MOVETO
    codes.append(v_code)

    all_verts = np.concatenate(verts)
    all_codes = np.concatenate(codes)
    return Path(all_verts, all_codes)