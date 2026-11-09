import numpy as np
import csv
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
def T(f, mH2O):
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
    if T >= (900+273.15):
        KD_OH_F = (KD_OH_F_W_DS+KD_OH_F_E_DS)/2
        KD_OH_Cl = (KD_OH_Cl_W_DS+KD_OH_Cl_E_DS)/2
    elif (800+273.15) <= T < (900+273.15):
        T_seg = (T-900)/(800-900)
        KD_OH_F = KD_OH_F_E_PTS * T_seg +(KD_OH_F_W_DS + KD_OH_F_E_DS) / 2 * (1-T_seg)
        KD_OH_Cl = KD_OH_Cl_E_PTS * T_seg + (KD_OH_Cl_W_DS + KD_OH_Cl_E_DS) / 2 * (1 - T_seg)
    else:
        KD_OH_F = KD_OH_F_E_PTS
        KD_OH_Cl = KD_OH_Cl_E_PTS
    return KD_OH_F, KD_OH_Cl

def X_OH(T, X_H2O):
   #  lnK1 =(-13.869 + 0.0002474 * P) + (3890.3 - 0.3948 * P) / T
   # K1 = np.exp(lnK1)
   K2 = 6.53 * ((np.exp(1)) ^ (-3100 / T))
   Term1 = (X_H2O**2)*8; Term2 = 8*X_H2O; Term3 = K2*(1-2* X_H2O); Term4 = (K2*(1-2*X_H2O))^2; Term5 = 16*K2*X_H2O*(1-X_H2O)
   XH2Om = Term1/(Term2+Term3+((Term4+Term5)**0.5))
   XOH = (X_H2O-XH2Om)*2
   return XOH
def KD_fluorine(T, XOH, XF, XCl): # T的单位为K
    KD_OH_F = np.exp(-1 / (T * 8.314) * (94600 - 40 * T + 1000 * (7 * (XF - XOH) - 11 * XCl)))  # Average: XF=0.256, XCl=0.21145, XOH=0.5324
    KD_OH_Cl = np.exp(-1 / (T * 8.314) * (72900 - 34 * T + 1000 * (5 * (XCl - XOH) - 10 * XF)))  # Average:
    return KD_OH_F, KD_OH_Cl

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
