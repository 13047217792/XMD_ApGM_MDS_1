import pyMelt as m
import numpy as np
import csv
from pymultinest.run import run
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.interpolate import interp1d
import json
import os
import pandas as pd
class inversion:
    def __init__(self,data,knowns, unknowns,resume=False,livepoints=500,step='D_total', type='No_NTS',name='Eastern'):
        self.resume = resume
        self.data = data
        self.step = step
        self.type = type
        self.name = name
        self.livepoints = livepoints
        self.unknowns = unknowns
        self.knowns = knowns

        if self.step == 'D_total':
            if self.type == 'No_NTS':
                if self.name == 'Eastern':
                    self.var_list = [['Sm_parent','Y_parent','Sm_wallrock', 'Y_wallrock','DSm_mineral/melt_0','DY_mineral/melt_0', 'r'], # 深部
                                    ['DSm_mineral/melt_1', 'DY_mineral/melt_1','F0'], # stage a
                                    ['DSm_mineral/melt_2','DY_mineral/melt_2', 'F1'], # stage b
                                    ['DSm_mineral/melt_3','DY_mineral/melt_3', 'F2'], # stage c0
                                    ['DSm_mineral/melt_4','DY_mineral/melt_4', 'F3'], # stage c1
                                    ['DSm_ap/melt','DY_ap/melt']]
                    # 原来：
                    """
                    [['Sm_parent', 'Y_parent','Sm_wallrock', 'Y_wallrock','DSm_mineral/melt_1', 'DY_mineral/melt_1'],
                    ['DSr_mineral/melt_2','DY_mineral/melt_2', 'DSr_ap/melt_2','DY_ap/melt_2', 'F1'],
                    ['DSr_mineral/melt_3','DY_mineral/melt_3', 'DSr_ap/melt_3','DY_ap/melt_3', 'F2'],
                    ['DSr_mineral/melt_4','DY_mineral/melt_4', 'DSr_ap/melt_4','DY_ap/melt_4', 'F3']]
                    """
                    # 修改：
                    """
                    [['Sr_parent','Y_parent','DSr_mineral/melt_0','DY_mineral/melt_0', 'r'], # 深部
                    ['DSr_mineral/melt_1', 'DY_mineral/melt_1','F0'], # stage a
                    ['DSr_mineral/melt_2','DY_mineral/melt_2', 'F1'], # stage b
                    ['DSr_mineral/melt_3','DY_mineral/melt_3', 'F2'], # stage c0
                    ['DSr_mineral/melt_4','DY_mineral/melt_4', 'F3'], # stage c1
                    ['DSr_ap/melt','DY_ap/melt']]
                    """
                else:
                    self.var_list = [['Sm_parent','Y_parent','Sm_wallrock', 'Y_wallrock','DSm_mineral/melt_0','DY_mineral/melt_0', 'r'], # 深部
                                    ['DSm_mineral/melt_1', 'DY_mineral/melt_1','F0'], # stage a
                                    ['DSm_mineral/melt_2','DY_mineral/melt_2', 'F1'], # stage b
                                    ['DSm_ap/melt','DY_ap/melt']]

            else:
                if self.name == 'Eastern':
                    self.var_list = [['Sm_parent','Y_parent','Sm_wallrock', 'Y_wallrock','DSm_mineral/melt_0','DY_mineral/melt_0', 'r'], # 深部
                                    ['DSm_mineral/melt_1', 'DY_mineral/melt_1','F0'], # stage a
                                    ['DSm_mineral/melt_2','DY_mineral/melt_2', 'F1'], # stage b
                                    ['DSm_ap/melt', 'DY_ap/melt']]
                else:
                    if self.type == 'Cu_NTS':
                        self.var_list = [
                            ['Sm_parent', 'Y_parent','Sm_wallrock', 'Y_wallrock','DSm_mineral/melt_0', 'DY_mineral/melt_0', 'r'],  # 深部
                            ['DSm_mineral/melt_CuNTS_1', 'DY_mineral/melt_CuNTS_1', 'F_CuNTS_0'],# PTS stage a
                            ['DSm_mineral/melt_CuNTS_2', 'DY_mineral/melt_CuNTS_2', 'F_CuNTS_1'],
                            ['DSm_ap/melt_CuNTS', 'DY_ap/melt_CuNTS']]
                    else:
                        self.var_list = [
                            ['Sm_parent','Y_parent','Sm_wallrock', 'Y_wallrock', 'DSm_mineral/melt_0','DY_mineral/melt_0', 'r'], # 深部
                            ['DSm_mineral/melt_NTS_1', 'DY_mineral/melt_NTS_1','F_NTS_0'],  # PTS stage a
                            ['DSm_mineral/melt_NTS_2', 'DY_mineral/melt_NTS_2', 'F_NTS_1'],  # PTS stage b
                            ['DSm_ap/melt_NTS', 'DY_ap/melt_NTS']]
        elif self.step == 'D_ind':
            if self.name == 'Eastern':
                self.var_list = []
            else:
                self.var_list = []
        else:
            if self.name == 'Eastern':
                self.var_list = []
            else:
                self.var_list = []

        self.unknowns_list = list()
        for row in self.var_list:
            for item in row:
                self.unknowns_list.append(item)
        self.parameters = len(self.unknowns_list)

    def prior(self, cube, ndim, nparams):
        i = 0  # 全局参数计数器，遍历全部维度
        # var_list与unknowns按下标phase_idx一一对应
        for phase_idx, var_group in enumerate(self.var_list):
            phase_dict = self.unknowns[phase_idx]
            for param_name in var_group:
                if param_name in phase_dict:
                    dist_type, dist_args = phase_dict[param_name]
                    u = cube[i]  # MultiNest 0‑1单位采样值
                    if dist_type == 'uni':
                        low, high = dist_args
                        cube[i] = low + u * (high - low)
                    elif dist_type == 'norm':
                        mu, sigma = dist_args
                        cube[i] = norm(loc=mu, scale=sigma).ppf(u)
                    elif dist_type == 'loguni':
                        low, high = dist_args
                        cube[i] = 10 ** (np.log10(low) + u * (np.log10(high) - np.log10(low)))
                    elif dist_type == 'lognorm':
                        mu_log, sigma_log = dist_args
                        cube[i] = lognorm(s=sigma_log, scale=np.exp(mu_log)).ppf(u)
                    i += 1
        return cube
    def loglike_function(self, X_data, X_prediction, sigma):
        """
        chi2 = np.average((X_prediction - X_data) ** 2 / (sigma ** 2))
        log_l = -0.5 * (chi2 + np.log(2 * np.pi * sigma ** 2))
        """
        chi2 = np.sum((X_prediction - X_data)**2 / (sigma**2))
        n_obs = len(X_prediction)
        log_l = -0.5 * (chi2 + n_obs * np.log(2 * np.pi * sigma ** 2))

        return log_l

    def extract_csv_columns(self, file_path):
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

        # 列数取第一行（表头）的长度
        n_cols = len(rows[0])
        # 跳过表头行，按列聚合
        columns = []
        for i in range(n_cols):
            col_data = [row[i] for row in rows[1:] if i < len(row)]
            columns.append(col_data)

        return columns

    def loglike(self,cube,ndim,nparams):
        """
        if self.step == 'D_total':
           if self.type == "No_NTS':
        """
        Y_data_1, Y_data_2, Y_data_3, Y_data_4, Y_data_5, Y_data_6, Y_data_7 = (self.data[0], self.data[2], self.data[4], self.data[6],
                                                                                self.data[8], self.data[10], self.data[12])
        Sm_data_1, Sm_data_2, Sm_data_3, Sm_data_4, Sm_data_5, Sm_data_6, Sm_data_7 = (self.data[1], self.data[3], self.data[5], self.data[7],
                                                                                self.data[9], self.data[11], self.data[13])

        n_rows = len(self.var_list)
        row_lengths = [len(sub_list) for sub_list in self.var_list]
        max_cols = max(row_lengths)
        x = np.zeros((n_rows,max_cols)) # 需要修改
        i = 0
        for phase_idx, var_group in enumerate(self.var_list):
            phase_dict = self.unknowns[phase_idx]
            for col_idx, param_name in enumerate(var_group):
                if param_name in phase_dict:
                    x[phase_idx, col_idx] = cube[i]
                    i+=1
                else:
                    x[phase_idx, col_idx] = self.knowns[phase_idx][param_name]
        likelihood = 0
        Sm0 = x[0, 0] * x[1, 2] ** (x[0, 4] / (1 - x[0, 6]) - 1) + x[0, 6] / (x[0, 6] + x[0, 4] - 1) * x[0, 2] * (
                    1 - x[1, 2] ** (x[0, 4] / (1 - x[0, 6]) - 1))
        Y0 = x[0, 1] * x[1, 2] ** (x[0, 5] / (1 - x[0, 6]) - 1) + x[0, 6] / (x[0, 6] + x[0, 5] - 1) * x[0, 3] * (
                    1 - x[1, 2] ** (x[0, 5] / (1 - x[0, 6]) - 1))
        if self.type == "No_NTS" and self.name == "Eastern":
            Sm1 = Sm0 * (x[2, 2] ** (x[1, 0] - 1.0)); Y1 = Y0 * (x[2, 2] ** (x[1, 1] - 1.0)) # 阶段b的初始值
            Sm2 = Sm1 * (x[3, 2] ** (x[2, 0] - 1.0)); Y2 = Y1 * (x[3, 2] ** (x[2, 1] - 1.0)) # 阶段c0的初始值
            Sm3 = Sm1 * (x[4, 2] ** (x[2, 0] - 1.0)); Y3 = Y1 * (x[4, 2] ** (x[2, 1] - 1.0)) # 阶段c1的初始值
            """
            Sr1 = x[0, 0] * (x[1, 4] ** (x[0, 2] - 1.0)); Y1 = x[0, 1] * (x[1, 4] ** (x[0, 3] - 1.0)) # stage b时的起点
            Sr2 = Sr1 * (x[2, 4] ** (x[1, 0] - 1.0)); Y2 = Y1 * (x[2, 4] ** (x[1, 1] - 1.0)) # stage c0时的起点
            Sr3 = Sr1 * (x[3, 4] ** (x[1, 0] - 1.0)); Y3 = Y1 * (x[3, 4] ** (x[1, 1] - 1.0)) # stage c1时的起点
            """
            # Y_data_1_new = np.delete(Y_data_1, 8) # 为了不影响反演，剔除第9个偏离显著的数字
            # Sm_data_1_new = np.delete(Sm_data_1, 8)
            F_1 = (Y_data_1 / (Y1 * x[5, 1])) ** (1 / (x[2, 1] - 1 + 1e-8))
            predicted_Sm_1 = Sm1 * x[5, 0] * F_1 ** (x[2, 0] - 1)
            sigma_1 = 100
            likelihood = likelihood + self.loglike_function(Sm_data_1, predicted_Sm_1, sigma_1)
            Y_data_2_new = np.delete(Y_data_2, 4)  # 为了不影响反演，剔除第0、4、6个偏离显著的数字
            Sm_data_2_new = np.delete(Sm_data_2, 4)
            F_2 = (Y_data_2_new / (Y2 * x[5, 1])) ** (1 / (x[3, 1] - 1 + 1e-8))
            predicted_Sm_2 = Sm2 * x[5, 0] * F_2 ** (x[3, 0] - 1)
            sigma_2 = 20
            likelihood = likelihood + self.loglike_function(Sm_data_2_new, predicted_Sm_2, sigma_2)
            Y_data_3_new = np.delete(Y_data_3, [0, 1, 4, 6]) # 为了不影响反演，剔除第0、4、6个偏离显著的数字
            Sm_data_3_new = np.delete(Sm_data_3, [0, 1, 4, 6])
            F_3 = (Y_data_3_new / (Y3 * x[5, 1])) ** (1 / (x[4, 1] - 1 + 1e-8))
            predicted_Sm_3 = Sm3 * x[5, 0] * F_3 ** (x[4, 0] - 1)
            sigma_3 = 10
            likelihood = likelihood + self.loglike_function(Sm_data_3_new, predicted_Sm_3, sigma_3)
            """
            F_1 = (Y_data_1/(Y1*x[1,3]))**(1/(x[1,1]-1+1e-8))
            predicted_Sr_Y_1 = Sr1*x[1,2]/(Y1*x[1,3])*F_1**(x[1,0]-x[1,1])
            sigma_1 = 1
            likelihood = likelihood + self.loglike_function(Sr_Y_data_1, predicted_Sr_Y_1, sigma_1)
            F_2 = (Y_data_2/(Y2*x[2, 3]))**(1/(x[2, 1]-1+1e-8))
            predicted_Sr_Y_2 = Sr2*x[2, 2]/(Y2*x[2, 3])*F_2**(x[2, 0]-x[2, 1])
            sigma_2 = 1
            likelihood = likelihood + self.loglike_function(Sr_Y_data_2, predicted_Sr_Y_2, sigma_2)
            F_3 = (Y_data_3/(Y3*x[3, 3]))**(1/(x[3, 1] - 1+1e-8))
            predicted_Sr_Y_3 = Sr3*x[3, 2]/(Y3*x[3, 3]) *F_3**(x[3, 0] - x[3, 1])
            sigma_3 = 1
            likelihood = likelihood + self.loglike_function(Sr_Y_data_3, predicted_Sr_Y_3, sigma_3)
            """
        elif (self.type == "No_NTS" and self.name == "Western") or (self.type == "NTS" and self.name == "Eastern"):
            Sm1 = Sm0 * (x[2, 2] ** (x[1, 0] - 1.0)); Y1 = Y0 * (x[2, 2] ** (x[1, 1] - 1.0))  # 阶段b的初始值
            if self.type == "No_NTS" and self.name == "Western":
                F_1 = (Y_data_4 / (Y1 * x[3, 1])) ** (1 / (x[2, 1] - 1 + 1e-8))
                predicted_Sm_4 = Sm1 * x[3, 0] * F_1 ** (x[2, 0] - 1)
                sigma_4 = 20
                likelihood = likelihood + self.loglike_function(Sm_data_4, predicted_Sm_4, sigma_4)
            else:
                F_1 = (Y_data_5 / (Y1 * x[3, 1])) ** (1 / (x[2, 1] - 1 + 1e-8))
                predicted_Sm_5 = Sm1 * x[3, 0] * F_1 ** (x[2, 0] - 1)
                sigma_5 = 20
                likelihood = likelihood + self.loglike_function(Sm_data_5, predicted_Sm_5, sigma_5)
        else:
            # PTS
            if self.type == 'Cu_NTS':
                Sm0_PTS = Sm0; Y0_PTS = Y0
                Sm1_PTS = Sm0_PTS * (x[2, 2] ** (x[1, 0] - 1.0)); Y1_PTS = Y0_PTS * (x[2, 2] ** (x[1, 1] - 1.0))  # PTS阶段b的初始值
                predicted_Sm_6 = Sm1_PTS * x[3, 0] * (Y_data_6 / (Y1_PTS * x[3, 1])) ** ((x[2, 0] - 1) / (x[2, 1] - 1 + 1e-8))
                sigma_6 = 30
                likelihood = likelihood + self.loglike_function(Sm_data_6, predicted_Sm_6, sigma_6)
            else:
                Sm0_NTS = Sm0; Y0_NTS = Y0
                Sm1_NTS = Sm0_NTS * (x[2, 2] ** (x[1, 0] - 1.0)); Y1_NTS = Y0_NTS * (x[2, 2] ** (x[1, 1] - 1.0))  # NTS阶段b的初始值
                predicted_Sm_7 = Sm1_NTS * x[3, 0] * (Y_data_7 / (Y1_NTS * x[3, 1])) ** ((x[2, 0] - 1) / (x[2, 1] - 1 + 1e-8))
                sigma_7 = 5
                likelihood = likelihood + self.loglike_function(Sm_data_7, predicted_Sm_7, sigma_7)
            # NTS
            """
            Sr0_NTS = x[0, 0] * x[1, 3] ** (x[0, 4] / (1 - x[0, 6]) - 1) + x[0, 6] / (x[0, 6] + x[0, 4] - 1) * x[0, 2] * (
                    1 - x[1, 3] ** (x[0, 4] / (1 - x[0, 6]) - 1))
            Y0_NTS = x[0, 1] * x[1, 3] ** (x[0, 5] / (1 - x[0, 6]) - 1) + x[0, 6] / (x[0, 6] + x[0, 5] - 1) * x[0, 3] * (
                    1 - x[1, 3] ** (x[0, 5] / (1 - x[0, 6]) - 1))
            Sr1_NTS = Sr0_NTS * (x[2, 3] ** (x[1, 0] - 1.0)); Y1_NTS = Y0_NTS * (x[2, 3] ** (x[1, 1] - 1.0))  # NTS阶段b的初始值
            predicted_Sr_Y_7 = Sr1_NTS * x[3, 0] / (Y1_NTS * x[3, 1]) * (Y_data_7 / (Y1_NTS * x[3, 1])) ** ((x[2, 0] - x[2, 1]) / (x[2, 1] - 1 + 1e-8))
            sigma_7 = 60
            likelihood = likelihood + self.loglike_function(Sr_Y_data_7, predicted_Sr_Y_7, sigma_7)
            """


        return likelihood

    def run_multinest(self,verbose=True):
        if not os.path.exists(self.name): os.mkdir(self.name)
        result = run(LogLikelihood=self.loglike,Prior=self.prior,n_dims=self.parameters,
                        outputfiles_basename=self.name + '/' + self.name + '_' + self.type,resume=self.resume,
                        verbose=verbose,n_live_points=self.livepoints,
                        evidence_tolerance=0.1, sampling_efficiency=0.3) # 高维参数设置, evidence_tolerance=0.1, sampling_efficiency=0.3
        json.dump(self.unknowns_list, open(self.name+'/params.json','w'))
        return result