'''
Author: your name
Date: 2022-04-22 13:21:17
LastEditTime: 2022-04-25 16:26:39
LastEditors: Please set LastEditors
Description: 
'''

from typing import (List, Tuple, Dict, Callable, Union)
from collections import namedtuple
from alphalens.utils import print_table
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号
"""画图相关"""

# def plotting_dichotomy_res(res_nametuple: namedtuple):

#     cols = 'IC Mean,mean_ret'.split(',')
#     cols1 = 'IC Std.,Risk-Adjusted IC,t-stat(IC),p-value(IC),IC Skew,IC Kurtosis'.split(
#         ',')

#     ic_frame = res_nametuple.ic_info_table
#     style_df = (ic_frame.style.format('{:.2%}',
#                                       subset=cols).format('{:.4f}',
#                                                           subset=cols1))

#     print_table(style_df)
#     size = ic_frame.shape[1]

#     if size % 2 == 0:
#         rows = size // 2
#     else:
#         rows = size // 2 + 1

#     gf = GridFigure(rows=rows, cols=2, figsize=(18, rows * 4))
#     for name, ser in res_nametuple.ic_info_table['mean_ret'].groupby(level=0):

#         ser = ser.reset_index(level=0)

#         ser.plot.bar(ax=gf.next_cell(), title=name)

#     plt.show()
#     gf.close()

#     gf = GridFigure(rows=rows, cols=2, figsize=(18, rows * 5))
#     for name, ser in res_nametuple.quantile_cum_returns.items():

#         ser.plot(ax=gf.next_cell(), title=name)

#     plt.show()
#     gf.close()


def plotting_dichotomy_res(res_nametuple: namedtuple):

    cols = 'IC Mean,mean_ret'.split(',')
    cols1 = 'IC Std.,Risk-Adjusted IC,t-stat(IC),p-value(IC),IC Skew,IC Kurtosis'.split(
        ',')

    ic_frame = res_nametuple.ic_info_table
    style_df = (ic_frame.style.format('{:.2%}',
                                      subset=cols).format('{:.4f}',
                                                          subset=cols1))

    print_table(style_df)
    size = len(ic_frame.index.levels[0])

    if size % 2 == 0:
        rows = size // 2
    else:
        rows = size // 2 + 1

    gf = GridFigure(rows=rows, cols=2, figsize=(18, rows * 4))

    for name, ser in res_nametuple.ic_info_table['mean_ret'].groupby(level=0):

        ser = ser.reset_index(level=0)

        ser.plot.bar(ax=gf.next_cell(), title=name)

    plt.show()
    gf.close()

    gf = GridFigure(rows=rows, cols=2, figsize=(18, rows * 5))
    for name, ser in res_nametuple.quantile_cum_returns.groupby(
            level='factor_name'):

        ax = ser.loc[name].plot(ax=gf.next_cell(), title=name)
        ax.set_ylabel('累计收益率')

    plt.show()
    gf.close()


class GridFigure(object):
    """
    It makes life easier with grid plots
    """
    def __init__(self, rows, cols, figsize: Tuple = None):
        self.rows = rows
        self.cols = cols
        if figsize is None:
            size = (14, rows * 7)
            self.fig = plt.figure(figsize=size)
        else:
            self.fig = plt.figure(figsize=figsize)
        self.gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.3)
        self.curr_row = 0
        self.curr_col = 0

    def next_row(self):
        if self.curr_col != 0:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, :])
        self.curr_row += 1
        return subplt

    def next_cell(self):
        if self.curr_col >= self.cols:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, self.curr_col])
        self.curr_col += 1
        return subplt

    def close(self):
        plt.close(self.fig)
        self.fig = None
        self.gs = None