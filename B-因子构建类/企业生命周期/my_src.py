'''
Author: Hugo
Date: 2022-04-18 16:53:10
LastEditTime: 2022-04-20 14:22:57
LastEditors: Please set LastEditors
Description: 
'''
import _imports
from typing import (List, Tuple, Dict, Callable, Union)

from Hugos_tools.Tdays import (Tdaysoffset, get_trade_period)
from Hugos_tools.BuildStockPool import Filter_Stocks

from my_factor import (quadrant, VolAvg, VolCV, RealizedSkewness, ILLIQ,
                       Operatingprofit_FY1, BP_LR, EP_Fwd12M, Sales2EV,
                       Gross_profit_margin_chg, Netprofit_chg)

from jqfactor import calc_factors
from jqdata import *

from tqdm import tqdm_notebook

import alphalens as al
import pandas as pd
import numpy as np
import empyrical as ep
from scipy import stats
from collections import (namedtuple, defaultdict)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号
"""划分象限"""


def get_daily_quadrant(watch_date: str,
                       method: str = 'ols',
                       is_scaler: bool = False) -> pd.DataFrame:
    """获取当日象限划分

    Parameters
    ----------
    watch_date : str
        观察日
    method : str, optional
        划分方式-主要是营收增长率的处理方式,quadrant类,详见, by default 'ols'
    is_scaler : bool, optional
        是否归一化处理, by default False

    Returns
    -------
    pd.DataFrame
        象限划分
    """
    trade = pd.to_datetime(watch_date)
    stock_pool_func = Filter_Stocks('A', trade)
    factor = quadrant()
    factor.method = method
    factor.is_scaler = is_scaler

    return calc_factors(stock_pool_func.securities, [factor], trade,
                        trade)['quadrant']


def get_daily_dichotomy(watch_date: str,
                        method: str = 'ols',
                        is_scaler: bool = False) -> pd.DataFrame:
    """获取二分象限

    Parameters
    ----------
    watch_date : str
        观察日
    method : str, optional
        划分方式-主要是营收增长率的处理方式,quadrant类,详见, by default 'ols'
    is_scaler : bool, optional
        是否归一化处理, by default False

    Returns
    -------
    pd.DataFrame
        划分
    """
    trade = pd.to_datetime(watch_date)
    stock_pool_func = Filter_Stocks('A', trade)
    factor = quadrant()
    factor.method = method
    factor.is_scaler = is_scaler

    calc_factors(stock_pool_func.securities, [factor], trade,
                 trade)['quadrant']
    return factor.dichotomy


def get_dichotomy(start: str,
                  end: str,
                  method: str = 'ols',
                  is_scaler: bool = False) -> pd.DataFrame:
    """获取区间象限划分合集

    Parameters
    ----------
    start : str
        开始日
    end : str
        结束日
    method : str, optional
        划分方式-主要是营收增长率的处理方式,quadrant类,详见, by default 'ols'
    is_scaler : bool, optional
        是否归一化处理, by default False
        
    Returns
    -------
    pd.DataFrame
        象限划分
    """
    periods = get_trade_period(start, end, 'ME')

    tmp = {}
    for trade in tqdm_notebook(periods, desc='划分高低端象限'):

        tmp[trade] = get_daily_dichotomy(trade, method, is_scaler)

    df = pd.concat(tmp, sort=True)
    return df


def get_quadrant(start: str,
                 end: str,
                 method: str = 'ols',
                 is_scaler: bool = False) -> pd.DataFrame:
    """获取区间象限划分合集

    Parameters
    ----------
    start : str
        开始日
    end : str
        结束日
    method : str, optional
        划分方式-主要是营收增长率的处理方式,quadrant类,详见, by default 'ols'
    is_scaler : bool, optional
        是否归一化处理, by default False
        
    Returns
    -------
    pd.DataFrame
        象限划分
    """
    periods = get_trade_period(start, end, 'ME')

    tmp = []
    for trade in tqdm_notebook(periods, desc='划分四象限'):

        tmp.append(get_daily_quadrant(trade, method, is_scaler))

    df = pd.concat(tmp, sort=True)
    return df


"""因子生成"""


def get_pricing(factor_df: pd.DataFrame,
                last_periods: str = None) -> pd.DataFrame:
    """获取价格数据

    Args:
        factor_df (pd.DataFrame): 因子数据  MultiIndex levels-0 date levels-1 code
        last_periods (str, optional): 最后一期数据. Defaults to None.

    Returns:
        pd.DataFrame
    """
    if last_periods is not None:
        periods = factor_df.index.levels[0].tolist() + [
            pd.to_datetime(last_periods)
        ]
    else:
        periods = factor_df.index.levels[0]

    securities = factor_df.index.levels[1].tolist()

    # 获取收盘价
    price_list = list(get_freq_price(securities, periods))
    price_df = pd.concat(price_list)
    pivot_price = pd.pivot_table(price_df,
                                 index='time',
                                 columns='code',
                                 values='close')
    return pivot_price


def get_freq_price(security: Union[List, str], periods: List) -> pd.DataFrame:
    """获取对应频率价格数据

    Args:
        security (Union[List, str]): 标的
        periods (List): 频率

    Yields:
        Iterator[pd.DataFrame]
    """
    for trade in tqdm_notebook(periods, desc='获取收盘价数据'):

        yield get_price(security,
                        end_date=trade,
                        count=1,
                        fields='close',
                        fq='post',
                        panel=False)


def trans2frame(dic: Dict) -> pd.DataFrame:
    def _func(df, col):
        ser = df.iloc[-1]
        ser.name = col
        return ser

    return pd.concat((_func(df, col) for col, df in dic.items()), axis=1)


def get_factors(quandrant_df: pd.DataFrame):

    periods = quandrant_df.index.tolist()
    codes = quandrant_df.columns.tolist()

    tmp = {}

    factors = [
        VolAvg(),
        VolCV(),
        RealizedSkewness(),
        ILLIQ(),
        Operatingprofit_FY1(),
        BP_LR(),
        EP_Fwd12M(),
        Sales2EV(),
        Gross_profit_margin_chg(),
        Netprofit_chg()
    ]

    for trade in tqdm_notebook(periods, desc='获取因子'):

        dic = calc_factors(codes, factors, trade, trade)
        tmp[trade] = trans2frame(dic)

    factor_df = pd.concat(tmp)
    factor_df.index.names = ['date', 'asset']

    return factor_df


"""因子分析相关"""


class get_factor_returns(object):
    def __init__(self, factors: pd.Series, max_loss: float) -> None:
        '''
        输入:factors MuliIndex level0-date level1-asset columns-factors
        '''
        self.factors = factors
        self.factor_name = factors.name
        self.name = self.factor_name
        self.max_loss = max_loss

    def get_calc(self,
                 pricing: pd.DataFrame,
                 periods: Tuple = (1, ),
                 quantiles: int = 5) -> pd.DataFrame:

        preprocessing_factor = al.utils.get_clean_factor_and_forward_returns(
            self.factors,
            pricing,
            periods=periods,
            quantiles=quantiles,
            max_loss=self.max_loss)

        # 预处理好的因子
        self.factors_frame = preprocessing_factor

        # 分组收益
        self.group_returns = pd.pivot_table(preprocessing_factor.reset_index(),
                                            index='date',
                                            columns='factor_quantile',
                                            values=1)

        # 分组累计收益
        self.group_cum_returns = ep.cum_returns(self.group_returns)

    def long_short(self, lower: int = 1, upper: int = 5) -> pd.Series:
        '''
        获取多空收益
        默认地分组为1,高分组为5
        '''
        try:
            self.group_returns
        except NameError:
            raise ValueError('请先执行get_calc')

        self.long_short_returns = self.group_returns[upper] - \
            self.group_returns[lower]
        self.long_short_returns.name = f'{self.name}_excess_ret'

        self.long_short_cum = ep.cum_returns(self.long_short_returns)
        self.long_short_cum.name = f'{self.name}_excess_cum'


def get_information_table(ic_data: pd.DataFrame) -> pd.DataFrame:
    """计算IC的相关信息

    Args:
        ic_data (pd.DataFrame): index-date columns-IC

    Returns:
        pd.DataFrame: index - num columns-IC_Mean|IC_Std|Risk-Adjusted IC|t-stat(IC)|...
    """
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["Risk-Adjusted IC"] = \
        ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)
    return ic_summary_table


def get_quantile_ic(factor_frame: pd.DataFrame) -> pd.DataFrame:
    """获取分组IC相关信息

    Args:
        factor_frame (pd.DataFrame): MultiIndex level0-date level1-asset 

    Returns:
        pd.DataFrame
    """
    tmp = []
    for num, df in factor_frame.groupby('factor_quantile'):

        ic = al.performance.factor_information_coefficient(df, False)
        ic_info = get_information_table(ic)
        #ic_info = ic_info.T
        ic_info.index = [num]
        tmp.append(ic_info)

    return pd.concat(tmp)


def get_factors_res(dichotomy_df: pd.DataFrame, factors_df: pd.DataFrame,
                    pricing: pd.DataFrame, cat_type: Dict) -> Dict:

    res = {}

    for name, dic in cat_type.items():

        res[name] = get_factor_res2namedtuple(dichotomy_df, factors_df,
                                              pricing,
                                              tuple(dic.items())[0])

    return res


def get_factor_res2namedtuple(categories_df: pd.DataFrame,
                              factors_df: pd.DataFrame, pricing: pd.DataFrame,
                              cat_tuple: Tuple) -> namedtuple:
    """计算每个象限的因子收益情况

    Args:
        categories_df (pd.DataFrame): MultiIndex level0-date level1-asset columns-分类情况
        factors_df (pd.DataFrame): 因子分值
        pricing (pd.DataFrame): 价格 index-date columns-codes
        cat_tuple (Tuple): 0-分类筛选表达 1-因子组

    Returns:
        namedtuple
    """
    factors_res = namedtuple(
        'factor_res',
        'factor_data,quantile_returns,quantile_cum_returns,ic_info_table')
    k = cat_tuple[0]
    factor_cols = list(cat_tuple[1])
    sel_idx = categories_df.query(k).index

    test_factor = factors_df.loc[sel_idx, factor_cols]
    test_factor['复合因子'] = test_factor.mean(axis=1)
    factor_data = {}
    quantile_returns = {}
    quantile_cum_returns = {}
    info_dic = {}

    for name, ser in test_factor.items():

        gfr = get_factor_returns(ser.dropna(), 0.25)
        gfr.get_calc(pricing)

        ic_info = get_quantile_ic(gfr.factors_frame)
        cols = al.utils.get_forward_returns_columns(gfr.factors_frame.columns)
        quantile_mean_ret = gfr.factors_frame.groupby(
            'factor_quantile')[cols].mean()
        ic_info['mean_ret'] = quantile_mean_ret
        info_dic[name] = ic_info

        factor_data[name] = gfr.factors_frame
        quantile_returns[name] = gfr.group_returns
        quantile_cum_returns[name] = gfr.group_cum_returns

    quantile_info = pd.concat(info_dic)

    return factors_res(factor_data, quantile_returns, quantile_cum_returns,
                       quantile_info)


CATEGORY = {
    'roe端==0':
    ['VolAvg_20D_240D', 'VolCV_20D', 'RealizedSkewness_240D', 'ILLIQ_20D'],
    'roe端==1': ['Operatingprofit_FY1_R20D'],
    '增长端==1': ['BP_LR', 'EP_Fwd12M', 'Sales2EV'],
    '增长端==0': ['Gross_profit_margin_chg', 'Netprofit_chg']
}


def quadrant_dic():

    dic = {
        'cat_type == 2': ['roe端==0', '增长端==1'],
        'cat_type == 1': ['roe端==1', '增长端==1'],
        'cat_type == 3': ['增长端==0', 'roe端==0'],
        'cat_type == 4': ['roe端==1', '增长端==0']
    }

    # 导入期 cat_type = 2
    # 成长期 cat_type = 1
    # 衰退期 cat_type = 3
    # 成熟期 cat_type = 4

    sub_dic = defaultdict(list)
    out_put = {}
    for label, (name, v) in zip(['导入期', '成长期', '衰退期', '成熟期'], dic.items()):

        for i in v:

            sub_dic[name].extend(CATEGORY[i])

        out_put[label] = sub_dic

    return out_put


def dichotomy_dic():

    label = ['低roe端', '高roe端', '高增长端', '低增长端']

    out_put = {name: {k: v} for name, (k, v) in zip(label, CATEGORY.items())}

    return out_put


"""画图相关"""


def plotting_dichotomy_res(res_nametuple: namedtuple):

    cols = 'IC Mean,mean_ret'.split(',')
    cols1 = 'IC Std.,Risk-Adjusted IC,t-stat(IC),p-value(IC),IC Skew,IC Kurtosis'.split(
        ',')

    style_df = (res_nametuple.ic_info_table.style.format(
        '{:.2%}', subset=cols).format('{:.4f}', subset=cols1))

    al.utils.print_table(style_df)
    size = style_df.shape[1]

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
    for name, ser in res_nametuple.quantile_cum_returns.items():

        #ser = ser.reset_index(level=0)

        ser.plot(ax=gf.next_cell(), title=name)
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