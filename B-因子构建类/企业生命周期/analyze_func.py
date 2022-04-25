'''
Author: your name
Date: 2022-04-22 13:21:17
LastEditTime: 2022-04-25 22:56:20
LastEditors: Please set LastEditors
Description: 
'''
import functools
import alphalens as al
import pandas as pd
import empyrical as ep

from my_scr import (calc_group_ic, add_group, get_group_return,
                    get_information_table)

from typing import (List, Tuple, Dict, Callable, Union)
from collections import namedtuple

# class get_factor_returns(object):
#     def __init__(self, factors: pd.Series, max_loss: float) -> None:
#         '''
#         输入:factors MuliIndex level0-date level1-asset columns-factors
#         '''
#         self.factors = factors
#         self.factor_name = factors.name
#         self.name = self.factor_name
#         self.max_loss = max_loss

#     def get_calc(self,
#                  pricing: pd.DataFrame,
#                  periods: Tuple = (1, ),
#                  quantiles: int = 5) -> pd.DataFrame:

#         preprocessing_factor = al.utils.get_clean_factor_and_forward_returns(
#             self.factors,
#             pricing,
#             periods=periods,
#             quantiles=quantiles,
#             max_loss=self.max_loss)

#         # 预处理好的因子
#         self.factors_frame = preprocessing_factor

#         # 分组收益
#         self.group_returns = pd.pivot_table(preprocessing_factor.reset_index(),
#                                             index='date',
#                                             columns='factor_quantile',
#                                             values=1)

#         # 分组累计收益
#         self.group_cum_returns = ep.cum_returns(self.group_returns)

#     def long_short(self, lower: int = 1, upper: int = 5) -> pd.Series:
#         '''
#         获取多空收益
#         默认地分组为1,高分组为5
#         '''
#         try:
#             self.group_returns
#         except NameError:
#             raise ValueError('请先执行get_calc')

#         self.long_short_returns = self.group_returns[upper] - \
#             self.group_returns[lower]
#         self.long_short_returns.name = f'{self.name}_excess_ret'

#         self.long_short_cum = ep.cum_returns(self.long_short_returns)
#         self.long_short_cum.name = f'{self.name}_excess_cum'


class analyze_factor_res(object):
    def __init__(self,
                 factors: pd.DataFrame,
                 ind_name: Union[str, List] = None,
                 direction: Union[str, Dict] = 'ascending') -> None:
        '''
        输入:factors MuliIndex level0-date level1-asset columns-factors
        ind_name (Union[str, List]): 需要分组的因子名
        direction (Union[str, Dict], optional):设置所有因子的排序方向，'ascending'表示因子值越大分数越高，
        'descending'表示因子值越小分数越;Defaults to 'ascending'.
        '''
        self.factors = factors.copy()
        self.ind_name = ind_name
        self.direction = direction

    def get_calc(self,
                 pricing: pd.DataFrame,
                 quantiles: int = 5) -> pd.DataFrame:
        """数据准备

        Args:
            pricing (pd.DataFrame): index-date columns-code value-close
            quantiles (int, optional): group_num (int, optional): 当为大于等于2的整数时,对股票平均分组;当为(0,0.5)之间的浮点数,
                                对股票分为3组,前group_num%为G01,后group_num%为G02,中间为G03. Defaults to 5.
        """
        next_returns: pd.DataFrame = al.utils.compute_forward_returns(
            pricing, (1, ))

        # 分组
        group_factor = add_group(self.factors,
                                 ind_name=self.ind_name,
                                 group_num=quantiles,
                                 direction=self.direction)

        group_factor['next_ret'] = next_returns[1]
        self.factors['next_ret'] = next_returns[1]

        # 因子分组
        self.group_factor = group_factor

        # 分组收益
        self.group_returns = get_group_return(group_factor)
        self.group_returns.columns.name = '分组'
        # 分组累计收益
        self.group_cum_returns = self.group_returns.groupby(
            level='factor_name').transform(lambda x: ep.cum_returns(x))

        
    def calc_ic(self) -> pd.DataFrame:

        ic_frame = calc_group_ic(self.factors, self.group_factor)
        return ic_frame


"""生成结果"""

# def get_factors_res(dichotomy_df: pd.DataFrame, factors_df: pd.DataFrame,
#                     pricing: pd.DataFrame, cat_type: Dict) -> Dict:

#     res = {}

#     for name, dic in cat_type.items():

#         res[name] = get_factor_res2namedtuple(dichotomy_df, factors_df,
#                                               pricing,
#                                               tuple(dic.items())[0])

#     return res


def get_factor_res(dichotomy: pd.DataFrame, factors: pd.DataFrame,
                   pricing: pd.DataFrame, cat_type: Dict, **kws) -> Dict:
    """获取因子分析报告

    Args:
        dichotomy (pd.DataFrame): 象限区分表 MultiIndex level0-date level1-code
        factors (pd.DataFrame): 因子 MultiIndex level0-date level1-code
        pricing (pd.DataFrame): 价格数据 index-date columns-code values-price
        cat_type (Dict): k-label v- 0-查询 1-选择的因子

    Returns:
        Dict: _description_
    """
    res = {}

    ind_name = kws.get('ind_name', None)
    direction = kws.get('direction', 'ascending')
    group_num = kws.get('group_num', 5)

    func = functools.partial(get_factor_res2namedtuple,
                             factor_df=factors,
                             pricing=pricing,
                             categories_df=dichotomy)

    for name, v in cat_type.items():

        res[name] = func(
            categories_dic={
                'cat_tuple': v,
                'ind_name': ind_name,
                'direction': direction,
                'group_num': group_num
            })

    return res


def get_factor_res2namedtuple(factor_df: pd.DataFrame, pricing: pd.DataFrame,
                              categories_df: pd.DataFrame,
                              categories_dic: Dict) -> namedtuple:
    """计算每个象限的因子收益情况

    Args:
        factors_df (pd.DataFrame): 因子分值
        pricing (pd.DataFrame): 价格 index-date columns-codes
        categories_df(pd.DataFrame):MultiIndex level0-date level1-asset columns-分类情况
        categories_dic (Dict):
            1. cat_tuple (Tuple): 0-分类筛选表达 1-因子组
            2. ind_name同add_group
            3. group_num同add_group
            4. direction同add_group

    Returns:
        namedtuple
    """
    factors_res = namedtuple(
        'factor_res', 'quantile_returns,quantile_cum_returns,ic_info_table')

    # 从categories_dic获取参数
    cat_tuple = categories_dic['cat_tuple']
    ind_name = categories_dic.get('ind_name', None)
    direction = categories_dic.get('direction', 'ascending')
    group_num = categories_dic.get('group_num', 5)

    # 获取查询及所需因子名称
    q, factor_cols = cat_tuple

    sel_idx = categories_df.query(q).index
    test_factor = factor_df.loc[sel_idx, factor_cols]
    if len(factor_cols) > 1:
        test_factor['复合因子'] = test_factor.mean(axis=1)

    afr = analyze_factor_res(test_factor,
                             ind_name=ind_name,
                             direction=direction)
    afr.get_calc(pricing, group_num)
    # 计算ic
    ic_df = afr.calc_ic()

    # ic_info_table = ic_df.groupby(level='factor_name').apply(
    #     lambda x: get_information_table(x.dropna()).T)
    ic_info_table = ic_df.groupby(
        level='factor_name').apply(lambda x: get_information_table(x.dropna()))

    ic_info_table['mean_ret'] = afr.group_returns.groupby(
        level='factor_name').mean().stack()

    quantile_returns = afr.group_returns
    quantile_cum_returns = afr.group_cum_returns

    return factors_res(quantile_returns, quantile_cum_returns, ic_info_table)


# def get_factor_res2namedtuple(categories_df: pd.DataFrame,
#                               factors_df: pd.DataFrame, pricing: pd.DataFrame,
#                               cat_tuple: Tuple) -> namedtuple:
#     """计算每个象限的因子收益情况

#     Args:
#         categories_df (pd.DataFrame): MultiIndex level0-date level1-asset columns-分类情况
#         factors_df (pd.DataFrame): 因子分值
#         pricing (pd.DataFrame): 价格 index-date columns-codes
#         cat_tuple (Tuple): 0-分类筛选表达 1-因子组

#     Returns:
#         namedtuple
#     """
#     factors_res = namedtuple(
#         'factor_res',
#         'factor_data,quantile_returns,quantile_cum_returns,ic_info_table')
#     k = cat_tuple[0]
#     factor_cols = list(cat_tuple[1])
#     sel_idx = categories_df.query(k).index

#     test_factor = factors_df.loc[sel_idx, factor_cols]
#     test_factor['复合因子'] = test_factor.mean(axis=1)
#     factor_data = {}
#     quantile_returns = {}
#     quantile_cum_returns = {}
#     info_dic = {}

#     for name, ser in test_factor.items():

#         gfr = get_factor_returns(ser.dropna(), 0.25)
#         gfr.get_calc(pricing)

#         ic_info = get_quantile_ic(gfr.factors_frame)
#         cols = al.utils.get_forward_returns_columns(gfr.factors_frame.columns)
#         quantile_mean_ret = gfr.factors_frame.groupby(
#             'factor_quantile')[cols].mean()
#         ic_info['mean_ret'] = quantile_mean_ret
#         info_dic[name] = ic_info

#         factor_data[name] = gfr.factors_frame
#         quantile_returns[name] = gfr.group_returns
#         quantile_cum_returns[name] = gfr.group_cum_returns

#     quantile_info = pd.concat(info_dic)

#     return factors_res(factor_data, quantile_returns, quantile_cum_returns,
#                        quantile_info)

# def get_factor_res2namedtuple(categories_df: pd.DataFrame,
#                               factors_df: pd.DataFrame, pricing: pd.DataFrame,
#                               cat_tuple: Tuple) -> namedtuple:
#     """计算每个象限的因子收益情况

#     Args:
#         categories_df (pd.DataFrame): MultiIndex level0-date level1-asset columns-分类情况
#         factors_df (pd.DataFrame): 因子分值
#         pricing (pd.DataFrame): 价格 index-date columns-codes
#         cat_tuple (Tuple): 0-分类筛选表达 1-因子组

#     Returns:
#         namedtuple
#     """
#     factors_res = namedtuple(
#         'factor_res',
#         'factor_data,quantile_returns,quantile_cum_returns,ic_info_table')
#     k = cat_tuple[0]
#     factor_cols = list(cat_tuple[1])
#     sel_idx = categories_df.query(k).index

#     test_factor = factors_df.loc[sel_idx, factor_cols]
#     test_factor['复合因子'] = test_factor.mean(axis=1)
#     factor_data = {}
#     quantile_returns = {}
#     quantile_cum_returns = {}
#     info_dic = {}

#     for name, ser in test_factor.items():

#         gfr = analyze_factor_res(ser.dropna(), 0.25)
#         gfr.get_calc(pricing)

#         ic_info = gfr.calc_ic(gfr.factors_frame)
#         cols = get_factor_columns(gfr.factors_frame.columns)

#         quantile_mean_ret = gfr.factors_frame.groupby(
#             'factor_quantile')[cols].mean()
#         ic_info['mean_ret'] = quantile_mean_ret
#         info_dic[name] = ic_info

#         factor_data[name] = gfr.factors_frame
#         quantile_returns[name] = gfr.group_returns
#         quantile_cum_returns[name] = gfr.group_cum_returns

#     quantile_info = pd.concat(info_dic)

#     return factors_res(factor_data, quantile_returns, quantile_cum_returns,
#                        quantile_info)