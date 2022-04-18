'''
Author: Hugo
Date: 2022-04-18 17:03:51
LastEditTime: 2022-04-18 17:24:37
LastEditors: Please set LastEditors
Description: 
'''
import os
from typing import (List, Tuple, Dict, Callable, Union)

import pandas as pd
import numpy as np

from my_src import (get_dichotomy, get_quadrant, get_factors, get_pricing)

Begin = '2013-01-01'
End = '2022-02-28'
Last_date = '2022-03-31'


def get_data():

    print('数据获取(起始日:%s,结束日:%s' % (Begin, End))
    # 划分高低端象限
    print('开始划分高低端象限...')
    dichotomy_df = get_dichotomy(Begin, End)
    dichotomy_df.to_csv('Data\dichotomy.csv')
    print('高低端象限数据获取完毕!')

    # 划分四象限
    print('开始划分四象限...')
    quandrant_df = get_quadrant(Begin, End)
    quandrant_df.to_csv('Data\quandrant_slop.csv')
    print('四象限数据获取完毕!')

    #
    print('开始获取因子数据...')
    factors_df = get_factors(quandrant_df)
    factors_df.to_csv('Data\factors_frame.csv')
    print('因子数据获取完毕!')

    print('开始获取收盘价数据...')
    pricing = get_pricing(quandrant_df, Last_date)
    pricing.to_csv('Data\pricing.csv')
    print('收盘价数据获取完毕!')


def load_data() -> Tuple:

    if os.path.exists('Data\dichotomy.csv'):

        dichotomy_df = pd.read_csv('Data\dichotomy.csv',
                                   index_col=[0, 1],
                                   parse_dates=True)

    if os.path.exists('Data\quandrant_slop.csv'):

        quandrant_slop = pd.read_csv('Data\quandrant_slop.csv',
                                     index_col=[0],
                                     parse_dates=True)

    if os.path.exists('Data\factors_frame.csv'):

        factors_frame = pd.read_csv('Data\factors_frame.csv',
                                    index_col=[0, 1],
                                    parse_dates=True)

    if os.path.exists('Data\pricing.csv'):

        pricing = pd.read_csv('Data\pricing.csv',
                              index_col=[0],
                              parse_dates=True)

    return (dichotomy_df, quandrant_slop, factors_frame, pricing)
