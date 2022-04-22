'''
Author: Hugo
Date: 2022-04-18 17:03:51
LastEditTime: 2022-04-22 14:42:11
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


def get_data(method='ALL'):

    dic = {
        'dichotomy': _dump_dichotomy,
        'quadrant': _dump_quadrant,
        'factor': _dump_factor,
        'price': _dump_price
    }

    if method.upper() == 'ALL':
        _dump_dichotomy()
        _dump_quadrant()
        _dump_factor()
        _dump_price()

    else:

        dic[method]()


def _dump_dichotomy():

    print('数据获取(起始日:%s,结束日:%s' % (Begin, End))
    # 划分高低端象限
    print('开始划分高低端象限...')
    dichotomy_df = get_dichotomy(Begin, End)
    dichotomy_df.to_csv(r'Data\dichotomy.csv')
    print('高低端象限数据获取完毕!')


def _dump_quadrant():
    # 划分四象限
    print('开始划分四象限...')
    quandrant_df = get_quadrant(Begin, End)
    quandrant_df.to_csv(r'Data\quandrant_slop.csv')
    print('四象限数据获取完毕!')


def _dump_factor():

    if os.path.exists(r'Data/quandrand_df.csv'):

        quandrant_df = pd.read_csv(r'Data/quandrand_df.csv',
                                   index_col=[0],
                                   parse_dates=True)
        print('开始获取因子数据...')
        factors_df = get_factors(quandrant_df)
        factors_df.to_csv(r'Data\factors_frame.csv')
        print('因子数据获取完毕!')
    else:
        print('缺少依赖数据:quandrand_df.csv,请先行下载quandrand_df.csv!')


def _dump_price():

    if os.path.exists(r'Data/quandrand_df.csv'):

        quandrant_df = pd.read_csv(r'Data/quandrand_df.csv',
                                   index_col=[0],
                                   parse_dates=True)
        print('开始获取收盘价数据...')
        pricing = get_pricing(quandrant_df, Last_date)
        pricing.to_csv('Data\pricing.csv')
        print('收盘价数据获取完毕!')

    else:
        print('缺少依赖数据:quandrand_df.csv,请先行下载quandrand_df.csv!')


def load_data() -> List:

    files = [
        'dichotomy.csv', 'quandrant_slop.csv', 'factors_frame.csv',
        'pricing.csv'
    ]

    out_put = []

    for file in files:

        file_path = rf'Data/{file}'

        if os.path.exists(file_path):

            if file in ['dichotomy.csv', 'factors_frame']:
                df = pd.read_csv(file_path, index_col=[0, 1], parse_dates=True)
                df.index.names = ['date', 'asset']
                out_put.append(df)
            else:
                df = pd.read_csv(file_path, index_col=[0], parse_dates=True)
                df.index.names = ['date']

            print('%s文件读取完毕!' % file)

        else:

            print('%s文件不存在请下载!' % file)

    return out_put
