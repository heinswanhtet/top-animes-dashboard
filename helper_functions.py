'''
In this file, all necessary custom defined functions here.
'''

import pandas as pd
import numpy as np


def is_subset(value, check_list):
    value_list = value.split(',')
    return any(value in check_list for value in value_list)


def get_category_count_from_str(col):
    cat_df = col.str.split(',', expand=True).stack().str.strip()
    cat_df = pd.get_dummies(cat_df, prefix='', prefix_sep='')

    cat_counts = cat_df.sum(axis=0)

    return cat_counts
