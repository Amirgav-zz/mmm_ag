import pandas as pd
import numpy as np


def geoDecay(alpha, L):
    '''
    weighted average with geometric decay.
    weight_T = alpha ^ T-1 .
    Return weights of length L to calculate weighted averages with.

    alpha: scalar, between [0,1], retention rate
    L: scalar, maximum duration of carryover effect
    returns: numpy array of size (L,).
    '''
    return alpha ** (np.ones(L).cumsum() - 1)[::-1]


def delayed_adstock(alpha, theta, L):
    """
    weighted average with dealyed adstock function
    weight_T = alpha ^ (T-1- theta)^2
    Return weights to calculate weighted averages with.

    alpha: scalar, between [0,1], retention rate
    L: scalar, maximum duration of carryover effect
    theta:
    returns: numpy array, size (L,).
    """
    return (alpha ** ((np.ones(L).cumsum() - 1) - theta) ** 2)[::-1]


def carryover(df, columns, alpha, theta=None, L=None, decay='geo', date_col='date'):
    weights = geoDecay(alpha, L) if decay == 'geo' else delayed_adstock(alpha, theta, L)
    print(weights)
    df_carryover = (df.set_index(date_col)[columns]
                    .rolling(window=L, center=False, min_periods=1)
                    .apply(lambda x: np.sum(weights[-len(x):] * x) / np.sum(weights[-len(x):]), raw=False)
                    .reset_index())

    columns_dict = {col: col + '_wma_{}'.format(window) for col in columns}
    df_carryover.rename(columns=columns_dict, inplace=True)

    return df_carryover


if __name__ =='main':
    a = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7], 'b': [4, 5, 6, 21, 3, 4, 5], 'date': [7, 6, 5, 4, 3, 2, 1]})
    carryover(a, ['a', 'b'], alpha=0.5, L=3)