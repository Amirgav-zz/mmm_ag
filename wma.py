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
    return alpha ** (np.ones(int(L)).cumsum() - 1)[::-1]


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
    return (alpha ** ((np.ones(int(L)).cumsum() - 1) - theta) ** 2)[::-1]


def carryover(df, alpha, theta=None, L=None, decay='geo', date_col='date'):
    weights = geoDecay(alpha, L) if decay == 'geo' else delayed_adstock(alpha, theta, L)

    df_carryover = (df.set_index(date_col)
                    .rolling(window=L, center=False, min_periods=1)
                    .apply(lambda x: np.sum(weights[-len(x):] * x) / np.sum(weights[-len(x):]), raw=False)
                    .reset_index())

    #     columns_dict = {col:col+'_wma_{}'.format(window) for col in columns}
    #     df_carryover.rename(columns=columns_dict, inplace=True)

    return df_carryover


def beta_hill(x, S, K, beta):
    """
    x: pandas or numpy array
    S: scalar, slope
    K: scalar, half saturation
    beta: channel coefficient

    """
    return beta - (K ** S * beta) / (x ** S + K ** S)


def response_additive(df, channel_params, treatment_columns=None, control_columns=None,
                      date_col='date', tau=0, lamb=None, simulate=False, eps=0.05 ** 2):
    """
    channel_params: dictionary : dictionary of dictionaries, keys = treatment_columns,
                    values are dictionaries with the following setup;
                    {'alpha':0.5, 'theta'=None, 'L':None, 'decay':'geo','S':1, 'K':0.3, 'beta':0.5}
    tau: scalar, baseline sales
    lamb: list, effects of control variables
    """

    b_hill = pd.DataFrame()
    y = tau

    if treatment_columns:

        for treatment_col in treatment_columns:
            params = channel_params[treatment_col]

            carry_over = (
                carryover(df, params['alpha'], params['theta'],
                          params['L'], params['decay'], date_col)[treatment_col])

            b_hill[treatment_col] = beta_hill(carry_over, params['S'], params['K'], params['beta'])

        y += b_hill.sum(axis=1)

    if control_columns:
        y += df[control_columns].mul(np.asanyarray(lamb)).sum(axis=1)

    noise = None

    if simulate:
        noise = np.random.normal(0, eps, y.shape[0])
        y += noise

    return y, noise


if __name__ =='__main__':
    # a = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7], 'b': [4, 5, 6, 21, 3, 4, 5], 'date': [7, 6, 5, 4, 3, 2, 1]})
    # carryover(a, ['a', 'b'], alpha=0.5, L=3)

    df = pd.DataFrame()
    df['x'] = np.linspace(start=0, stop=1.5, num=100)
    df['y'] = np.linspace(start=0, stop=10, num=100)
    df['z'] = np.linspace(start=0, stop=30, num=100)
    df['w'] = np.linspace(start=0, stop=200, num=100)
    df['date'] = np.linspace(start=0, stop=1.5, num=100)
    treatment_columns = ['x', 'y']
    control_columns = ['z', 'w']
    alpha = 0.5
    L = 4
    decay = 'geo'
    date_col = 'date'
    theta = None
    S = 1
    K = 0.5
    beta = 0.3
    lamb = [0.5, 0.4]
    tau = 0
    simulate = True
    eps = 0.05 ** 2
    date_col = 'date'
    ch1_dict = {'alpha': 0.5, 'theta': None, 'L': 4, 'decay': 'geo', 'S': 1, 'K': 0.3, 'beta': 0.5}
    ch2_dict = {'alpha': 0.5, 'theta': None, 'L': 4, 'decay': 'geo', 'S': 1, 'K': 0.3, 'beta': 0.5}
    channel_params = {'x': ch1_dict, 'y': ch2_dict}

    y, noise = response_additive(df, channel_params=channel_params, treatment_columns=treatment_columns,
                                 control_columns=control_columns,date_col=date_col, tau=tau, lamb=lamb,
                                 simulate=simulate, eps=eps)
