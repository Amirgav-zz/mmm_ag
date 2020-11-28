import pandas as pd
import numpy as np
import math
import theano
import timeit



def yearly_period(x, amplitude, vertical_shift, period, phase_shift):
    return amplitude * np.sin((2*math.pi / period) * (x-phase_shift)) + vertical_shift


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


def carryover_theano(x, alpha, theta=None, L=None, decay='geo'):
    """

    :param x: numpy series:
    :param alpha: theano.tensor.var.TensorVariable  (decay rate)
    :param theta: theano.tensor.var.TensorVariable (delay of decay)
    :param L:  scalar:  number of time periods for carryover effect
    :param decay: str: 'geo' or 'decay
    :return: numpy series of shape(x), with carryover effect
    """
    weights = geoDecay(alpha, L) if decay == 'geo' else delayed_adstock(alpha, theta, L)
    transformed_x = []

    for t in range(x.shape[0]):
        upper_index = t + 1
        current_window_x = x[:upper_index] if (upper_index - L) < 0 else x[upper_index - L:upper_index]
        t_in_window = len(current_window_x)
        carried_x = (current_window_x * weights[-t_in_window:]).sum()
        transformed_x.append(carried_x / weights[-t_in_window:].sum())

    return np.array(transformed_x)



def carryover(df, alpha, theta=None, L=None, decay='geo', date_col='date'):
    weights = geoDecay(alpha, L) if decay == 'geo' else delayed_adstock(alpha, theta, L)
    df_carryover = (df.set_index(date_col)
                    .rolling(window=L, center=False, min_periods=1)
                    .apply(lambda x: (weights[-len(x):] * x).sum() / (weights[-len(x):]).sum(), raw=False)
                    .reset_index())

    return df_carryover


def beta_hill(x, S, K, beta):
    """
    x: pandas or numpy array
    S: scalar, slope
    K: scalar, half saturation
    beta: channel coefficient

    """
    return beta - (K ** S * beta) / (x ** S + K ** S)

def response_additive_pd(df, channel_params, treatment_columns=None, control_columns=None,
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


def response_additive(df, channel_params, treatment_columns=None, control_columns=None, lamb=None):
    """
    channel_params: dictionary : dictionary of dictionaries, keys = treatment_columns,
                    values are dictionaries with the following setup;
                    {'alpha':0.5, 'theta'=None, 'L':None, 'decay':'geo','S':1, 'K':0.3, 'beta':0.5}
    tau: scalar, baseline sales
    lamb: list, effects of control variables
    """
    start_time = timeit.default_timer()

    data = []
    if treatment_columns:

        for treatment_col in treatment_columns:
            params = channel_params[treatment_col]

            carry_over = (carryover_theano(
                df[treatment_col].values, params['alpha'], params['theta'],params['L'], params['decay']))
            b_hill = [beta_hill(x, params['S'], params['K'], params['beta']) for x in carry_over]
            data.append(b_hill)

    if control_columns:
        for i, col in enumerate(control_columns):
            data.append([x*lamb[i] for x in df[col].values])

    y = np.array(data).sum(axis=0).tolist()

    stop_time = timeit.default_timer()
    print('Time: ', stop_time - start_time)

    return y



