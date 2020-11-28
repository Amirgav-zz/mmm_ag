import pandas as pd
import numpy as np
import random
import theano
from sklearn.preprocessing import MinMaxScaler
import scipy
import pymc3 as pm
from statsmodels.tsa import arima_process as arima

from src.utils import yearly_period, response_additive

import timeit


if __name__ =='__main__':

    start = 0
    end = 52 * 2
    duration = int(end - start)
    intervals = 1
    period = 52
    phase_shift = 0
    vertical_shift = 0
    amplitude = 1

    mvn_mue = [0, 0, 0]
    mvn_sig = 1.4
    mvn_cor = 0.00

    time = np.arange(start, end, intervals)

    media_channels = yearly_period(time, amplitude=amplitude,
                                   vertical_shift=vertical_shift, period=period, phase_shift=0)
    # create covariance matrix for media channels
    cov = np.zeros((3, 3)) + mvn_cor
    np.fill_diagonal(cov, mvn_sig)

    # add gaussian noise
    random.seed(0)

    noise_1, noise_2, noise_3 = np.random.multivariate_normal(mvn_mue, cov, int(duration / intervals)).T
    # noise_1 = np.random.normal(0,mvn_sig,int(duration/intervals))
    # noise_2 = np.random.normal(0,mvn_sig,int(duration/intervals))
    # noise_3 = np.random.normal(0,mvn_sig,int(duration/intervals))

    scaler_minmax = MinMaxScaler()

    # media channel 1
    media_ch1 = media_channels + noise_1
    media_ch1 = np.expand_dims(media_ch1, axis=1)
    media_ch1 = scaler_minmax.fit_transform(media_ch1)

    # media channel 2
    media_ch2 = media_channels + noise_2
    media_ch2 = np.expand_dims(media_ch2, axis=1)
    media_ch2 = scaler_minmax.fit_transform(media_ch2)

    # media channel 1
    media_ch3 = media_channels + noise_3
    media_ch3 = np.expand_dims(media_ch3, axis=1)
    media_ch3 = scaler_minmax.fit_transform(media_ch3)

    # ARIMA variable

    np.random.seed(12345)
    arparams = np.array([.7, .6])
    maparams = np.array([.1, .02])
    ar = np.r_[1, arparams]  # add zero-lag and negate
    ma = np.r_[1, maparams]  # add zero-lag
    price = arima.arma_generate_sample(ar, ma, duration)
    price = np.expand_dims(price, axis=1)

    scaler_minmax = MinMaxScaler(feature_range=(-0.4, 0.4))
    price = scaler_minmax.fit_transform(price)

    dates = np.expand_dims(time, axis=1)
    df = (pd.DataFrame(data=np.hstack((dates, media_ch1, media_ch2, media_ch3, price)),
                       columns=['date', 'ch1', 'ch2', 'ch3', 'price']))

    ch1_dict = {'alpha': 0.6, 'theta': 5, 'L': 13, 'decay': 'delayed', 'S': 1, 'K': 0.2, 'beta': 0.8}
    ch2_dict = {'alpha': 0.8, 'theta': 3, 'L': 13, 'decay': 'delayed', 'S': 2, 'K': 0.2, 'beta': 0.6}
    ch3_dict = {'alpha': 0.8, 'theta': 4, 'L': 13, 'decay': 'delayed', 'S': 2, 'K': 0.2, 'beta': 0.3}
    channel_params = {'ch1': ch1_dict, 'ch2': ch2_dict, 'ch3': ch3_dict}
    tau = 4
    lamb = [-0.5]
    var_eps = 0.05 ** 2

    treatment_columns = ['ch1', 'ch2', 'ch3']
    control_columns = ['price']
    date_col = 'date'

    simulate = True

    sales = response_additive(df=df, treatment_columns=treatment_columns,
                                     channel_params=channel_params,
                                     control_columns=control_columns,
                                     lamb=lamb)
    sales = np.array(sales) + tau
    if simulate:
        noise = np.random.normal(0, var_eps, sales.shape[0])
        sales += noise

    #### Bayesian simulation ###
    with pm.Model():
        # Priors

        alpha = pm.Beta('alpha', 3, 3, shape=3)
        theta = pm.Normal('theta', 0, 12, shape=3)
        k = pm.Beta('k', 2, 2, shape=3)
        s = pm.Gamma('s', 3, 1, shape=3)
        beta = pm.HalfNormal('beta', sigma=1, shape=3)
        tau = pm.HalfNormal('intercept', sigma=5)
        lamb = pm.Normal('lamb', 0, 1)
        var_eps = pm.InverseGamma('noise', 0.05, 5e-4)

        # params
        ch1_dict = {'alpha': alpha[0], 'theta': theta[0], 'L': 13, 'decay': 'delayed', 'S': s[0], 'K': k[0],
                    'beta': beta[0]}
        ch2_dict = {'alpha': alpha[1], 'theta': theta[1], 'L': 13, 'decay': 'delayed', 'S': s[1], 'K': k[1],
                    'beta': beta[1]}
        ch3_dict = {'alpha': alpha[2], 'theta': theta[2], 'L': 13, 'decay': 'delayed', 'S': s[2], 'K': k[2],
                    'beta': beta[2]}
        channel_params = {'ch1': ch1_dict, 'ch2': ch2_dict, 'ch3': ch3_dict}
        lamb_list = [lamb]

        start_time = timeit.default_timer()

        # model
        mu = response_additive(df=df, treatment_columns=treatment_columns,channel_params=channel_params,
                               control_columns=control_columns, lamb=lamb_list)
        mu += tau

        stop_time = timeit.default_timer()
        print('Time: ', stop_time - start_time)

        sales_hat = pm.Normal('sales_hat', mu=mu, noise=var_eps, observed=sales)

        trace = pm.fit(method='svgd')