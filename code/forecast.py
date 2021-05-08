
import pandas as pd
import numpy as np
from collections.abc import Sequence
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Lasso
from statsmodels.regression.quantile_regression import QuantReg

"""Function for getting hourly seperated indices."""
get_hourly_df = lambda df, column: df.groupby(df.index.hour)[column].apply(
        lambda col: col.to_dict()).to_frame().dropna()


def get_naive_forecast(price_df, forecast_dates):
    """Function to get naive forecast by utilizing historical data.
    Tuesday, Wednesday, Thursday and Friday values equal to the same hour's values
    of previous day. Monday, Saturday and Sunday values equal to last week same hour's values.
    Prediction distributions are generated using historical simulation.
    Notes: Price_df must have a datetimeindex with relevant timezone info.
    Args:
        price_df (pd.DataFrame): df including observed values to be used.
        forecast_dates (tuple): dates to forecast, it should have (start date,
            end date) structure. Dates can be str or datetime-like.
    Returns:
        (pd.DataFrame): including Naive point and distributional forecast.
    """
    # Naive Forecast
    price_df_naive_raw = price_df.assign(weekday = lambda df: df.index.day_of_week)
    price_df_naive = pd.concat([
        price_df_naive_raw.loc[lambda df: df['weekday'] < 4, 'Price_MWh'].shift(1, freq='D'),
        price_df_naive_raw.loc[lambda df: (df['weekday'] > 4) | (df['weekday'] == 0), 'Price_MWh'].shift(7, freq='D'),
    ]).sort_index().to_frame(name='Naive_forecast')

    # hourly divided dataframes
    hourly_naive_forecast = get_hourly_df(price_df_naive, 'Naive_forecast')
    hourly_realized = get_hourly_df(price_df, 'Price_MWh')

    PI_construction_dates = pd.date_range(
        forecast_dates[0],
        forecast_dates[1],
        freq='D',
        tz=price_df.index.tzinfo
    )

    PI_historical_df = pd.DataFrame()
    for forecast_day in PI_construction_dates:
        # dates to utilize
        first_day = forecast_day - pd.Timedelta(52*7, unit='D')
        end_day = forecast_day - pd.Timedelta(1, unit='H')
        for hour in range(24):
            # training values for PI
            train_df = hourly_naive_forecast.loc[hour].truncate(first_day, end_day)
            # index to naive forecast including time
            forecast_index = forecast_day + pd.Timedelta(hour, unit='hour')
            # naive forecast
            point_forecast = hourly_naive_forecast.loc[hour, forecast_index].values[0]
            # construct distribution and add point forecast
            PI_historical = train_df.merge(
                hourly_realized.loc[hour],
                left_index=True,
                right_index=True,
                how='left'
            ).assign(
                diff = lambda df: df['Naive_forecast'] - df['Price_MWh']
            )['diff'].add(point_forecast).to_numpy()
            # add distribution to general df
            PI_historical_df = pd.concat([
                PI_historical_df,
                pd.Series(
                    [PI_historical],
                    index=[forecast_index]
                )
            ])

    return price_df_naive.assign(PI_historical = PI_historical_df).dropna()


def add_price_lags(df, lags):
    """(Nested) Function to add price lags to a given pd.DataFrame. It can also
    differentiate and process dataframe with hourly seperated index.
    Args:
        df (pd.DataFrame): includes values with datetimeindex and with a column
            named "Price_MWh".
        lags (list-like): lags to include.
    Returns:
        df_copy (pd.DataFrame): df also including lags of Price_MWh.
    """
    df_copy = df.copy()
    if isinstance(df.index, pd.MultiIndex): # hourly divided df
        for num in lags:
            df_copy[f'Price_MWh_l{num}'] = df.groupby(level=0).Price_MWh.shift(num)
    else: # hourly consecutive df
        for num in lags:
            df_copy[f'Price_MWh_l{num}'] = df.Price_MWh.shift(num*24)
    return df_copy


def add_threshold_ind_col(df, price_df):
    """(Nested) Function to add threshold indicators for Threshold Autoregressive Model.
    If yesterday's mean is above last week yesterday (8 days ago)'s mean,
    it gets 1 as threshold indicators else 0.
    Args:
        df (pd.DataFrame): to add the indicator column.
        price_df (pd.DataFrame): observed values to use in calculations.
    Returns:
        df_w_threshold_ind (pd.DataFrame): df with threshold indicator column
            named "threshold_ind".
    """

    df_w_threshold_ind = df.assign(
        mean_8days_ago = get_hourly_df(
            price_df.resample('D').transform('mean').shift(8*24),
            "Price_MWh"
        ),
        mean_yesterday = get_hourly_df(
            price_df.resample('D').transform('mean').shift(24),
            "Price_MWh"
        ),
        threshold_ind = lambda df: (df.mean_8days_ago < df.mean_yesterday).astype(int),
    ) \
    .drop(columns=['mean_8days_ago', 'mean_yesterday'])

    return df_w_threshold_ind


def get_threshold_cols(df):
    """(Nested) Function to generate columns for Threshold Autoregressive Model.
    By using threshold_ind column in given df, it generates two sets of columns:
    columns with suffix 'state_0' have 0's in the rows with threshold_ind equals to 1,
    original values otherwise; and vice versa for columns with 'state_1' suffix.
    Args:
        df (pd.DataFrame): must include columns 'Price_MWh' and 'threshold_ind'.
    Returns:
        df_final (pd.DataFrame): dataframe to use in TAR model.
    """
    # State 0
    df0 = df.mask(df.threshold_ind==0, 0).drop(['threshold_ind', 'Price_MWh'],1)
    df0.rename(columns={x: f"{x}_state0" for x in df0.columns}, inplace=True)
    # State 1
    df1 = df.mask(df.threshold_ind==1, 0).drop(['threshold_ind', 'Price_MWh'],1)
    df1.rename(columns={x: f"{x}_state1" for x in df1.columns}, inplace=True)
    # Merged df
    df_final = pd.concat([df.Price_MWh,df0,df1],axis=1)

    return df_final


def get_mARX_cols(df):
    """Function to add interaction terms in multi-day ARX model.
    Args:
        df (pd.DataFrame): must have columns; Price_MWh_l1, Saturday, Sunday and Monday
    Returns:
        (pd.DataFrame): df with added columns e.g., Monday_Price_MWh_l1
    """
    for day in ['Saturday', 'Sunday', 'Monday']:
        df[f"{day}_Price_MWh_l1"] = df[day].multiply(df.Price_MWh_l1)

    return df.assign(
        Price_MWh_l1 = df.mask((df.Saturday==1) | (df.Sunday==1) | (df.Monday==1),0)
    )



def bootstrap_exogs(input_matrix, model_type, lags, prices_new, number_of_exogs, threshold_ind_list, loop_index):
    """(Nested) Function to get exogenous values to use bootstrap prediction interval
    generation in AR and TAR based models.
    Note: input_matrix first column should be endogenous variable (Price_MWh in this case).
    Args:
        input_matrix (np.ndarray): includes values to train the models.
        model_type (str): 'AR' for general Autoregressive models 'TAR' for
            Threshold Autoregressive models. This function adds zeros
            depending on the threshold value if model is "TAR".
        lags (list-like): lags included in the model.
        prices_new (np.array): prices calculated using estimated coefficients.
        number_of_exogs (int): number of exogenous variables in the model.
        threshold_ind_list (list-like): threshold indicator in the TAR model.
        loop_index (int): index of the loop in the nested function `bootstrap_new_prices`.
    Returns:
        (np.ndarray): including exogenous variables, price_lags and generated
            prices by bootstrapping.
    """
    exogs = np.concatenate(
        (input_matrix[loop_index, 1:number_of_exogs+1], prices_new[np.array(lags)*-1])
    )
    if model_type=='TAR': # if the model is TAR
        if threshold_ind_list[loop_index] == 0: # if below threshold
            return np.concatenate((exogs, np.zeros(len(exogs))))
        else: # if above threshold
            return np.concatenate((np.zeros(len(exogs)), exogs))
    elif model_type=='mAR':
        return np.concatenate(
            (exogs, input_matrix[loop_index,number_of_exogs+len(lags)+1:])
        )
    elif model_type=='AR': # if model is AR
        return exogs


def bootstrap_new_prices(input_matrix, beta_matrix, lags, res, model_type, number_of_exogs, threshold_ind_list):
    """(Nested) Function to get generated prices by bootstrapping
    Args:
        input_matrix (np.ndarray): includes values to train the models.
        beta_matrix (np.array): includes estimated coefficients of the model
        lags (list-like): lags included in the model.
        res (np.array): residuals of the model
        model_type (str): 'AR' for general Autoregressive models 'TAR' for
            Threshold Autoregressive models.
        number_of_exogs (int): number of exogenous variables in the model.
        threshold_ind_list (list-like): threshold indicator in the TAR model.
    Returns:
        (np.array): generated prices by bootstrapping
    """

    prices_new = input_matrix[:max(lags),0]
    for i in range(max(lags), len(input_matrix)):
        new_value = np.sum([
            beta_matrix[0], # constant
            np.multiply(
                bootstrap_exogs(
                input_matrix, model_type, lags, prices_new,
                number_of_exogs, threshold_ind_list, i
                ),# X
                beta_matrix[1:] # estimated coefficients
            ).sum(),
            np.random.choice(res, size=1, replace=True).sum() # eps
        ])
        prices_new = np.append(prices_new, new_value)
    return prices_new

def get_forecast_by_method(method, df, forecast_exogs, lasso_args):
    """Function to train a model and get a one-step ahead forecast.
    Currently two options are available; arima with yule_walker equations and
    lasso.
    Args:
        method (str): 'ARIMA' or 'LASSO'
        df (pd.DataFrame): df including endogenous and exogenous variables,
            endogenous variable should be called Price_MWh.
        forecast_exogs (pd.Series): exogenous variables to use in forecasting
        lasso_args (dict): args for sklearn.linear_model.Lasso
    Returns:
        model (statsmodels.tsa.arima.model.ARIMAResultsWrapper or
            sklearn.linear_model._coordinate_descent.Lasso): model of interest
        forecast (np.array): 1-D array with one-step ahead forecast
    """

    if method.upper() == 'ARIMA':
        model = ARIMA(
            endog=df.Price_MWh,
            exog=df.drop('Price_MWh',1),
        ).fit(method='yule_walker')

        forecast = model.forecast(exog=forecast_exogs).to_numpy()
        model.params = model.params[:-1] # exclude sigma^2 - GARCH

    elif method.upper() == 'LASSO':
        model = Lasso(**lasso_args).fit(
            y=df['Price_MWh'],
            X=df.drop('Price_MWh', 1)
        )
        if isinstance(forecast_exogs, pd.Series):
            forecast_exogs = forecast_exogs.values.reshape(1,-1)

        forecast = model.predict(X=forecast_exogs)
        model.resid = np.subtract(
            df.Price_MWh, model.predict(df.drop('Price_MWh', 1))
        )
        model.params = np.insert(model.coef_, 0, model.intercept_)
    else:
        raise NotImplementedError('Specified model not implemented.')

    return model, forecast


def get_PI_bootstrap(
    days, hour, B, model, sub_train_df, forecast_exogs, lags,
    model_type, number_of_exogs, threshold_ind_list,
    method, lasso_args
):
    """(Nested) Function to get distributional predictions using bootstrapping.
    Each hour is forecasted using a seperate model due to correlation.
    Args:
        days (tuple): dates to forecast, it should have (start date,
            end date) structure. Dates can be str or datetime-like.
            same as ´forecast_dates´ in `get_forecast_AR` function.
        hour (int): key to get the relevant subset.
        B (int): bootstrap times count.
        sub_train_df (pd.DataFrame): includes training data.
        forecast_exogs (pd.DataFrame): includes exogenous variables to use in forecasting.
        lags (list-like): lags included in the model.
        number_of_exogs (int): number of exogenous variables.
        threshold_ind_list (list-like): threshold indicator in the TAR model.
    Returns:
        (list): including distributional predictions generated by bootstrapping
    """
    input_matrix = sub_train_df.to_numpy()
    beta_matrix = np.array(model.params)
    res = model.resid.to_numpy()

    PI_bootstrap = []

    for i in range(B):
        new_prices = bootstrap_new_prices(
            input_matrix=input_matrix,
            beta_matrix=beta_matrix,
            lags=lags,
            res=res,
            model_type=model_type,
            number_of_exogs=number_of_exogs,
            threshold_ind_list=threshold_ind_list
        )
        if model_type=='TAR':
            # merge states together
            new_sub_train_df_raw = sub_train_df.rename(
                columns={c: c[:-7] for c in sub_train_df.columns if 'state' in c}
            ) \
            .groupby(level=0, axis=1).mean().multiply(2) \
            .assign(
                Price_MWh = new_prices,
                threshold_ind = threshold_ind_list.reset_index(drop=True)
            )
            new_sub_train_df = add_price_lags(new_sub_train_df_raw, lags).combine_first(new_sub_train_df_raw)
            new_sub_train_df = get_threshold_cols(new_sub_train_df)
        else:
            # change Price column with new prices
            new_sub_train_df_raw = sub_train_df.assign(Price_MWh = new_prices).copy()
            # add price lags and fillna
            new_sub_train_df = add_price_lags(new_sub_train_df_raw, lags).combine_first(sub_train_df)
            if model_type=='mAR':
                # add mARX columns
                new_sub_train_df = get_mARX_cols(new_sub_train_df)

        _, new_forecast = get_forecast_by_method(
            method, new_sub_train_df, forecast_exogs.loc[hour], lasso_args
        )

        PI_bootstrap.append(new_forecast[0])

    return [np.array(PI_bootstrap)]


def get_forecast_AR(
    main_df, price_df, lags, forecast_dates,
    model_type='AR', method='ARIMA',
    lasso_args={'fit_intercept':True, 'selection':'random'},
    PI_calculate=False, bootstrap_B=200
):
    """Function to get point and distributional forecasts using Autoregressive models.
    Currently only simple Autoregressive and Threshold Autoregressive models are
    included. Coefficients are estimated by using Yule-Walker equations.
    Generation of distributional forecasts significantly increases the runtime.
    Note: Forecasts are done assuming main_df includes 1st differenced series,
    since it is the most common case in electricity day-ahead prices.
    Args:
        main_df (pd.DataFrame): includes endogenous and exogenous variables.
        price_df (pd.DataFrame): observed data, i.e. Price/MWh.
        lags (list-like): lags to include in the model.
        forecast_dates (tuple): dates to forecast, it should have (start date,
            end date) structure. Dates can be str or datetime-like.
        model_type (str): 'AR' for general Autoregressive models 'TAR' for
            Threshold Autoregressive models. This function adds zeros
            depending on the threshold value if model is "TAR".
            default is 'AR'.
        method (str): 'ARIMA' for default Autoregressive models, 'Lasso' for
            regularized coefficients in the model. Default is Arima.
        lasso_args (dict): args for sklearn.linear_model.Lasso
        PI_calculate (bool): If True also includes distributional forecasts.
        bootstrap_B (int): bootstrap times count, default is 200.
    Returns:
        forecast_df (pd.DataFrame): includes forecasts for each hour between
            start and end date in forecast_dates.
    """

    PI_construction_dates = pd.date_range(
            forecast_dates[0],
            forecast_dates[1],
            freq='D',
            tz=main_df.index.get_level_values(1).tzinfo
        )
    forecast_df_index = pd.date_range(
            forecast_dates[0],
            f"{forecast_dates[1]} 23:00",
            freq='H',
            tz=main_df.index.get_level_values(1).tzinfo
        )
    PI_construction_dates.freq=None
    number_of_exogs = len(main_df.drop('Price_MWh', axis=1).columns)
    forecast_df = pd.DataFrame()
    for forecast_day in PI_construction_dates:
        # dates to utilize
        first_day = forecast_day - pd.Timedelta(52*7, unit='D')
        end_day = forecast_day - pd.Timedelta(1, unit='H')

        # add explanatory variables
        if model_type=='TAR':
            main_df_w_threshold_ind = add_threshold_ind_col(main_df, price_df)
            main_df_w_lags = add_price_lags(main_df_w_threshold_ind, lags)
            full_df_w_exogs = get_threshold_cols(main_df_w_lags)
            threshold_ind_df = main_df_w_lags.threshold_ind.copy()
        elif model_type=='AR':
            full_df_w_exogs = add_price_lags(main_df, lags)
            main_df_w_lags = None
        elif model_type=='mAR':
            main_df_w_lags = add_price_lags(main_df, lags)
            full_df_w_exogs = get_mARX_cols(main_df_w_lags)


        # get training dataframe for each hour
        train_df = full_df_w_exogs.groupby(level=0).apply(
            lambda x: x.droplevel(0).truncate(first_day, end_day)
        )
        # variables to use in forecasting
        forecast_exogs = full_df_w_exogs.groupby(level=0).apply(
            lambda x: x.droplevel(0).truncate(
                before=forecast_day,
                after=forecast_day+pd.Timedelta(23, unit='h')
            )
        ) \
        .drop(['Price_MWh'], axis=1) \
        .droplevel(1, axis=0)

        for hour in range(24):
            sub_train_df = train_df.loc[hour].reset_index(drop=True).copy()
            if model_type=='TAR':
                threshold_ind_list = threshold_ind_df.loc[hour].copy()
            else:
                threshold_ind_list = None

            model, forecast = get_forecast_by_method(
                method, sub_train_df, forecast_exogs.loc[hour], lasso_args
            )

            temp_forecast_df = pd.Series(forecast).to_frame('Forecast')

            if PI_calculate:
                temp_forecast_df = temp_forecast_df.assign(
                    Resid = [model.resid.to_numpy()],
                    Resid_variance = model.resid.std(),
                    Resid_normal = lambda df: [np.random.normal(
                        scale=df.Resid_variance,
                        size=len(model.resid)
                    )],
                    PI_bootstrap_raw = lambda df: get_PI_bootstrap(
                        hour=hour,
                        days=(first_day, end_day),
                        B=bootstrap_B,
                        model=model,
                        sub_train_df=sub_train_df,
                        lags=lags,
                        forecast_exogs=forecast_exogs,
                        model_type=model_type,
                        number_of_exogs=number_of_exogs,
                        threshold_ind_list=threshold_ind_list,
                        method=method,
                        lasso_args=lasso_args
                    )
                )
            forecast_df = pd.concat([forecast_df,temp_forecast_df])

    # Add yesterdays prices to balance the differencing
    # and construct prediction intervals
    forecast_df = forecast_df.set_index(forecast_df_index).assign(
        yesterday_price = lambda df: price_df.shift(24).loc[df.index],
        Forecast_added = lambda df: df.Forecast + df.yesterday_price,
    )
    if PI_calculate:
        forecast_df = forecast_df.assign(
            PI_historical = lambda df: df.apply(lambda x: x.Resid + x.Forecast_added, axis=1),
            PI_distributional = lambda df: df.apply(lambda x: x.Resid_normal + x.Forecast_added, axis=1),
            PI_bootstrap = lambda df: df.apply(lambda x: x.PI_bootstrap_raw + x.yesterday_price, axis=1),
        )

    return forecast_df.drop(['yesterday_price'], axis=1)


def get_forecast_QRA(qra_df, qra_range, forecast_dates):
    """Function to get Quantile Regression Averaging Forecasts from point
    forecasts of different models.
    Args:
        qra_df (pd.DataFrame): should include point forecasts of individual models
            as well as observed prices in column named as "Price_MWh".
        qra_range (array-like): quantiles to forecast.
        forecast_dates (tuple): dates to forecast, it should have (start date,
            end date) structure. Dates can be str or datetime-like.
    Returns:
        (pd.DataFrame): string columns from qra_range.
    """
    training_end = (pd.Timestamp(forecast_dates[0]) - pd.Timedelta(24, unit='h')).strftime('%x')

    train_qra_df = qra_df.loc[:training_end].copy()
    to_forecast_qra = qra_df.loc[forecast_dates[0]:forecast_dates[1]].drop('Price_MWh',1).copy()

    hourly_qra = pd.concat([get_hourly_df(train_qra_df, col) for col in train_qra_df.columns], axis=1)
    hourly_qra_forecast = pd.concat([get_hourly_df(to_forecast_qra, col) for col in to_forecast_qra.columns], axis=1)

    PI_construction_dates = pd.date_range(
                forecast_dates[0],
                forecast_dates[1],
                freq='D',
                tz=qra_df.index.tzinfo
            )
    qra_forecast_df = pd.DataFrame()
    for forecast_date in PI_construction_dates:
        for hour in range(24):
            qra = QuantReg(hourly_qra.loc[hour].Price_MWh, hourly_qra.loc[hour].drop('Price_MWh',1))
            forecast_var_df = hourly_qra_forecast.loc[hour].loc[forecast_date.strftime('%x')]
            if isinstance(forecast_var_df, pd.Series):
                forecast_var_df = forecast_var_df.to_frame().T

            qra_temp_forecast_df = pd.Series(qra_range).apply(
                lambda x: qra.fit(q=x, max_iter=99999).predict(forecast_var_df)
            ).T
            qra_temp_forecast_df.columns = qra_range.round(2).astype(str)
            qra_forecast_df = pd.concat([
                qra_forecast_df,
                qra_temp_forecast_df
            ])

    return qra_forecast_df


def PI_combinations(models, PI, beta):
    """Function to get various Prediction Interval(PI) combinations;
    ´mean´, ´median´, ´interior trimming´, ´exterior trimming´,
    ´Probability averaging of endpoints and simple averaging of midpoints´.
    Note: column names of each df in models should be strings.
    Args:
        models (list of pd.Dataframes): list of models to include in combinations
            each dataframe should have corresponding PI columns,
            e.g., if PI=0.9 df should have "0.05" and "0.95" as columns
        PI (list-like or float): list of prediction invervals to use
        beta (float): trimming percentage (only applied to interior and exterior
            trimming). Rounded to nearest upper level, e.g., if there are 10 models
            and beta=0.05 it is rounded to 0.9 and 9 models are used.
    Returns:
        (pd.Dataframe): dataframe with multilevel columns; (Method, level)
    """

    # merge whole models
    merged_df = pd.concat(models, axis=1)

    # drop midpoint if available - to avoid confusion
    if 0.5 in merged_df.columns.astype(float):
        merged_df.drop('0.5', 1, inplace=True)
    PI_mean_df = merged_df.groupby(level=-1, axis=1).mean()
    PI_median_df = merged_df.groupby(level=-1, axis=1).median()
    PI_min_df = merged_df.groupby(level=-1, axis=1).min()
    PI_max_df = merged_df.groupby(level=-1, axis=1).max()

    PI_trimming_lower = merged_df.groupby(level=-1, axis=1).apply(
        lambda model: model.apply(
            lambda x: np.mean(sorted(x)[:int(len(x)*(1-beta))])
            , axis=1
        )
    )
    PI_trimming_upper = merged_df.groupby(level=-1, axis=1).apply(
        lambda model: model.apply(
            lambda x: np.mean(sorted(x)[int(len(x)*(1-beta)):])
            , axis=1
        )
    )
    if not isinstance(PI, Sequence):
        PI = [PI]

    return_df = pd.concat([PI_mean_df, PI_median_df], axis=1, keys=['Mean', 'Median'])
    for pi in PI:
        bounds = np.array([(1-pi)/2, (1+pi)/2]).round(2).astype(str)

        # Envelope
        PI_envelope_df = pd.concat([PI_min_df[bounds[0]], PI_max_df[bounds[1]]], axis=1)

        # Trimming
        PI_InteriorTrimming_df = pd.concat([
            PI_trimming_lower[bounds[0]],
            PI_trimming_upper[bounds[1]]
            ],
            axis=1
        )
        PI_ExteriorTrimming_df = pd.concat([PI_trimming_upper[bounds[0]], PI_trimming_lower[bounds[1]]], axis=1)

        # Probability averaging of endpoints and simple averaging of midpoints
        PI_PM = pd.concat(models, axis=1)[bounds].mean(axis=1) \
            .apply(lambda x: stats.norm.ppf(q=bounds.astype(float), loc=x)) \
            .apply(pd.Series).rename(columns={0:bounds[0], 1:bounds[1]})
        # combine
        temp_df = pd.concat([
            PI_envelope_df,
            PI_InteriorTrimming_df,
            PI_ExteriorTrimming_df,
            PI_PM,
        ], axis=1, keys=['Envelope', 'IntTrim', 'ExtTrim', 'PM'])
        return_df = pd.concat([return_df, temp_df], axis=1)

    return return_df
