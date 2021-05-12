
import pandas as pd
import numpy as np
from collections.abc import Sequence
from scipy import stats

"""Function for getting hourly seperated indices."""
get_hourly_df = lambda df, column: df.groupby(df.index.hour)[column].apply(
        lambda col: col.to_dict()).to_frame().dropna()


def get_PI_from_distribution(distribution_series, return_negative_values=True, percentiles=np.linspace(1, 99, 99)):
    """Functio to get quantiles from an array of values.
    Args:
        distribution_series (pd.Series): includes values of a prediction distribution.
        return_negative_values (bool): In some markets, negative values are
            not allowed, thus, prediction intervals with negative values are set to
            0 if True.
        percentiles (np.array): percentiles to get from the values.
    Returns:
        percentile_df (pd.DataFrame) including percentiles as columns with string names.
    """

    percentile_df = distribution_series.apply(lambda x: np.percentile(x, percentiles)).apply(pd.Series)
    percentile_df.columns = (percentiles / 100).round(2).astype(str)
    if not return_negative_values:
        percentile_df = percentile_df.apply(lambda col: col.apply(lambda x: 0 if x<0 else x))
    return percentile_df


def get_coverage_indicators(percentile_df, realized, PI):
    """(Nested) Function to get coverage indicators, i.e., 1 if realized value is in the
    prediction intervals, 0 otherwise.
    Args:
        percentile_df (pd.DataFrame): df including prediction quantiles.
        realized (pd.DataFrame or pd.Series): observed values i.e. Price/MWh.
        PI (float): Prediction interval.
    Returns:
        hourly_cov_ind_df (pd.DataFrame) hourly seperated df with coverage
            indicators for each timestamp in percentile_df.

    """
    bounds = (1-PI)/2, (1+PI)/2
    bounds = np.round(bounds, 2)
    lower, upper = str(bounds[0]), str(bounds[1])

    col_name = realized.name if isinstance(realized, pd.Series) else realized.columns[0]
    # coverage indicators dataframe
    cov_ind_df = percentile_df.loc[:,[lower, upper]].merge(
        realized,
        left_index=True,
        right_index=True,
        how='left'
    ) \
        .apply( # 1 if realized is between bounds else 0
            lambda x:
                ( (x[lower] < x[col_name]) & (x[col_name] < x[upper]) )*1,
            axis=1
        )
    # hourly coverage indicators dataframe
    hourly_cov_ind_df = get_hourly_df(cov_ind_df.to_frame('Cov_ind'), 'Cov_ind')

    return hourly_cov_ind_df

def calculate_UC_LR_score(series, PI):
    """(Nested) Function to calculate Unconditional Coverage Likelihood Ratio Score.
    Args:
        series (pd.Series): includes indicators whether the realized value is
            in the prediction intervals.
        PI (float): Prediction Interval.
    Returns:
        (float) Unconditional Coverage Likelihood Ratio.
    """
    hit_count = series.sum()
    miss_count = ((series == 0) * 1).sum()
    hit_pct = series.mean()

    nominator = miss_count*np.log(1-PI) + hit_count*np.log(PI)
    denominator = miss_count*np.log(1-hit_pct) + hit_count*np.log(hit_pct)

    return 2*(-nominator+denominator)

def calculate_Ind_LR_score(n_00, n_10, n_01, n_11):
    """(Nested) Function to calculate Independence Likelihood Ratio Score
    Args:
        n_00 (pd.Series): includes indicators whether the realized value is
            out of prediction intervals in both current and previous timeframe.
        n_10 (pd.Series): includes indicators whether the realized value is
            out of prediction intervals in current but not in previous timeframe.
        n_01 (pd.Series): includes indicators whether the realized value is
            in prediction intervals in current but not in previous timeframe.
        n_11 (pd.Series): includes indicators whether the realized value is
            in prediction intervals in current and previous timeframe.
    Returns:
        (float): Independence Likelihood Ratio.
    """
    total = n_00+n_10+n_01+n_11
    hit_pct = (n_01+n_11)/total
    p_01, p_11 = n_01 / (n_00+n_01), n_11 / (n_10+n_11)
    nominator = (n_00+n_01)*np.log(1-hit_pct) + (n_01+n_11)*np.log(hit_pct)
    denominator = n_00*np.log(1-p_01) + n_01*np.log(p_01) + n_10*np.log(1-p_11) + n_11*np.log(p_11)
    return 2*(-nominator+denominator)

def Indepence_LR_score(hourly_cov_ind_df, lag):
    """(Nested) Wrapper function for calculating Independence Likelihood Ratio Score
    Uses `calculate_Ind_LR_score` function.
        Args:
            hourly_cov_ind_df (pd.DataFrame): includes hourly seperated
                coverage indicators.
            lag (int): lag value to get previous timeframe (1 for a day before).
        Returns:
            (pd.DataFrame) including Independence Likelihood Ratio for each hour.
    """

    count_df = hourly_cov_ind_df.groupby(level=0).apply(
        lambda hour: hour.assign(
            lagged_values = lambda df: df.shift(lag)
        ).dropna().assign(
            transition = lambda df: df.lagged_values.astype(int).astype(str) + df.Cov_ind.astype(int).astype(str)
        ).groupby('transition').Cov_ind.count()
    )
    return count_df.apply(
        lambda x: calculate_Ind_LR_score(
            n_00=x['00'],
            n_10=x['10'],
            n_01=x['01'],
            n_11=x['11'],
        ),
        axis=1
    ).to_frame(f'Ind_LR_lag{lag}')


def Christoffersen_scores(percentile_df, realized, PI, lags):
    """Function to calculate Christoffersen scores (Unconditional Coverage LR,
    Indepence_LR_score and Conditional Coverage LR. Conditional Coverage is
    equal to sum of Unconditional coverage LR and Indepence LR. ,
    This function uses `calculate_UC_LR_score`, `CC_LR_score_df` and
    `get_coverage_indicators` functions.
    Args:
        percentile_df (pd.DataFrame): including prediction interval quantiles
            e.g., if PI=0.9 it should have "0.05" and "0.95" as columns.
        realized (pd.DataFrame or pd.Series): observed values i.e. Price/MWh.
        lags (list-like or int): lags to include for Conditional Coverage LR.
    Returns:
        (pd.DataFrame): includes hourly values with columns;
            Unconditional Coverage LR scores (UC_LR),
            Indepence LR scores for each lag (Ind_LR_lag#) and
            Conditional Coverage LR score for each lag (CC_LR_lag#).
    """
    # transform lags into list if not list-like
    if not isinstance(lags, Sequence):
        lags = [lags]
    # get hourly seperated coverage indicators
    hourly_cov_ind_df = get_coverage_indicators(
        percentile_df,
        realized,
        PI
    )
    # Unconditional Coverage Likelihood-Ration Scores
    UC_LR_score_df = hourly_cov_ind_df.groupby(level=0).apply(lambda hour: calculate_UC_LR_score(hour, PI))
    UC_LR_score_df.columns = ['UC_LR']
    # Independence Likelihood-Ration Scores
    Ind_LR_score_df = pd.concat(
        [Indepence_LR_score(hourly_cov_ind_df, lag) for lag in lags],
        axis=1
    )
    # Conditional Coverage Likelihood-Ration Scores
    # CC = UC + Ind
    CC_LR_score_df = Ind_LR_score_df.add(UC_LR_score_df.squeeze(), axis=0)
    CC_LR_score_df.rename(
        columns={x:f"CC_LR_lag{x[-1]}" for x in CC_LR_score_df.columns},
        inplace=True
    )
    # combine and return all
    return pd.concat([
        UC_LR_score_df,
        Ind_LR_score_df,
        CC_LR_score_df
    ], axis=1)


def calculate_pinball_loss(q, forecast, realized):
    """(Nested) Function to calculate Pinball loss for a given quantile
    Since percentile_df in  ´pinball_loss´ function has string column names,
    function uses q_str to get the values.
        Args:
            q (str or float): quantile
            forecast (pd.DataFrame): df including forecast with given quantile.
            realized (pd.DataFrame or pd.Series): realized values
                it should have max one column.
        Returns:
            (pd.DataFrame): df including Pinball loss for each common timestamp.
    """

    if isinstance(q, float):
        q_str = str(q)
    elif isinstance(q, str):
        q_str, q = q, float(q)

    if isinstance(realized, pd.DataFrame):
        real_ = realized.columns[0]
    elif isinstance(realized, pd.Series):
        real_ = realized.name

    score_df = pd.concat([forecast, realized], 1) \
        .dropna() \
        .apply(
            lambda x: (1-q)*(x[q_str]-x[real_])if x[real_]<x[q_str] else q*(x[real_]-x[q_str]),
            axis=1
        )
    return score_df


def pinball_loss(percentile_df, realized):
    """Function to get pinball loss for a given dataframe including percentiles
    Uses ´calculate_pinball_loss´ function.
    Args:
        percentile_df (pd.DataFrame): includes forecast percentiles,
            optimally it has 99 columns with string names [0.01 ... 0.99].
        realized (pd.DataFrame or pd.Series): realized values.
    Returns:
        (pd.DataFrame): hourly divided df with MultiIndex (hour, timestamp)
        including pinball loss for each timestamp in percentile_df and realized.
    """
    pinball_df = percentile_df.apply(
        lambda col: calculate_pinball_loss(col.name, col, realized)
    )
    return pd.concat([get_hourly_df(pinball_df, col) for col in pinball_df],1)


def calculate_winkler_score(real, low, up, alpha):
    """(Nested) Function to calculate Winkler Score.
    Args:
        real (float): realized value.
        low (float): lower bound of prediction intervals.
        up (float): upper bound of prediction intervals.
        alpha (float): significance level.
    Returns:
        (tuple): Prediction Interval Width, Penalty.
    """
    if real < low: # lower than lower bound
        return ( up-low, (2/alpha)*(low-real) )
    elif real > up: # higher than upper bound
        return ( (up-low), (2/alpha)*(real-up) )
    else: # within boundaries
        return ( (up-low), 0) # distance btw bounds


def winkler_score(percentile_df, realized, PI):
    """Function to get Winkler Score for a given prediction interval
    Winkler scores are calculated using ´calculate_winkler_score´ function.
    Args:
        percentile_df (pd.DataFrame): df including forecast percentiles
            optimally it should have a shape of (Number_of_hours, 99).
        realized (pd.DataFrame or pd.Series): realized values,
            it needs to have only one columns or be a Series.
        PI (float): Prediction Interval.
    Returns:
        (pd.DataFrame): hourly divided dataframe including Winkler Score,
        Prediction Interval Width and Penalty with a multiindex (hour, timestamp).
    """

    bounds = (1-PI)/2, (1+PI)/2
    bounds = np.round(bounds, 2)
    lower, upper = str(bounds[0]), str(bounds[1])

    # column name of realized df
    real_ = realized.name if isinstance(realized, pd.Series) else realized.columns[0]

    winkler_df = percentile_df.loc[:,[lower, upper]].merge(
            realized,
            left_index=True,
            right_index=True,
            how='left'
        ) \
        .apply(
            lambda x: calculate_winkler_score(x[real_], x[lower], x[upper], 1-PI),
            axis=1
        ) \
        .apply(pd.Series) \
        .rename(columns={0: 'PI_Width', 1:'Penalty'}) \
        .assign(Winkler_score = lambda df: df.PI_Width + df.Penalty)
    return pd.concat([get_hourly_df(winkler_df, col) for col in winkler_df],1)


def DM_test(model1_score, model2_score, alpha, two_sided=False):
    """Function to conduct Diebold-Mariano test.
    Args:
        model1_score (pd.DataFrame): df including scores of model1
            should have multiindex; level 0 as hours (0,23)
            level 1 as the timestamp.
        model2_score (pd.DataFrame): df including scores of model2
            same structure as model1_score.
        alpha (float): significance level.
        two_sided (bool): if True two sided test is conducted else one-sided.
    Returns:
        (pd.DataFrame): dataframe with 1 and 0's for each hour;
            1: model 1 is significantly better than model 2
            0: no significant difference
            -1: model 2 is significantly better than model 1 (if only
                two_sided=True).
    """
    # Get common indices
    merged_df = pd.concat([model1_score, model2_score], axis=1, keys=['m1', 'm2']).dropna()
    # one sided function
    dm_test_onesided = lambda x: 1 if x>stats.norm.ppf(1-alpha) else 0
    # two sided function
    def dm_test_twosided(x):
        if x>stats.norm.ppf(1-alpha/2):
            return -1
        elif x<stats.norm.ppf(alpha/2):
            return 1
        else:
            return 0
    # choose function to run
    function_to_run = dm_test_twosided if two_sided else dm_test_onesided
    return merged_df['m1'].sub(merged_df['m2']) \
        .groupby(level=0).apply(
            lambda hour: np.sqrt(len(hour)) * (np.mean(hour) / np.std(hour))
        ) \
        .apply(function_to_run).to_frame('DM-decision')
