# Import statement

# Data loading and management
import numpy as np
import pandas as pd
import os

# Utilities for data-driven methods
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats

# From bill2watt package
from bill2watt.common.common import *
from bill2watt.utils.normalizing import XRowNormalizer, YRowNormalizer
from bill2watt.utils.utils import eval_x, spread_y
from bill2watt.predictors.flat_predictor import FlatPredictor
from bill2watt.predictors.slp_predictor import SLPPredictor
from bill2watt.predictors.category_predictor import CategoryPredictor
from bill2watt.predictors.classification_predictor import ClassificationPredictor
from bill2watt.predictors.regression_predictor import RegressionPredictor

# Data visualization and fancy stuff
from tqdm import tqdm
from matplotlib import pyplot as plt
#%%
# Plot settings
#%%
# Load the training/testing dataset

data = pd.read_csv('dataset.csv', sep=';', index_col=[0, 1])

# Columns names
col_user = 'user'  # code of the end user (in index)
col_month = 'month'  # number of the month (in index)
col_type = 'type'  # type of end user
col_class = 'class'  # class of end user
col_category = 'category'  # category of end user
cols_nd = [f"nd_j{j}" for j in js]  # number of days of each day-type in the month
cols_x = [f"x_f{f}" for f in fs]  # values of the x (ToU energy bill)
cols_y = [f"y_i{i}" for i in range(ni)]  # values of the y (hourly typical load profile)

# Visualize the dataset
print(data.head(12))
#%%
# Tests settings
# Here, we specify the different tests to run, i.e., predictor that are going to be tested

# Categories predictor
# NOTE: see 'CategoriesPredictor' class' doc for further detail
category = col_class  # identify new SLPs by grouping end users in the training set by class

# Classification predictor
# NOTE: see 'ClassificationPredictor' class' doc for further detail
clustering = KMeans(n_clusters=11)  # identify centers of typical load profiles
classifier = DecisionTreeClassifier()  # idrntify decision rule to assign bills to cluster centers

# Classification predictor
# NOTE: see 'ClassificationPredictor' class' doc for further detail
regressor = KNeighborsRegressor(n_neighbors=9)

tests = dict(
    flat=dict(label='Flat', predictor=FlatPredictor),
    slp=dict(label='SLP', predictor=SLPPredictor),
    categories=dict(label='Categories', predictor=CategoryPredictor, predictor_kwargs=dict(category=col_class)),
    classification=dict(label='Classification', predictor=ClassificationPredictor,
                        predictor_kwargs=dict(clustering=clustering, classifier=classifier)),
    regression=dict(label='Regression', predictor=RegressionPredictor, predictor_kwargs=dict(regressor=regressor))
)
#%%
# Normalization
# Here, we normalize the X and Y data along the rows
# NOTE: normalization is performed row-wise, to remove 'magnitude' effects of the different samples (i.e., end users)

x_row_norm, y_row_norm = XRowNormalizer(), YRowNormalizer()

# NOTE: the original magnitude is kept in the attribute 'x_sum' of the normalizers, however, we use a new DataFrame
data_norm = data.copy()
data_norm[cols_x] = x_row_norm.fit_transform(data[cols_x].values)
data_norm[cols_y] = y_row_norm.fit_transform(data[cols_x].values, data[cols_y].values)

# Visualize normalized data
print(data_norm.head(12))
#%%
# Training/test datasets
# Here, we split X and Y and create batches for leave-one-group-out cross-validation

x_data, y_data = data_norm[cols_x], data_norm[cols_y]

# Store also 'nd' (number of days) data, needed for 'flat' approach
nd_data = data[cols_nd]

# Store also users categorization data, needed for 'categories' approach
categ_data = data[[col_type, col_class, col_category]]

# Create batches for leave-one-group-out (LOGO) cross-validation
# NOTE: each group is composed of the samples from a single end user (typically, 12)
for index in (y_data.index, nd_data.index, categ_data.index):
    assert (index == x_data.index).all(), \
        "Indices of 'x_data', 'y_data', 'nd_data' and 'categ_data' must match to avoid issues when splitting."
make_groups = lambda:\
    LeaveOneGroupOut().split(x_data, groups=data.index.get_level_values(col_user))

# Visualize
print(f"Data split into {len(list(make_groups()))} groups of training/test sets")
#%%

def scale_gse(x, nd, y_ref):
    """
    Function 'scale_gse'
    ____________
    DESCRIPTION
    The function evaluates hourly load profiles (y_scale) in each type of
    day (j) scaling given reference load profiles (y_ref) in order to respect
    the monthly energy consumption divided into tariff time-slots (x).
    ______
    NOTES
    The method evaluates one scaling factor for each tariff time-slot, which
    is equal to the actual monthly consumption in that tariff time-slot
    divided by the consumption associated with the reference load profile.
    The latter is then scaled separately for the time-steps in each time-slot.
    ____________
    PARAMETERS
    x : np.ndarray
        Monthly electricity consumption divided into tariff time-slots
        Array of shape (nf,) where 'nf' is the number of tariff time-slots.
    nd : np.ndarray
        Number of days of each day-type in the month
        Array of shape (nj,) where 'nj' is the number of day-types
        (according to ARERA's subdivision into day-types).
    y_ref : np.ndarray
        Array of shape (nj*nh) where 'nh' is the number of time-steps in each
        day, containing the reference profiles.
    _______
    RETURNS
    y_scal : np.ndarray
        Estimated hourly load profile in each day-type
        Array of shape (nj*nh) where 'nh' is the number of time-steps in each
        day.
    status : str
        Status of the solution.
        Can be : 'ok', 'unphysical', 'error'.
    _____
    INFO
    Author : G. Lorenti (gianmarco.lorenti@polito.it)
    Date : 29.11.2022 (last update: 29.11.2022)
    """
    # ------------------------------------
    # check consistency of data
    # division of 'x' into tariff time slots
    assert (size := x.size) == nf, \
        "'x' must have size {}, not {}.".format(nf, size)
    # division of 'n_days' into day-types
    assert (size := nd.size) == nj, \
        "'nd' must have size {}, not {}.".format(nj, size)
    # total number of time-steps in 'y_ref'
    assert (size := np.size(y_ref)) == nh*nj, \
        "'y_ref' must have size {}, not {}.".format(nh*nj, size)
    # ------------------------------------
    # scale reference profiles
    # evaluate the monthly consumption associated with the reference profile
    # divided into tariff time-slots
    x_ref = eval_x(y_ref, nd)
    # calculate scaling factors k (one for each tariff time-slot)
    k_scale = x / x_ref
    # evaluate load profiles by scaling the reference profiles
    y_scal = y_ref.copy()
    # time-steps belonging to each tariff time-slot are scaled separately
    for if_, f in enumerate(fs):
        y_scal[arera.flatten() == f] = \
            y_ref[arera.flatten() == f] * k_scale[if_]
    # ---------------------------------------
    # return
    return y_scal, 'optimal'

# Test various predictors
# Here, we re-built the whole dataset using the different predictors through a LOGO cross-validation

# Execute each test
for test in tests.values():
    print(f"Performing test: {test['label']}")

    # Get test details
    predictor = test['predictor']
    predictor_kwargs = test.get('predictor_kwargs', dict())

    # Initialize output
    y_test = y_data.copy()
    x_test = x_data.copy()

    # Easy-exit if test is not 'data-driven'
    if predictor is FlatPredictor:
        y_test.loc[:] = predictor().predict(x_data.values, nd_data.values)

    elif predictor is SLPPredictor:
        keys = list(categ_data.reset_index().set_index([col_type, col_month]).index)
        y_test.loc[:] = predictor().predict(x_data.values, keys)

    else:
        for group in tqdm(list(make_groups())):
            train_indices, test_indices = group
            x_train = x_data.loc[x_data.index[train_indices]].copy()
            y_train = y_data.loc[y_data.index[train_indices]].copy()
            x_test_ = x_data.loc[x_data.index[test_indices]].copy()

            if predictor is CategoryPredictor:
                categ_train = categ_data.loc[categ_data.index[train_indices]]
                categ_test = categ_data.loc[categ_data.index[test_indices]]
                x_train = categ_train.join(x_train).set_index(
                    list(categ_train.columns), append=True)
                y_train = categ_train.join(y_train).set_index(
                    list(categ_train.columns), append=True)

                y_test_ = predictor(x_data=x_train, y_data=y_train,
                                    **predictor_kwargs) \
                    .predict(x_test_, categ_test[predictor_kwargs['category']])

            else:
                y_test_ = predictor(x_data=x_train, y_data=y_train,
                                    **predictor_kwargs).predict(x_test_)

            y_test.loc[y_test.index[test_indices]] = y_test_


    # Re-evaluate bill
    x_test.loc[:] = np.array([eval_x(y_, nd_) for y_, nd_ in
                              zip(y_test.values, nd_data.values)])

    y_test.loc[:] = YRowNormalizer().fit_transform(x_test.values, y_test.values)
    x_test.loc[:] = XRowNormalizer().fit_transform(x_test.values)


    # x_test.loc[:] = x_row_norm.inverse_transform(x_test.values)
    # y_test.loc[:] = y_row_norm.inverse_transform(y_test.values)
    if predictor is not FlatPredictor:
        for i, y_test_ in y_test.iterrows():
            x_test_ = x_data.loc[i].values
            nd_ = nd_data.loc[i].values

            y_test_, _ = scale_gse(x_test_, nd_, y_test_.values)
            x_test_ = eval_x(y_test_, nd_)

            y_test.loc[i] = y_test_
            x_test.loc[i] = x_test_

    test['y_data'] = y_test
    test['x_data'] = x_test


#%%

# Performance evaluation
# Here, we evaluate the performances of the various predictors according to different error metrics




def absolute_error(y_true, y_pred):
    return np.abs(y_true - y_pred).sum()

def energy_distance(y_true, y_pred, nd):
    weights = np.repeat(nd, nh)
    return stats.energy_distance(y_true, y_pred, u_weights=weights, v_weights=weights)

def duration_curve_error(y_true, y_pred, nd):
    dc_true = np.sort(spread_y(y_true, nd))
    dc_pred = np.sort(spread_y(y_pred, nd))
    return absolute_error(dc_true, dc_pred)


# def percentile_error(y_true, y_pred, percentile):
#     return np.percentile(y_true, percentile) - np.percentile(y_pred, percentile)

def f_absolute_error(row, y_true=y_data):
    return absolute_error(y_true.loc[row.name].values, row.values)

def f_energy_distance(row, y_true=y_data, nd=nd_data):
    return energy_distance(y_true.loc[row.name].values, row.values,
                           nd.loc[row.name].values)

def f_dc_error(row, y_true=y_data, nd=nd_data):
    return duration_curve_error(y_true.loc[row.name].values, row.values,
                                nd.loc[row.name].values)

def add_metric(metrics, new_metric, name):
    return metrics.join(new_metric.to_frame(name))

def f_y_diff(y):
    return pd.concat((y, y.iloc[:,:1]), axis=1).T.apply(np.diff, axis=0).T


# Calculate performance for each predictor
print("Evaluating tests")
for test in tests.values():


    x_pred = test['x_data']
    y_pred = test['y_data']

    assert (x_pred.index == x_data.index).all(), ""
    assert (y_pred.index == y_data.index).all(), ""


    # Absolute error on the X values
    x_error = x_pred.apply(f_absolute_error, axis=1, y_true=x_data)

    # Absolute error on the Y values (weighted on 'nd')
    y_error = {col: y_pred.iloc[:, j*nh:(j+1)*nh].apply(
        f_absolute_error, axis=1, args=(y_data.iloc[:, j*nh:(j+1)*nh],))
        for j, col in enumerate(cols_nd)}
    y_error = (pd.DataFrame(y_error) * nd_data).sum(axis=1)

    # Absolute error on the duration curve (syntethic, using 'nd')
    dc_error = y_pred.apply(f_dc_error, axis=1)

    # Energy Distance on the distribution of the predicted y (weighted on 'nd')
    e_distance = y_pred.apply(f_energy_distance, axis=1)

    #
    e_distance_diff = f_y_diff(y_pred).apply(f_energy_distance, axis=1,
                                             args=(f_y_diff(y_data),))

    test['metrics'] = pd.DataFrame({'x_error': x_error,
                                    'y_error': y_error,
                                    'dc_error': dc_error,
                                    'e_distance': e_distance,
                                    'e_distance_diff': e_distance_diff,
                                    })


#%%

from plotting import create_radar_plot, create_row_boxplots

metrics = dict(x_error=dict(ylabel='Absolute Error (%)',
                            multiplier=50,
                            title='Error on ToU bill values'),
               y_error=dict(ylabel='Absolute Error (%)',
                            multiplier=50,
                            title='Error on hourly load values'),
               dc_error=dict(ylabel='Absolute Error (%)',
                            multiplier=50,
                             title='Error on synthetic duration curves'),
               e_distance=dict(ylabel='Energy distance (-)',
                               title='Error on hourly load distributions'),
               e_distance_diff=dict(ylabel='Energy distance, diff (-)',
                               title='Error on hourly load change rate distributions'),
)

# Create a collection of dataframes showing the tests along the rows and, for
# each test, the error in all samples along the columns
metrics_samples = dict()

# Then, evaluate median and IQR of the errors (for each metric separately) for
# each test divided by 'class' of end user
metrics_median_class = dict()
metrics_iqr_class = dict()

# Go through all the metrics considered
for key, metric in metrics.items():
    assert all(key in test['metrics'].columns for test in tests.values()), \
        f"All tests have the key {key} in their 'metrics' DataFrame."

    # Select the error metric for each test
    dfs = [test['metrics'][key].to_frame(name) for name, test in tests.items()]

    df = pd.concat(dfs, axis=1)
    class_groups = df.join(categ_data['class']).groupby('class')
    df_median_class = class_groups.median()
    df_iqr_class = class_groups.quantile(0.75) - class_groups.quantile(0.25)

    metrics_samples[key] = df.T
    metrics_median_class[key] = df_median_class.T
    metrics_iqr_class[key] = df_iqr_class.T



#%%

for key, metric in metrics.items():

    assert key in metrics_samples, ""

    multiplier = metric.pop('multiplier', 1)
    whis = (0.05, 1)

    df = metrics_samples[key] * multiplier
    df_flat = df.loc[df.index == 'flat', :]
    df = df.loc[df.index != 'flat', :]

    fig = create_row_boxplots(df, whis=whis, showmeans=True, **metric)
    ax = fig.get_axes()[0]

    # Plot text of the median values
    width = 0.3
    decimal_points = 1
    positions = [p + width for p in ax.get_xticks()]
    values = df.median(axis=1).to_list()
    for p, v in zip(positions, values):
        v = np.median(v)
        text = "{:.{}f}".format(v, decimal_points)
        ax.text(p, v, text, va='center', ha='left', )

    # Plot IQR and median of 'flat' test
    ax.axhspan(np.quantile(df_flat.values, 0.25), np.quantile(df_flat.values, 0.75),
             color='yellow', alpha=0.2)
    ax.axhline(np.quantile(df_flat.values, 0.5), color='tab:red')


    plt.show()


for key, metric in metrics.items():

    assert key in metrics_samples, ""

    multiplier = metric.pop('multiplier', 1)

    df = metrics_median_class[key] * multiplier


    fig = create_radar_plot(df, **metric)


    plt.show()









