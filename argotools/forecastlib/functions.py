import pandas as pd
import numpy as np
import scipy.stats as stats
import copy
import sys
import os
from argotools.config import *
import time

''' Auxiliary functions '''

def preds2matrix(preds_dict):
    # Receives preds in Predictor object format.
    pred_arrays = []
    for model, preds in preds_dict.items():
        pred_arrays.append(np.vstack(preds))

    return np.hstack(pred_arrays)

def convert_index(dataFrame, index_type):
    'Function that transforms a PDs dataframe index before data proccessing'

    if index_type == 'date':
        dataFrame.index = pd.to_datetime(dataFrame.index)

    return dataFrame


def np2dataframe(index, column_titles, data):
    '''
    Generates a dataframe based off Predictor object's data.
    Input:
        index (list or  pandas timeseries): the index for each of the data samples
        (N of index labels == N of rows in data)
        column_titles (str or list of str) : Same N than N of columns in data
    data: data to be converted.
    '''
    s = np.shape(data)

    if isinstance(column_titles,str):
        column_titles = [column_titles]*np.shape(data)[1]

    for i, title in enumerate(column_titles):
        if i > 0:
            column_titles[i] += '_{0}'.format(i)
    return pd.DataFrame(data=data, index=index, columns=column_titles)

def gen_folder(folder_name, c = 0):

    if c == 0:
        if os.path.exists(folder_name):
            new_folder_name = gen_folder(folder_name, c+1)
            return new_folder_name
        else:
            os.makedirs(folder_name)
            return folder_name
    else:
        if os.path.exists(folder_name+'_{0}'.format(c)):
             new_folder_name = gen_folder(folder_name, c+1)
             return new_folder_name
        else:
            os.makedirs(folder_name+'_{0}'.format(c))
            return folder_name+'_{0}'.format(c)

'''
Loading functions for data class

A loading function must satisfy the following conditions:

    Input:
        fname : path to the file to load.
        index_type : 'date' / 'other'.
        index_label :
        start_period :
        end_period :
    Output:
        target = The gold standard for the experiment
        features = The input to the prediction algorithms
        benchmarks = if found, return other models to compare with during
                    analysis phase. If not found in the ordered file, they
                    can be loaded using the load_benchmark() function.


Standard structuring of ordered data is as follows:
- CSV file
- First column : index (e.g. dates for the timeseries in disease forecasting)
- Second column : gold standard (e.g. Flunet series, CDC health reports, etc)
- Next n columns : Features

If the ordered data file contains benchmarks or it doesnt follow the previously
mentioned structure,
 '''

def read_standard_features(path_to_file, start_period, end_period):

    try:
        dataFrame = pd.read_csv(path_to_file, index_col=0)
    except Exception as t :
        print('Error in reading file with standard format. Error reads : {0}'.format(t))

    dataFrame = dataFrame[start_period : end_period]

    features =  copy.deepcopy(dataFrame)

    return features

def load_standard(fname, index_type, index_label, start_period, end_period):

    try:
        dataFrame = pd.read_csv(fname, index_col=0)
        dataFrame = convert_index(dataFrame, index_type)
    except IOError:
        print('Error in Metadata dictionary. Please check values')

    dataFrame = dataFrame[start_period : end_period]

    target = copy.deepcopy(dataFrame.ix[:,0])

    dataFrame.drop(dataFrame.columns[0], axis=1, inplace=True)

    features =  copy.deepcopy(dataFrame)

    return target, features, None

def load_latam(fname, index_type, index_label, start_period, end_period):

    try:
        dataFrame = pd.read_csv(fname, index_col=0)
    except IOError:
        print('Error in Metadata dictionary. Please check values')

    if index_type == 'date':
        dataFrame.index = pd.to_datetime(dataFrame.index)

    dataFrame = dataFrame[start_period : end_period]

    target = copy.deepcopy(dataFrame['ILI'])
    benchmarks = dataFrame['GFT']
    del dataFrame['ILI']
    del dataFrame['GFT']
    features =  copy.deepcopy(dataFrame)

    return target, features, benchmarks

def load_gc_features(path_to_file, start_period, end_period):
    try:
        df = pd.read_csv(path_to_file, skiprows=10, index_col=0)
        df.drop(df.columns[1], axis=1)
        df = df[start_period:end_period]
        return copy.deepcopy(df)
    except IOError:
        print('Was not unable to read GC file : {0}'.format(path_to_file))


'''
    Training window functions for Experiment Class.
    A training window function must satisfy the following conditions:

        Input:
            window_type :  Str
            current_window_size : Int
            initial_index:  Int
            curent_index :  Int
            features: Pandas dataframe
            target: Pandas dataFrame
        Output:
            window_size = The new window size.
            window_indices = A python list containing integers.

    Idea behind this standard is to let the user define their own time window structures
    Returning a list of indices gives flexibility for omitting inbetween indices whereas
    returning the full window may intervene with the 'horizon' parameter in Experiment class.

'''

def static_window(current_window_size,initial_index, \
                  current_index, features, target):
    return current_window_size, [i for i in range(current_index-current_window_size,\
                                       current_index)]

def expanding_window(current_window_size,initial_index, \
                     current_index, features, target):
    return current_index - initial_index, [i for i in range(initial_index,\
                                       current_index)]

'''
    Predictor preproccessing functions

'''
def no_preproc(X_train, Y_train, X_pred):
    return X_train, Y_train, X_pred

def preproc_rmv_values(X_train, Y_train, X_pred,ignore_values=None, verbose=False, remove_feature_nans=True):
    '''
        Removes elements in both series based on series2.

        e.g. If series2 has a nan at index 2, removes element at position 2 from
        both series1 and series2.

        timeseries_rmv_values removes NAN values by default and also any value
        assigned to ignore_values as a list.
    '''

    if ignore_values is None:
        ignore_values = [NAN_TO_VALUE]
    df = pd.DataFrame(np.hstack([Y_train.reshape([Y_train.shape[0],1]), X_train]))

    n_values = np.shape(df)[0]
    df = df[ df[0].notnull() ]
    rem_nans = n_values - np.shape(df)[0]

    for i, val in enumerate(ignore_values):
        df = df[df[0] != val]

    rem_vals =  n_values  - np.shape(df)[0] - rem_nans

    n_values_final = np.shape(df)[0]

    if verbose:
        print('Removed {0} NANs and {1} specified values from list :{2} \n'.format(rem_nans, rem_vals,ignore_values))
        print('Number of values before = {0}'.format(n_values))
        print('Number of values after =  {0}'.format(n_values_final))

    if remove_feature_nans:
        # check for nans inside the X_train_shape
        for name in df:
            df = df[ df[name].notnull() ]
            for v in ignore_values:
                df = df[np.invert(np.equal(df, v))]
        if verbose:
            print('{0} values removed from checking the features'.format(n_values_final-df.shape[0]))
            print(df)



    return df.iloc[:,1:X_train.shape[1]+1], df[0].values, X_pred


'''
    Pre-processing functions prior to entering the iteration loop.
    most processing can be found in : https://en.wikipedia.org/wiki/Feature_scaling
'''

def handle_nans(V):
    V[np.isnan(V)] = 0
    return V

def unnormalized(X_train, X_predict, Y_train):
    # Casting data into np arrays prior to prediction
    return np.array(X_train), np.array(X_predict), np.array(Y_train)


def zscore(X_train, X_predict, Y_train):
    # Basic ML standardization using z-score

    X = np.vstack([X_train, X_predict])
    Y = np.array(Y_train)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std[std==0] = 1
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    #Y -= np.mean(Y)
    #Y /= np.std(Y)

    X = handle_nans(X)
    #Y = handle_nans(Y)

    return X[:-1,:], X[-1,:], Y

def rescaling(X_train, X_predict, Y_train):
    X = np.vstack([X_train, X_predict])
    Y = np.array(Y_train)


    x_max = np.max(X, axis=0)
    x_min = np.min(X, axis=0)
    X -= x_min
    X /= x_max - x_min

    #y_max = np.max(Y)
    #y_min = np.min(Y)

    #Y -= y_max
    #Y /= y_max-y_min

    return X[:-1,:], X[-1,:], Y

def mean_normalization(X_train, X_predict, Y_train):
    X = np.vstack([X_train, X_predict])
    Y = np.array(Y_train)


    x_max = np.max(X, axis=0 )
    x_min = np.min(X, axis=0)
    x_mean = np.mean(X, axis=0)
    X -= x_mean
    X /= x_max - x_min

    #y_max = np.max(Y)
    #y_min = np.min(Y)
    #y_mean = np.mean(Y)

    #Y -= y_mean
    #Y /= y_max-y_min

    return X[:-1,:], X[-1,:], Y


def unit_length_normalization(X_train, X_predict, Y_train):
    X = np.vstack([X_train, X_predict])
    Y = np.array(Y_train)


    X /= np.linalg.norm(X, axis=0)
    #Y /= np.linalg.norm(Y, axis=0)

    return X[:-1,:], X[-1,:], Y

'''
    Ensemble preprocessing functions. Essentially the same functionality than
    regular method preprocessing functions, but additional operations are done onto
    the data used for ensemble models
'''

def stack_no_preproc(X_train, Y_train, X_pred, Y_ensemble, Y_index, predictor_output):
    return X_train, Y_train, X_pred, Y_ensemble, Y_index, predictor_output

'''
Metrics
'''


def metric_RMSE(timeseries, target):
    timeseries = np.array(timeseries)
    target = np.array(target)

    rmse = np.sqrt(np.mean(np.power(timeseries-target,2)))

    return rmse

def metric_NRMSE(timeseries, target):
    "RMSE NORMALIZED USING THE TIMESERIES VECTOR NORM"
    timeseries = np.array(timeseries)
    target = np.array(target)
    rmse = np.sqrt(np.mean(np.power(timeseries-target,2)))/np.linalg.norm(target)
    return rmse


def metric_pearson(timeseries, target):


    s = stats.pearsonr(timeseries, target)

    return s[0]

def metric_MSE(timeseries, target):
    timeseries = np.array(timeseries)
    target = np.array(target)

    mse = np.mean(np.power(timeseries-target,2))

    return mse


''' Voting Selector class functions'''

def get_timeseries_historic_winner(df, k, models, target_name):

    """
    Returns a timeseries built by the estimations of a list of models.
    Loops form the first to the last index starting at the k+1 row of the Dataframe and selects
    and for that row selects the model in "models" which had the least regular error in the past 3 days

    """


    n_samples = df.shape[0]
    preds = []
    for i in range(k):
        preds.append(df.iloc[k][models[0]])


    for i in range(k, n_samples):

        winner_name = error_selector(df.iloc[i-k:i], 0, k, models, target_name)
        preds.append(df.iloc[i][winner_name])

    return pd.Series(preds, index=df.index.values)









def error_selector(df, window, k, models, target_name):

    model_counter = dict(zip(models, [0 for mod in models]))

    for j in range(k):

        sub_df = df.iloc[-1-j]
        target = float(sub_df[target_name])
        round_min = float('inf')
        round_winner = ''
        for model in models:
            series = float(sub_df[model])
            mod_val = np.abs(target-series)
            if mod_val < round_min:
                round_min = mod_val
                round_winner = model

        model_counter[round_winner]+=1


    winner = []
    max_counts = 0
    for mod, counts in model_counter.items():
        if max_counts == counts:
            winner.append(mod)
        elif counts > max_counts:
            winner = [mod]
            max_counts = counts
    if len(winner)> 1:
        #tie breaker choose any randomly because virtually they're just as good
        return winner[np.random.permutation(len(winner))[0]]
    else:
        return winner[0]

def rmse_selector(df, window, k, models, target_name):

    model_counter = dict(zip(models, [0 for mod in models]))

    for j in range(k):

        sub_df = df.iloc[j:j+window]
        target = sub_df[target_name].values

        round_min = float('inf')
        round_winner = ''

        for model in models:
            series = sub_df[model].values
            mod_val = metric_handler['RMSE'](series, target)
            if mod_val < round_min:
                round_min = mod_val
                round_winner = model

        model_counter[round_winner] +=1


    winner = []
    max_counts = 0
    for mod, counts in model_counter.items():
        if max_counts == counts:
            winner.append(mod)
        elif counts > max_counts:
            winner = [mod]
            max_counts = counts

    if len(winner)> 1:
        #tie breaker choose any randomly because virtually they're just as good
        return winner[np.random.permutation(len(winner))[0]]
    else:
        return winner[0]


def get_winners_from_df(df, target_name='ILI', models = []):

    diff_df = copy.copy(df)
    for name in models:
        diff_df[name] = (diff_df[name].values - diff_df[target_name].abs())
    df['winners'] = diff_df[models].idxmin(axis=1)

    return df['winners'].to_frame()

'''
Data cleaning functions
'''

def timeseries_rmv_values(series1, series2, ignore_values, verbose=False):
    '''
        Removes elements in both series based on series2.

        e.g. If series2 has a nan at index 2, removes element at position 2 from
        both series1 and series2.

        timeseries_rmv_values removes NAN values by default and also any value
        assigned to ignore_values as a list.

    '''

    df = pd.DataFrame({'s1':series1, 's2':series2})

    n_values = np.shape(df)[0]
    df = df[ df['s2'].notnull() ]
    rem_nans = n_values - np.shape(df)[0]

    for i, val in enumerate(ignore_values):
        df = df[df['s2'] != val]

    rem_vals =  n_values  - np.shape(df)[0] - rem_nans

    n_values_final = np.shape(df)[0]

    if verbose == True:
        print('Removed {0} NANs and {1} specified values from list :{2} \n'.format(rem_nans, rem_vals,ignore))
        print('Number of values before = {0}'.format(n_values))
        print('Number of values after =  {0}'.format(n_values_final))

    return df['s1'].values, df['s2'].values

def rmv_values(df, ignore = None, verbose = False):

    if ignore is None:
        ignore = [NAN_TO_VALUE]
    n_values = np.shape(df)[0]
    df = df[ df[TARGET].notnull() ]
    rem_nans = n_values - np.shape(df)[0]

    for i, val in enumerate(ignore):
        df = df[df[TARGET] != val]

    rem_vals =  n_values  - np.shape(df)[0] - rem_nans

    n_values_final = np.shape(df)[0]

    if verbose == True:
        print('Removed {0} NANs and {1} specified values from list :{2} \n'.format(rem_nans, rem_vals,ignore))
        print('Number of values before = {0}'.format(n_values))
        print('Number of values after =  {0}'.format(n_values_final))

    return df

def filter_ar_columns(df, ignore=None, default_lags=['AR1'], verbose=False):

    count_cols = 0
    if ignore is None:
        ignore = [NAN_TO_VALUE]

    for col in df:
        if (np.isnan(df[col]).values.sum() > 0 or np.sum([np.equal(df[col],x).values.sum() for x in ignore]) > 0 ) and col not in default_lags:
            df[col] = df[col].apply(lambda x: 0)
            count_cols += 1
    if verbose:
        print('A total of {0} out of {1} were erased'.format(count_cols, len(list(df))))
        print('\n \n', new_list)
        time.sleep(2)
    return df

def filter_by_similarity(df,target, func=None, threshold=None, column_limit=None):
    """
    Function that performs feature selection based on a similarity metricself.
    Parameters
    __________
    df : Object
        Pandas dataframe containing the information column-wise.
    target : Object
        Numpy array containing only the information from our target column wise.
    func : function, optional (default is None)
        Function that returns the similarity score. If None is set, then it uses Pearson correlation
    threshold : float, optional
        Threshold value for which to accept or reject a column based on the similarity metric
        If set to None, then it accepts all columns
    column_limit : integer, optional (default is None)
        The maximum number of columns to return. If set to None, then returns all the columns
    """

    if func is None:
        func = stats.pearsonr
        f_name = 'pearson'
    else:
        f_name = 'custom'

    metric_values = []
    col_names = list(df)

    for col in list(df):
        val =  func(df[col].values.ravel(), target)
        if f_name == 'pearson':
            val = np.abs(val[0])

        metric_values.append(val)

    ordered_names = []
    ordered_values = []
    counter = 0
    for value, name in sorted(zip(metric_values, col_names), key = lambda x : x[0], reverse=True):
        ordered_names.append(name)
        ordered_values.append(value)

    if column_limit and column_limit < len(col_names):
        for col in ordered_names[column_limit:]:
            df[col] = df[col].apply(lambda x : 0)

    if threshold: #AFTER KNOWING A THRESHOLD FAIL THEN YOU CAN ZERO THE REST OF COLUMNS (TODO)
        if column_limit:
            r = range(column_limit)
        else:
            r = range(len(col_names))

        for i in r:
            col = ordered_names[i]
            val = ordered_values[i]
            if val < threshold:
                df[col] = df[col].apply(lambda x : 0)
                counter +=1

    return df

def dynamic_transformations(data_train, data_predict, data_target, transformations=[np.log, np.sqrt], verbose=False):
    '''
        Parameters
        __________

        data_train : Pandas dataframe or numpy array.
            Training data in an mxn pandas dataframe (m samples and n predictors)
        data_predict : Pandas dataframe or numpy array 1xn.
        data_target :  Pandas dataframe or numpy array 1xm
    '''

    if verbose:
        print('Entering dynamic transformations')
    final_data =copy.copy(data_train)
    final_predict = copy.copy(data_predict)
    for transform in transformations:
        if verbose:
            print(transform)
        transformed_data = transform(data_train)

        transformed_data[transformed_data.isnull()] = 0
        transformed_data[transformed_data == float('-inf')] = 0
        transformed_data[transformed_data == float('inf')] = 0

        for name in list(transformed_data):
            transformed_corr = stats.pearsonr(transformed_data[name].values.ravel(), data_target.ravel())[0]
            final_corr = stats.pearsonr(final_data[name].values.ravel(), data_target.ravel())[0]

            if verbose:
                print('comparing transformations : {0}, {1}'.format(transformed_corr, final_corr))
            if transformed_corr > final_corr:
                if verbose:
                    print('Replacing values in term {0}. {1} vs {2}'.format(name, transformed_corr, final_corr))
                    print(final_predict[name])
                final_data[name] = copy.copy(transformed_data[name].values)

                final_val = transform(data_predict[name])

                if final_val == float('inf') or final_val == float('-inf') or final_val == float('nan'):
                    final_predict[name] = 0
                else:
                    final_predict[name] = final_val

    return final_data, final_predict

def check_predict_features(X_train, X_predict):
    counter = 0
    location = []
    for i,v in enumerate(X_predict):
        if np.isnan(v):
            X_train[:,i] = 0
            location.append(i)
            counter += 1


    return X_train, X_predict, counter, location

def mse(predictions, targets):
   return ((predictions - targets) ** 2).mean()

def handle_zero_vectors(X_train, X_pred):
    non_zero_vector_indices = []
    zero_vector_indices = []
    for i, v in enumerate(np.equal( X_train.sum(axis=0),0)):
        if v:
            zero_vector_indices.append(i)
        else:
            non_zero_vector_indices.append(i)

    return X_train.iloc[:,non_zero_vector_indices], X_pred[non_zero_vector_indices], zero_vector_indices

# Season analysis data

def get_all_periods(metrics_df, model, metric, periods):

    values = []

    sub_df = metrics_df[(metrics_df['MODEL'] == model) & (metrics_df['METRIC'] == metric)]

    for period in periods:
        values.append(sub_df[period].values)
    return np.concatenate(values).reshape(-1, 1)


def getRanks(metrics_df, metric, ids, models, periods):

    """
        Returns a matrix containing counts of the number of times a model obtained first, second, third... etc place
        in all the periods. The rows indicates the rank  (0 is first place, 1 is second, etc) and column specifies the model
        (models[0] to column 0, models[1] to column 1, etc)
    """

    n_models = len(models)
    ranks = np.zeros([n_models, n_models])

    for id_ in ids:


        sub_df = metrics_df[(metrics_df['METRIC'] == metric) & (metrics_df['ID'] == id_)][['MODEL']+periods]


        for period in periods:
            values = []
            for mod in models:
                values.append(sub_df[sub_df['MODEL'] == mod][period].values)

            tups = zip(models, values)


            if metric in ['PEARSON']:
                ordered_tups = sorted(tups, key=lambda x:x[1], reverse=True)
            else:
                ordered_tups = sorted(tups, key=lambda x:x[1])


            for  i, tup in enumerate(ordered_tups):
                model_index = models.index(tup[0])
                ranks[i, model_index] += 1

    return ranks
