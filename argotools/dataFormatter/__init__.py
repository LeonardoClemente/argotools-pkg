import pandas as pd
import time
import copy
import numpy as np
from argotools.forecastlib.functions import *
from argotools.config import *
import os


class Load:

    """
    The Load class generates an object that aids in the management of data throughout the whole ARGO methodology forecastingself.
    If you're working with a database or a dataset that already has a format, you can try used LoadFromTemplate class, which generates
    objects that import data from already used templates.

    The Load class imports information from both the Target of interest and the input features used in modelling based
    on a few assumptions of the file paths indicated for each.

    The period parameters are fixed to every id in the data to avoid confusion.

    Parameters
    __________
    start_period : str, optional
        String of the form YYYY-MM-DD or the specific index type used to specify the
        lower limit of the domain of the data sample index.
    end_period : str, optional
        String of the form YYYY-MM-DD or the specific index type used to specify the
        upper limit of the domain of the data sample index.

    Attributes
    __________

    id : list of str
        List containing the id names for the objects of interest to study
    target: dict
        dictionary that contains the target information for each id organized in a pandas dataframe.
    features: dict
        dictionary that contains the features information for each id organized in a pandas dataframe.
        Features are considered any type of information that counts as a proxy to the target activity.
        They're uploaded using the new_id() function or .add_features() function.
    ar_lags: dict
        dictionary that contains the autoreggresive models information for each id organized in a pandas dataframe.
        Each column corresponds to a different AR lag and is identified with a number in their name. (E.G.
        If a column title for the dataframe is 'AR_3', this is corresponds to the target information at a time
        t-3)
    """

    def __init__(self, start_period=None, end_period=None, ids = None):

        self.id = []
        self.data = {}
        self.target = {}
        self.file_paths = {}
        self.features = {}
        self.ar_lags = {}
        self.ar_lag_ids = {}
        self.start_period = start_period
        self.end_period = end_period

        if ids:
            for id_ in ids:
                self.new_id(id_)
            print('Load object initialized with the following ids : {0}'.format(ids))

    def new_id(self, id_):

        if id_ not in self.id:
            self.id.append(id_)
            self.target[id_] = None
            self.features[id_] = None
            self.ar_lags[id_] = None
            self.ar_lag_ids[id_] = None
        else:
            print('id:{0} already initialized'.format(id_))

        return
    def read_formatted_data(self,name=None, path_to_target=None, path_to_features=None,\
               merge_indices=None, verbose=True, col_name=None):
        """
         The new_id function is the main option to upload information to the Load object. It loads
         data from two different sources specified from the user. NOTE: new_id assumes that the data
         read in has already been formatted to a csv format readable by pandas (no skiprows, index in the first column,
         only timeseries data after the first column). To load data from unformatted (such as google correlate)
         please see "read_from_source" and its possible readings)

         Parameters
         __________

         name : str
            Name to identify the object to analyze.
        path_to_target : str
            The path to the file containing the target information.
            The file is assumed be a csv that's pandas readable, the index of the
            dataframe being the very first column of the file and that the target
            can be identified as a column with the name parameter as identifier.
        path_to_features :  list
            The list of paths to the feature information.
            The file is assumed be a csv that's pandas readable, the index of the
            dataframe being the very first column of the file and that the rest of
            the columns can be identified as features of interest for the target
        merge_indices : Boolean, optional (default None)
            If indices between pandas dataframe of the target and the features is not equal
            but equal in length, it replaces the feature's index with the target indexself.
            If set to None, the it asks directly the user for the input.
        col_name : Str, optional (default None)
            Column name to look for in the target data. If set to None, function
            looks up the region name.
        """
        sp = self.start_period
        ep = self.end_period

        if col_name:
            pass
        else:
            col_name = name

        if os.path.isfile(path_to_target):

            try:
                target_df = pd.read_csv(path_to_target, index_col=[0])
                series = copy.copy(target_df[col_name][sp:ep].to_frame())

                #Renaming series to avoid naming problems
                series.columns = [name]
                self.target[name] = series
            except Exception as t:
                print('Could not load target data successfully:{0}'.format(t))

        else:
            print('Path to target information is not recognized as a real file path. \
                  Please verify this information')
            return

        features = None
        for p in path_to_features: #Repeating same process for all paths
            if os.path.isfile(p):

                try:
                    feature_df = pd.read_csv(p, index_col=[0])
                    if features is None:
                        features = copy.copy(feature_df[sp:ep])
                    else:
                        features = pd.concat([features, feature_df[sp:ep]], axis=1)
                except Exception as t:
                    print('Could not load feature data successfully:{0}'.format(t))

            else:
                print('Path {0}  is not recognized as a real feature file path. \
                      Please verify this information'.format(p))

        target_index = self.target[name].index.values
        features_index = features.index.values
        #Check if indices are identical
        if len(target_index) == len(features_index):
            equal_indices = [i == j for i,j in zip(target_index, features_index)]
            if all(equal_indices):
                pass
            else:
                print('WARNING! Some indices values in the features and target do not \
                      coincide. Please revise the following')
                for i, is_equal in enumerate(equal_indices):
                    if not is_equal:
                        print('Index row {0} with features index = {1} and target index = {2} \n'.format(i, features_index[i], target_index[i]))
                if merge_indices:
                    features.set_index(self.target[name].index, inplace=True)
                else:
                    s =input('Use the same index in the features? (Y/N)')
                    if s == 'Y':
                        features.set_index(self.target[name].index, inplace=True)
        else:
            print('WARNING! Feature and target indices do not have the the same length ({0}, {1}). \
                  \n  First indices: target={2}   features={3} \n Last indices: target{4} \
                  features = {5}'.format(len(target_index), len(features_index), target_index[0],\
                                         features_index[0], target_index[-1], features_index[-1]))
            time.sleep(2)

        self.features[name] = features
        self.id.append(name)

        if verbose: print('Successfully loaded id ({0}) data.'.format(name))

    def add_target(self, id_, path_to_file, target_name=None, verbose=False):

        '''
        Function that individuallly adds target data to the load object. NOTE:
        add_target assumes data has a basic readable pandas csv format (no skiprows, index at 0th column,
        only features after the 0th column). To read data with a different style, please refer to add_target_customSource

        Parameters
        __________

        id_ : string
            Identifier (must be initialized)

        path_to_file : string
            String contianing the path to the csv

        target_name : string, optional (default is None)
            column name for the target in the pandas DataFrame. If set to None, add_target assumes the data is named
            'id_'
        '''

        if id_ in self.id:
            df = pd.read_csv(path_to_file, index_col=0)
            if target_name:
                self.target[id_] = df[target_name][self.start_period:self.end_period].to_frame()
            else:
                self.target[id_] = df[id_][self.start_period:self.end_period].to_frame()
        else:
            print("ID not found within object. Please initialize it using 'new_id' function ")

    def add_features_customSource(self, id_, path_to_file, source='GC', overwrite=False, verbose=True, autoanswer=None):
        '''
        Function designed to read data from a specific source and add it as features for the specific ID.

        Parameters
        __________

        id_ : string
            Name of the location of interest
        path_to_file : string
            Path to file to read in.
        source : string, optional (default is 'GC')
            Specify the source of the data to select reading format. Please see available options in handlers
        overwite : Boolean, optional (Default is False)
            If data already exists within id_ features, then erase and only use this data as input. Otherwise, merge
            data
        autoanswer = String, optional (default is None):
            Automate response to the input query that might arise. answer can be 'y' for yes or 'n' for no.
        '''

        reader_functions = {
            'GC': load_gc_features,
            'standard':read_standard_features
        }

        features = reader_functions[source](path_to_file, self.start_period, self.end_period)

        if verbose: print("Following features read from file : {0}".format(features.iloc[0:5,:]))


        if isinstance(self.features[id_], pd.DataFrame):
            old_features = self.features[id_]

            if old_features.index.equals(features.index):
                print('WARNING! New features index have a different index. \n \
                      the following indices from the new features are not within the old index')

                for ind in list(features.index.values):
                    if ind not in list(old_features.index.values):
                        print(ind)

                answer = input('Would you still like to merge both sources of data? y/n')

                if answer == 'y':
                    new_features = pd.concat([old_features, features], axis=1)
                    self.features[id_] = new_features
                    print('New features generated for {0} : \n {1}'.format(id_, new_features))


                if answer == 'n':
                    print('New data ignored.')
                    return

        elif self.features[id_] is None:
            self.features[id_] = features

        if isinstance(self.target[id_], pd.DataFrame):

            if not self.features[id_].index.equals(self.target[id_].index):

                print('WARNING! features and target indices for {0} are not equal. Indices from \
                      features not within the target are the following:'.format(id_))
                for ind in list(self.features[id_].index.values):
                    if ind not in list(self.target[id_].index.values):
                        print(ind)

                print('Index sizes: features = {0} target = {1}'.format(np.size(self.features[id_].index),np.size(self.target[id_].index)))
        #check if data from features already exist
    def add_features(self, id_=None, file_to_features=None, append=True):

        try:
            if append:
                features = self.features[id_]
            else:
                features = None
        except Exception as t:
            print('An error ocurrer when checking the append parameter : {0}'.format(t))
            return

        if os.path.isfile(file_to_features):

            try:
                feature_df = pd.read_csv(file_to_features, index_col=[0])

                if features is None:
                    features = copy.copy(feature_df[sp:ep])
                else:
                    features = pd.concat([features, feature_df[sp:ep]], axis=1)

                self.features[id_] =features
                print('Successfully added the specified features')

            except Exception as t:
                print('Could not load feature data successfully:{0}'.format(t))
                return
        else:
            print('Path {0}  is not recognized as a real feature file path. \
                  Please verify this information'.format(p))
            return
    def add_feature_lags(self, id_, feature_name, which_lags, verbose = False, store=True):
        '''
            Generate autoreggresive lags for an specific data id.
            Inputs are the specific Id (must exist within data.ids)
            and the lag terms (list containing the integer numbers)

            generate_ar_lags outputs a dataFrame containing the lags within
            a column id format of 'AR' + lag_number (e.g. AR14)
        '''
        lags = {}
        success = []
        for i, lag_number in enumerate(which_lags):
            try:
                lag = generate_lag(self.features[id_][feature_name].values, lag_number)
                lags['{1}{0}'.format(lag_number, feature_name)] = lag.ravel()
                success.append(lag_number)
            except IOError:
                print('There was a problem while generating  lag {0} for  feature : {1}. Please review your data'.format(lag_number, feature_name))

        lag_df = pd.DataFrame(lags, index = self.features[id_].index.values)
        if store == True:
            self.features[id_] = pd.concat([self.features[id_], lag_df], axis=1)
        else:
            return lag_df

        if verbose == True:
            print('Successfully generated the following feature terms : {0}'.format(success) )
    def generate_ar_lags(self, which_id, which_lags, verbose = False, store=True):
        '''
            Generate autoreggresive lags for an specific data id.
            Inputs are the specific Id (must exist within data.ids)
            and the lag terms (list containing the integer numbers)

            generate_ar_lags outputs a dataFrame containing the lags within
            a column id format of 'AR' + lag_number (e.g. AR14)
        '''
        lags = {}
        success = []
        for i, lag_number in enumerate(which_lags):
            try:

                lag = generate_lag(self.target[which_id].values, lag_number)
                lags['AR{}'.format(lag_number)] = lag.ravel()
                success.append(lag_number)
            except IOError:
                print('There was a problem while generating autoregressive term {0}. Please review your data'.format(lag_number))

        if store == True:
            self.ar_lags[which_id] = pd.DataFrame(lags, index = self.features[which_id].index.values)
        else:
            return pd.DataFrame(lags, index = self.features[which_id].index)

        if verbose == True:
            print('Successfully generated the following AR terms : {0}'.format(success) )
    def interpolate_single_values(self, id_=None, target = True, features=False):
        """
            This function finds missing values within dataset
            that have no neighbor observations with missing values as well and performs
            linear interpolation to fill them out.
        """
        # Check for id
        if id_ in self.id:
            pass
        else:
            print('Id not recognized. Please check.')
            return


        target = self.target[id_]

        nan_locs = np.isnan(target).values
        nan_locs = [i for i,v in enumerate(nan_locs) if v == True]
        multiple_nans = []
        if 1 in nan_locs:
            nan_locs.remove(1)
            multiple_nans.append(1)
        if len(target)-1 in nan_locs:
            nan_locs.remove(len(target)-1)
            multiple_nans.append(len(target)-1)


        for loc in nan_locs:

            if not np.isnan(target.iloc[loc+1].values) and not np.isnan(target.iloc[loc-1].values):
                target.iloc[loc] = (target.iloc[loc-1].values + target.iloc[loc+1].values )/2
            else:
                multiple_nans.append(loc)

        print('Interpolating data for {3}, {0} NaNs found in timeseries. {1} were single, {2} were multiple'.format(\
                len(nan_locs), len(nan_locs) - len(multiple_nans), len(multiple_nans), id_))
    def interpolate_double_values(self, id_=None, target = True, features=False):
        """
            This function finds missing values within dataset
            that have only one neighbor observations with missing values as well and performs
            linear interpolation to fill them out.
        """
        # Check for id
        if id_ in self.id:
            pass
        else:
            print('Id not recognized. Please check.')
            return


        target = self.target[id_]

        nan_locs = np.isnan(target).values
        nan_locs = [i for i,v in enumerate(nan_locs) if v == True]
        multiple_nans = []
        n_locs = len(nan_locs)
        #checking boundaries
        if 1 in nan_locs:
            if 2 in nan_locs:
                if 3 not in nan_locs:
                    target.iloc[1] = target.iloc[3].values
                    target.iloc[2] = target.iloc[3].values
                    nan_locs.remove(1)
                    nan_locs.remove(2)
                else:
                    nan_locs.remove(1)
                    nan_locs.remove(2)
                    multiple_nans.append(1)
                    multiple_nans.append(2)
            else:
                target.iloc[1] = target.iloc[2].values
                nan_locs.remove(1)

        last_index =  len(target)-1
        if  last_index in nan_locs:
            if last_index -1 in nan_locs:
                if last_index -2 in nan_locs:
                    multiple_nans.append(last_index)
                    multiple_nans.append(last_index-1)

                else:
                    target.iloc[last_index] = target.iloc[last_index - 2 ].values
                    target.iloc[last_index-1] = target.iloc[last_index - 2].values
                nan_locs.remove(last_index)
                nan_locs.remove(last_index-1)
            else:
                target.iloc[last_index] = target.iloc[last_index-1].values
                nan_locs.remove(last_index)
                multiple_nans.append(last_index)
        elif last_index -1 in nan_locs:
            if last_index - 2 in nan_locs and last_index - 3 not in nan_locs:
                delta = (target.iloc[last_index]-target.iloc[last_index-3])/3
                target.iloc[last_index-2] = target.iloc[last_index - 3].values + delta
                target.iloc[last_index-1] = target.iloc[last_index - 3].values + delta*2
                nan_locs.remove(last_index-2)
                nan_locs.remove(last_index-1)
            else:
                nan_locs.remove(last_index-2)
                multiple_nans.append(last_index)

        for loc in nan_locs:

            left_neighbor = target.iloc[loc-1].values
            right_neighbor = target.iloc[loc+1].values
            second_right_neighbor = target.iloc[loc+2].values

            if np.isnan(right_neighbor) and not np.isnan(left_neighbor) and not np.isnan(second_right_neighbor):
                delta = (second_right_neighbor-left_neighbor)/3
                target.iloc[loc] = left_neighbor + delta
                target.iloc[loc+1] = left_neighbor + delta*2
                nan_locs.remove(loc)
                nan_locs.remove(loc+1)
            elif not np.isnan(left_neighbor and not np.isnan(right_neighbor)):
                multiple_nans.append(loc)
            elif np.isnan(left_neighbor):
                multiple_nans.append(loc)
                nan_locs.removes(loc)
            else:
                multiple_nans.append(loc)
                multiple_nans.append(loc+1)
                nan_locs.remove(loc)
                nan_locs.remove(loc+1)

        print('Interpolating data for {3}, {0} NaNs found in timeseries. {1} were double, {2} were either single or above 2 missing values'.format(\
                n_locs, n_locs - len(multiple_nans), len(multiple_nans), id_))
    def remove_sparse_features(self, id_=None, sparse_value=0, threshold=.80):

        if id_ in self.id:
            pass
        else:
            print('Id not recognized. Please check.')
            return

        features = self.features[id_]
        removed_cols = []
        for col_name in features:

            n_nans = np.equal(features[col_name].values, sparse_value).sum()

            if n_nans > len(features.index)*threshold:
                del features[col_name]
                removed_cols.append(col_name)

                print('Columns removed based on a {0} threshold : {1}'.format(threshold, col_name))
    def patch_missing_values(self, id_=None, value=None):
        '''
        This function maps missing values (nans) within dataframe to an specific value.
        The idea of mapping a missing value is to avoid LASSO from breaking out (it doesn't run on NAN data)
        This idea is usable only if the number of missing values is low. (Most likely used in the areas where there's more
        than two missing values). WARNING!: Having too many mapped values could make your model to overshoot pretty easily
        Please use this as your last resort.
        '''
        if value is None:
            value = NAN_TO_VALUE
        if id_ in self.id:
            pass
        else:
            print('Id not recognized. Please check.')
            return
        target = self.target[id_]
        nan_locs = np.isnan(target).values
        target[nan_locs]=value
        print('{0} NANs mapped within data.'.format(nan_locs.sum()))








class LoadFromTemplate:
    '''
    meta_data = {
        data_type =   geo, finance, etc
        ids =   identifier (used in load functions to extract data correctly)
        per_id_file_path = []
        per_id_fileformat = []
        index_type =  index type (numeric, date, etc)
        index_label =
        start_period =
        end_period =
        single_output_dir =
        condensed_output_dir =
    }
    '''

    def __init__(self, meta_data = None):

        load_handler = {
            'standard': load_standard,
            'latam': load_latam,
        }


        self.id = []
        self.data = {}
        self.target = {}
        self.file_paths = {}
        self.features = {}
        self.benchmarks = {}
        self.ar_lags = {}
        self.ar_lag_ids = {}

        if meta_data is not None:

            self.id = meta_data['ids']

            for i, ID in enumerate(self.id):

                file_format = meta_data['per_id_file_format'][i]

                try:
                    target, features, benchmarks = load_handler[file_format](meta_data['per_id_file_path'][i], \
                                             meta_data['index_type'], \
                                             meta_data['per_id_index_label'][i], \
                                             meta_data['per_id_start_period'][i], \
                                             meta_data['per_id_end_period'][i])
                except IOError:
                    print('Error ocurred while using load_handler. Please verify data')

                self.target[ID] = target
                self.features[ID] = features
                self.ar_lags[ID] = None
                self.benchmarks[ID] = benchmarks
                self.file_paths[ID] = meta_data['per_id_file_path'][i]

            print('Data object initialized with dictionary data.')
        else:
            print('Data object initialized. Please add your data.')



    def add_id(self, id_tag=None):
        return

    def generate_ar_lags(self, which_id, which_lags, verbose = False, store=True):
        '''
            Generate autoreggresive lags for an specific data id.
            Inputs are the specific Id (must exist within data.ids)
            and the lag terms (list containing the integer numbers)

            generate_ar_lags outputs a dataFrame containing the lags within
            a column id format of 'AR' + lag_number (e.g. AR14)
        '''
        lags = {}
        success = []
        for i, lag_number in enumerate(which_lags):
            try:

                lag = generate_lag(self.target[which_id], lag_number)
                lags['AR{}'.format(lag_number)] = lag.transpose()
                success.append(lag_number)
            except IOError:
                print('There was a problem while generating autoregressive term {0}. Please review your data'.format(lag_number))

        if store == True:
            self.ar_lags[which_id] = pd.DataFrame(lags, index = self.features[which_id].index.values)
        else:
            return pd.DataFrame(lags, index = self.features[which_id].index.values)

        if verbose == True:
            print('Successfully generated the following AR terms : {0}'.format(success) )

    def add_features(self):
        return
    def add_target(self):
        return
    def add_benchmark(self):
        return

def find_kinds(dataFrame):
    return

def generate_lag(timeseries, lag_number):
    #Transforming to numpy array.

    timeseries = np.array(timeseries)
    lag = np.zeros_like(timeseries)
    series_length = np.size(timeseries)
    lag[lag_number: series_length] = timeseries[0: series_length - lag_number]
    return lag
