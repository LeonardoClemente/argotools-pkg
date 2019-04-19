import numpy as np
import pandas as pd
import argotools.dataFormatter as dt
import os
import time
from argotools.config import *
from argotools.forecastlib.handlers import *
import datetime
import copy
import pickle
from progress.bar import ChargingBar
import scipy.stats
'''
Experiment class
'''


class Predictor:

    '''
        Predictor class is used as a container for an specific model to run.
        A Predictor object is created to contain all data and pre-processing
        related to a certain model.

        initialized using the following parameters:
            - model: A model must be specified. Predictor class sets the
            prediction_function and preproc_function attributes based on this
            parameter.

            - mode: predictor class works on two modes, "store" and "single".
            For "store" modem on every predict(), call, predictor class generates and stores a
            new set of predictions, coefficients and hyperparams . In case you
            don't need to generate all of this information, its possible to
            change the settings calling the store_settings() function.

        object attributes:
            -Predictions : single or multi valued.
            -coefficients : the coefficients generated to fit the model.
            -hyper_params : Parameters used to fit the respective models.
                e.g. Lasso's Alpha value
            -prediction_function : contains the specified model
            -preproc_function : contains a preprocessing function based on the model
    '''
    def __init__(self, model = 'ARGO', mode = 'store', metric=metric_RMSE, model_funct=None, model_preproc=None,\
                 mod_params = None, verbose=True):

        if verbose: print('Starting predictor {0} with model={1} and model_preproc={2}'.format(model, mod_params, model_preproc))
        self.predictions = []
        self.hyper_params = []
        self.coefficients = []
        self.insample_metric = []
        self.out_of_sample_metric = []
        self.current_model = None
        if model_funct:
            self.prediction_function = model_funct
        else:
            self.prediction_function = method_handler[model]
        if model_preproc:
            self.preproc_function = model_preproc
        else:
            self.preproc_function = preproc_handler[model]

        self.mod_params = mod_params
        self.mode = mode
        self.metric = metric

        # Store settings
        self.store_predictions = True
        self.store_hyper_params = True
        self.store_coefficients = True
        self.store_settings = [self.store_predictions, self.store_hyper_params, \
                              self.store_coefficients]


    def fit(self, X_train, Y_train, X_pred, X_test=None, Y_test=None):

        X_train, Y_train, X_pred = self.preproc_function(X_train, Y_train, X_pred)
        X_train, X_pred, zero_vector_indices = handle_zero_vectors(X_train, X_pred)

        self.current_model = self.mod_params.fit(X_train, Y_train)

        if self.metric:
            preds = self.current_model.predict(X_train)
            metric_value = self.metric(preds[-12:], Y_train[-12:])#CHANGE

            if X_test is not None:

                ind = list(range(X_test.shape[1]))

                for i in zero_vector_indices:
                    if X_test.shape[0] == 1:

                        X_test[0,i] = 0
                    else:
                        X_test[:,i] = 0
                    ind.remove(i)

                ind = np.array(ind)


                X_test, Y_test, X_pred = self.preproc_function(X_test, Y_test, X_pred)
                X_test = np.array(X_test)

                if X_test.shape[0] == 1:
                    preds = self.current_model.predict(X_test[0,ind].reshape(1,-1))
                elif X_test.shape[0] == 0:
                    print('No out of sample dataset')
                    preds = np.zeros(np.size(Y_test)).ravel()

                else:
                    preds = self.current_model.predict(X_test[:,ind])
                oos_metric_value = self.metric(preds, Y_test.ravel())#CHANGE
            else:
                oos_metric_value = None

        else:
            metric_value = None
            oos_metric_value = None

        if self.mode == 'single':
            self.insample_metric = metric_value
            self.out_of_sample_metric = oos_metric_value

        elif self.mode == 'store':
            self.insample_metric.append(metric_value)
            self.out_of_sample_metric.append(oos_metric_value)


    def predict2(self, X_train, Y_train, X_pred):
        X_train, Y_train, X_pred = self.preproc_function(X_train, Y_train, X_pred)
        X_train, X_pred, zero_vector_indices = handle_zero_vectors(X_train, X_pred)

        predictions = self.current_model.predict(X_pred.reshape(1,-1)).ravel()

        #After predictions

        if isinstance(self.current_model, (LassoCV, Lasso)):
            coefficients, hyper_params = lasso_params(self.current_model, zero_vector_indices)

        if self.mode == 'single':
            self.predictions = predictions
            self.coefficients = coefficients
            self.hyper_params = hyperparams

        elif self.mode == 'store':

            if self.store_predictions == True:
                self.predictions.append(predictions)
            if self.store_coefficients == True:
                self.coefficients.append(coefficients)
            if self.store_hyper_params == True:
                self.hyper_params.append(hyper_params)



    def preproc(self, X_train, Y_train, X_pred):

        return self.preproc_function(X_train, Y_train, X_pred)

    def predict(self, X_train, Y_train, X_pred):
        # Input must be numpy arrays with samples as rows and features as columns
        try:
            X_train, Y_train, X_pred = self.preproc(X_train, Y_train, X_pred)

        except IOError:
            print('Error ocurred while preproccessing. Please check your data.')

        try:
            predictions, coefficients, hyper_params, metric_value = self.prediction_function(X_train,\
                                                                              Y_train,\
                                                                              X_pred, \
                                                                              self.store_settings,\
                                                                              self.mod_params,\
                                                                              self.metric)
        except IOError:
            print('Error ocurred while calling the prediction function. Pease verify.') #erika barcenas

        if self.mode == 'single':
            self.predictions = predictions
            self.coefficients = coefficients
            self.hyper_params = hyperparams
            self.insample_metric = metric_value

        elif self.mode == 'store':

            if self.store_predictions == True:
                self.predictions.append(predictions)
            if self.store_coefficients == True:
                self.coefficients.append(coefficients)
            if self.store_hyper_params == True:
                self.hyper_params.append(hyper_params)

            if self.metric:
                self.insample_metric.append(metric_value)


    def store_settings(store_predictions = True, store_coefficients = True, \
                       store_hyper_params = True, verbose = True):

        if store_predictions == False and store_coefficients == False \
        and store_hyper_params == False:
            print('WARNING! All storing settings have been turned off.')

        self.store_predictions = store_predictions
        self.store_hyper_params = store_hyper_params
        self.store_coefficients = store_coefficients

        self.store_settings = [self.store_predictions, self.store_hyper_params, \
                              self.store_coefficients]

        if verbose == True:
            print('Store settings successfully modified.')

    def clean(newData):

        self.predictions = []
        self.alpha = []
        self.coefficients = []

class  ModelSelector:
    '''
        The stacking forecast class is used to generate predictions based on fitted predictors
        using timeseries data.

        StackingForecast is used to do ensemble exploration and analysis. It is not intended for
        deployment (although it can be used for that).
    '''

    def __init__(self, params, data, load_folder = None):

        '''
            params = {
                id = '',
                period_start = '',
                id_period_end = '',
                ensemble_models = '' List containing the ensemble models (To consult a list of available ensemble_models
                type experiment.display('ensemble_models'))

                predictors_per_ensemble = {} Dictionary containing the predictors on each ensemble.
                    Can also be a list containing a set of predictors. The list will be used on every ensemble.


                OPTIONAL
                output_folder = str containing the path to produce all results
                meta_features = dict containing additional information that may be needed by an ensemble model
                training_window = '' # window type ('static', 'expanding')
                training_window_size = int or str  Number of data points (starting from start period or str 'expanding')
                ensemble
                horizon = {} # Number of data points to predict ahead of most recent training sample index
                    e.g. if last training sample index is '10' or '10-ago-2018', then horizon 1 would make Predictions
                    for index '11' or '18-ago-2018' (in case of weekly index), horizon 2 would predict for index '12' or
                    '25-ago-2018', and so on.
                feature_preproccessing = String
            }

            id_data = A Data class object containing the neccesary information regarding the ID of interest

            load_folder = String containing the path to an EnsembleForecast experiment.
        '''

        id_ = params['id']

        try:
            path_output = params['output_folder'] + '/'
        except:
            print('No output folder detected for the experiment. Using {0} as default.'.format(os.get_cwd()))
            path_output = ''

        self.id = id_
        self.folder_output = gen_folder(path_output + id_)

        if self.folder_output != path_output+id_:
            print('Warning! Folder with ID already exists. Using {0} as folder instead'.format(self.folder_output))

        if not params['id'] in data.id:
            print('Data object does not contrain information about {0}. Please review your input'.format(self.id))
            return

        self.target = data.target[id_]
        self.features = data.features[id_]
        try:
            self.benchmarks = data.benchmarks[id_]
        except AttributeError:
            print('Initializing ensemble model for {0} without any benchmarks.'.format(id_))


        if isinstance(params['ensemble_models'],list):

            self.ensemble_list, dropped_labels = handlerContains('ensembles', params['ensemble_models'])

            if dropped_labels == len(params['ensemble_models']):
                print('Ensemble model list not contained within handler keys. Please review your data \
                      or consult available models.')
                return
            if dropped_labels > 1:
                print('Ommited some ensemble models that were not contained within the handlers.\
                      please review your input')
        else:
            print('Parameter : ensemble models should contain a list. not a {0}'.format(type(params['ensemble_models'])))

        # Initializing predictors
        if isinstance(params['predictors_per_ensemble'],list):

            self.predictor_list, dropped_labels = handlerContains('methods', params['predictors_per_ensemble'])
            if dropped_labels > 0 :
                print('Some predictors were not found within handler list. Please review your input.')

            self.predictors_per_ensemble = dict(zip(self.ensemble_list, [self.predictor_list]*len(self.ensemble_list)))

        elif isinstance(params['predictors_per_ensemble'], dict):
            concat_predictors = []
            self.predictors_per_ensemble = {}

            for key, predictor_list in params['predictors_per_ensemble'].items():
                contained, dropped_labels = handlerContains('methods', predictor_list)
                print('{0} models found for {1} ensemble model'.format(contained, key))
                concat_predictors += contained
                self.predictors_per_ensemble[key] = contained

                if dropped_labels == len(predictor_list):
                    print('Predictors not found for {0}. Please review your input'.format(id_))
                    return
                elif dropped_labels > 0:
                    print('Some predictors were not found within handler list. Please review your input.')


            self.predictor_list = list(set(concat_predictors))


        if len(self.predictor_list) == 0:
            print('Couldnt register any predictors for the ensemble models. Please review your input')
            return

        self.period_start = params['period_start']
        self.period_end = params['period_end']


        # Checking optional values.
        try:
            self.training_window = params['training_window']
            self.training_window_size = params['training_window_size']
        except KeyError:
            print('No training window information found, setting to default.')
            self.training_window = TRAINING_WINDOW_TYPE_DEFAULT
            self.training_window_size = TRAINING_WINDOW_SIZE_DEFAULT

        try:
            self.horizon = params['horizon']
        except KeyError:
            print('No horizon data found. Setting to default')
            self.horizon = 1

        try:
            self.feature_preproc = params['feature_preproccessing']
        except KeyError:
            print('No feature preprocessing found. Setting to default.')
            self.feature_preproc = FEATURE_PREPROCESSING_DEFAULT




        self.ar_model = {}

        try:
            concat_ar = []
            for i, predictor in enumerate(self.predictor_list):
                if predictor in params['ar_model'] and isinstance(params['ar_model'][predictor], list):
                    self.ar_model[predictor] = params['ar_model'][predictor]
                    concat_ar += params['ar_model'][predictor]
                else:
                    print('Ar_model initialization for {0} contains errors. Using default. \
                          please check your input'.format(predictor))
                    self.ar_model[predictor] = ar_handler[predictor]
                    concat_ar += ar_handler[predictor]



        except KeyError:
            concat_ar = []
            print('No AR_model specification found. Looking in handlers or setting to default.')
            for i, predictor in enumerate(self.predictor_list):
                    self.ar_model[predictor] = ar_handler[predictor]
                    concat_ar += ar_handler[predictor]

        self.ar_lag_list = list(set(concat_ar))
        self.ar_lags = data.generate_ar_lags(which_id= id_, which_lags = self.ar_lag_list, store=False)

        try:
            self.feature_preproc = params['feature_preprocessing']
        except KeyError:
            print('feature_preprocessing not identified. Using default.')
            self.feature_preproc = FEATURE_PREPROCESSING_DEFAULT

        #Writing out data to overview_folder
        #data_file = open(self.output_folder + '/' + TXT_DATA ,'w')
        experiment_file = open(self.folder_output + '/' + TXT_EXPERIMENT , 'w')

        try:
            if isinstance(params['meta_features'],dict):
                self.meta_features = params['meta_features']
                for model in self.ensemble_list:
                    if model not in self.meta_features:
                        print('No meta-feature initialization found for {0}.'.format(model))
                        self.meta_features[model] = None
        except KeyError:
            print('No meta features assigned for ensemble-models.')
            self.meta_features = dict(zip(self.ensemble_list, [[None]]*len(self.ensemble_list)))

        for k, v in params.items():
            experiment_file.write('{0}:\n\n{1}\n\n'.format(k,v))

        #for k, v in meta_data.items():
        #    data_file.write('{0}: \n\n{1} \n\n'.format(k,v))

        #data_file.close()
        experiment_file.close()


    def run(self, verbose = False):

        if verbose : print('Initializing variables')

        id_ = self.id

        # Entering main loop
        id_dir = self.folder_output

        #Entering data generation loop per id
        highest_ar_lag = np.max(self.ar_lag_list)
        target = self.target
        features = self.features
        period_start = self.period_start
        period_end = self.period_end
        horizon = self.horizon - 1
        feature_preproc = self.feature_preproc
        meta_features = self.meta_features
        ar_model = self.ar_model
        w = self.training_window
        s = self.training_window_size

        # Initializing predictors and AR terms.
        if verbose : print('Initializing predictors and ensembles')
        predictors = {}
        ensembles = {}

        for j, model in enumerate(self.predictor_list):
            predictors[model] = Predictor(model=model)

        for j, model in enumerate(self.ensemble_list):
            ensembles[model] = StackEnsemble(model=model, meta_features=meta_features[model])


        if isinstance(s, int) and s < features.shape[0]:
            training_window_size = s
        else:
            print('WARNING! Error with training window size. Using default')
            training_window_size = TRAINING_WINDOW_SIZE_DEFAULT

        try:
            if w in training_window_handler.keys():
                training_window = w
                training_window_size = s

        except KeyError:
            print('WARNING! Error defining the training window size. Using default')
            training_window = TRAINING_WINDOW_TYPE_DEFAULT

        if verbose : print('Generated window size = {0} and type {1}'.format(training_window, training_window_size))

        index_start = features.index.get_loc(period_start)
        index_end = features.index.get_loc(period_end)
        first_available_index =  highest_ar_lag + horizon + training_window_size
        Y_ensemble = []
        Y_index = []

        if index_start >= first_available_index:
            first_prediction_index = index_start
            first_window_index = index_start - training_window_size
        else:
            print('Prediction start date falls within minimum training dataset period \n \
                  period_start=[{0}, loc_in_df:{1}] \n \
                  first_available_index=[{2}, loc_in_df:{3}] \n \
                  highest_ar_lag = {4}, training_window_size {5}. \n\n \
                  Prediction should start at index {1} but model minimum\
                   data points needed for training set at start is: {6}.\n \
                   Please update the training dataset or autoreggresive parameters.\
                  '.format(period_start, index_start, first_available_index,\
                           features.index[first_available_index], highest_ar_lag,\
                           training_window_size, training_window_size + highest_ar_lag))
            return
        last_prediction_index = index_end
        dates = target.index[first_prediction_index:last_prediction_index+1]
        if verbose:
            print('Main loop parameters \n \
                  period_start=[{0}, loc_in_df:{1}] \n \
                  first_available_index=[{2}, loc_in_df:{3}] \n \
                  highest_ar_lag = {4}, training_window_size ={5}. \n \
                  last_prediction_index = [{6}, date: {7}]\n \
                  '.format(period_start, index_start, first_available_index,\
                           features.index[first_available_index], highest_ar_lag,\
                           training_window_size, last_prediction_index, period_end))

        rg = list(range(first_prediction_index, last_prediction_index+1))


        bar = ChargingBar('Processing', max=len(rg))
        for j in rg:
            bar.next()
            # range(a,b) runs from a to b-1.

            '''
            Main iteration loop works as follows:
            1.- defines the training window to work on
            2.- generates next day prediction
            '''

            # Preparing training window
            training_window_size, window_indices = training_window_handler[training_window](\
                            training_window_size,first_window_index, \
                              j, features, target)

            window_indices = np.array(window_indices)
            Y_train = target[window_indices]
            Y_ensemble.append(target[j])
            Y_index.append(j)

            window_indices -= horizon

            regular_input_train = copy.deepcopy(features.iloc[window_indices,:])
            regular_input_predict  = copy.deepcopy(features.iloc[j,:])
            #Preprocessing
            if verbose: print('Input shape prior to preprocessing',X_train.shape)

            regular_input_train, regular_input_predict, Y_train = \
            feature_preproccessing_handler[feature_preproc](regular_input_train, \
                                                            regular_input_predict, Y_train)

            if verbose: print('Input shape after preprocessing',X_train.shape)



            # generating data from predictors
            if verbose: print('Starting predictor Loop')
            for k, predictor_model in enumerate(self.predictor_list):

                X_train = []
                X_predict = []

                if has_ar[predictor_model]:
                    which_lags = ['AR{0}'.format(v) for i,v in enumerate(ar_model[predictor_model])]

                    if len(which_lags) > 0:

                        ar_pd = self.ar_lags[which_lags]

                        ar_train = copy.deepcopy(ar_pd.iloc[window_indices,:])
                        ar_predict = copy.deepcopy(ar_pd.iloc[j,:])

                        ar_train, ar_predict, Y_train = feature_preproccessing_handler[feature_preproc](ar_train, \
                                                                        ar_predict, Y_train)
                        X_train.append(ar_train)
                        X_predict.append(ar_predict)
                    else:
                        print('Warning! No AR data found for {0}'.format(predictor_model))

                if has_regularInput[predictor_model]:
                    X_train.append(regular_input_train)
                    X_predict.append(regular_input_predict)


                X_train = np.hstack(X_train)
                X_predict = np.hstack(X_predict)

                if verbose:
                    print('Predicting with {0}'.format(predictor_model))
                    print('regular ar {0} \n has_regularInput = {1}'.format(\
                     has_ar[predictor_model], has_regularInput[predictor_model]))

                predictors[predictor_model].predict(X_train, Y_train, X_predict)


            # generating data from ensemble_models
            if verbose: print('Starting Ensemble Loop')

            for k, ensemble_name in enumerate(self.ensemble_list):
                sub_dict = {}
                for l, predictor_name in enumerate(self.predictors_per_ensemble[ensemble_name]):
                    sub_dict[predictor_name] = predictors[predictor_name]

                if verbose: print('Fitting {0} with {1}'.format(ensemble_name, list(sub_dict.keys())))
                ensembles[ensemble_name].predict(sub_dict, X_train, Y_train,\
                                                 X_predict, Y_ensemble, Y_index)



            #After finishing all the predictions, we extract all data and export it
            # to a folder generated by the code

            self.first_prediction_index = first_prediction_index
            self.last_prediction_index = last_prediction_index

            models = self.predictor_list + self.ensemble_list
            predictors.update(ensembles)

        bar.finish()

        for k, model in enumerate(models):
            if verbose: print('Exporting data for {0}'.format(model))

            if k == 0:

                if predictors[model].store_predictions == True:
                    model_preds = np2dataframe(dates, model,np.array(predictors[model].predictions))

                if predictors[model].store_coefficients == True:
                    model_coefficients = np2dataframe(dates, model,np.array(np.vstack(predictors[model].coefficients)))
                if predictors[model].store_hyper_params == True:
                    model_hyper_params = np2dataframe(dates, model,np.array(np.vstack(predictors[model].hyper_params)))
            else:
                if predictors[model].store_predictions == True:
                    model_preds = pd.concat([model_preds,\
                                             np2dataframe(dates, model,np.array(predictors[model].predictions))], axis=1)
                if predictors[model].store_coefficients == True:
                    model_coefficients = pd.concat([model_coefficients,\
                                                   np2dataframe(dates, model,np.array(np.vstack(predictors[model].coefficients)))], axis = 1)
                if predictors[model].store_hyper_params == True:
                    model_hyper_params = pd.concat([model_hyper_params, \
                                                    np2dataframe(dates, model,np.array(np.vstack(predictors[model].hyper_params)))], axis = 1)

        model_preds.to_csv(id_dir+'/' + ID_PREDS)
        model_coefficients.to_csv(id_dir + '/' + ID_COEFF)
        model_hyper_params.to_csv(id_dir + '/' + ID_HPARAMS)

        exportfeatures = features.copy()
        exportfeatures.insert(0, TARGET, target)

        model_preds[TARGET] = target[first_prediction_index:last_prediction_index+1]
        model_preds.to_csv(id_dir+'/' + ID_PREDS)
        exportfeatures.to_csv(id_dir + '/' + ID_DATA)

        print('Data for {0} generated.'.format(id_))
        print('Succesfully completed prediction iteration loop.')





    def graph():
        return 0

    def save():
        return 0

class ARGO:
    '''
        ARGO net is an ensemble approach used in https://www.biorxiv.org/content/early/2018/06/14/344580
    '''

    def __init__(self, data=None, model_dict=None, output_name=None, training_window= 'static', \
                training_window_size=104, horizon=1, feature_preprocessing='zscore',\
                ar_model=52, load_folder = None, ar_column_filtering=True):
        '''
            Parameters
            ----------
                output_name : string, optional
                    Directory path where all results will be written to

                training_window : string, optional
                    Specify the training window type ('static' or 'expanding').
                    If static, the training dataset size stays constant with size
                    training_window_size specified by the user. If 'expanding' is
                    selected, the training dataset size keeps growing as data becomes
                    available (E.G. training dataset with 200 samples predicts for
                    time t becomes a dataset of 201 samples after adding data from time t
                    date to predict t+1).

                training_window_size : int, optional
                    The training dataset size.

                horizon : int, optional
                    Defines the number of samples gap between your expected target
                    to your most recent training sample.
                    e.g. if you're forecasting weekly points, a horizon 0
                    prediction would mean you are predicting for week t with your
                    most recent training data sample being t-1, horizon 1
                    would mean you're missing the t-1 sample from your dataset
                    and are actually predicting for t with t-2 being your most
                    recent sample. This description only makes sense for sequential data.

                feature_preprocessing : string, optional
                    specify the type of data pre-proccessing to input data. pre-proccessing
                    occurs iteratively within time-windows.

                ar_model : int or list of integers
                    Specify the autoreggresive model to use for the ARGONET implementation.
                    If int, ARGONET object generates the AR models using lags from t-1 to t-ar_model.
                    If a list of integers, ARGONET generates a custom set of autoreggressive lags based
                    on the list (e.g. if [1,3,10] is passed, the autoreggresive model will only contain
                    lags from t-1, t-3 and t-10).

                model_dict : dict, optional
                    Model dict gives the parameters the class works with when testing the ARGO model.
                    You construct a model giving the following parameters in a list:

                    fitting_function: (Functions available in argo_methods_.py)
                    pre_proccessing_function: (functions available in loading_functions_.py)
                    top_n_terms: (positive integer number)
                    correlation_threshold: (filters variables based on pearson correlation threshold)
                    dynamic_transformations: (transforms variables based on different operations)
                    mod_params: model params for the function you're using

                load_folder : boolean, optional
                    Specify a folder that already contains information on an ARGONET experiment.
                period_start : str, optional
                    String with date format YYYY-MM-DD.
                period_end : str, optional
                    String with date format YYYY-MM-DD.
                data : DATA object containing information from all regions of interest.
                    See DATA documentation (TODO add link)for a detailed explanation
                ar_column_filtering : Boolean, optional
                    Sets the column filter for the autoreggressive model on or off.
                    filter_ar_columns function checks for invalid data values (NaNs and
                    a list of values to ignore specified by the user)
        '''

        if model_dict is not None:
            self.model_dict = model_dict
        else:
            print('Need to specify a model dict')

        models = ['AR']
        for model, params in self.model_dict.items():
            models.append(model)

        self.models = models

        # Check if there's a name for the folder
        try:
            if output_name == None or not isinstance(output_name,str):
                now = datetime.datetime.now()
                self.output_name = 'ARGO_experiment'
            else:
                self.output_name = output_name
        except Exception as t:
            print(t)
            self.output_name = 'ARGO_experiment'
        # Checking optional values.
        try:
            self.training_window = training_window
            self.training_window_size = training_window_size
        except Exception as t:
            print('No training window information found, setting to default.')
            self.training_window = TRAINING_WINDOW_TYPE_DEFAULT
            self.training_window_size = TRAINING_WINDOW_SIZE_DEFAULT
        try:
            self.horizon = horizon
        except Exception as t:
            print('No horizon data found. Setting to default')
            self.horizon = 1
        try:
            self.feature_preproc = feature_preprocessing
        except Exception as t:
            print(t)
            print('No feature preprocessing found. Setting to default.')
            self.feature_preproc = 'zscore'
        try:
            if isinstance(ar_model,int):
                self.ar_model = list(range(1, ar_model+1))
            elif isinstance(ar_model, list) and all(isinstance(x, int) for x in ar_model):
                self.ar_model = ar_model
        except Exception as t:
            print(t)
            print('No AR_model specification found.')
            return
        try:
            self.data = data
        except Exception as t:
            print(t)
            print('No data object found')
        try:
            self.ar_column_filtering = ar_column_filtering
        except Exception as t:
            print('filter_ar_columns variable not initialized properly')

        print('Succesfully initialized experiment object.')


    def run(self, period_start, period_end, verbose=False, data=None, cont=False):

        if cont:
            try:
                self.overview_folder = self.output_name + '/' + OVERVIEW_FOLDER
                checkpoint = pickle.load(open(self.overview_folder+'/'+PICKLE_ARGONET_CHECKPOINT,'rb'))
                cp_phase = checkpoint[0]
                cp_id =  checkpoint[1]
                print('Experiment checkpoints successfully loaded. Experiment will continue at phase:{0} on id: {1}'.format(cp_phase, cp_id))
            except Exception as t:
                print('An error ocurred while trying to continue the experiment {0}'.format(self.output_name))
                print(t)
                return
        else:
            cp_phase = False
            cp_id = False

            output_name = gen_folder(self.output_name)
            if output_name != self.output_name:
                print('Warning! A folder called {0} already exists.'.format(self.output_name))
                self.output_name = output_name
            self.overview_folder = self.output_name + '/' + OVERVIEW_FOLDER
            gen_folder(self.overview_folder)

        if data is None:
            try:
                data = self.data
            except Exception as t:
                print('Could not find any data to work with. Exiting')
                print(t)
                return

        ids = data.id
        neighbor_ARGOdata = {}
        neighbor_ILIdata = {}
        neighbor_ARdata = None
        ar_model = self.ar_model
        which_lags = list(set(ar_model))

        model_dict = self.model_dict
        predictors = {}

        # ARGO MODEL PART
        # ---------------------------------------------------------------------
        print('Entering main computation loop. \n')


        # CHECK IF we need to continue
        if cont and cp_phase == 'ARGO':
            id_ind = ids.index(cp_id)
            cont_ids = ids[id_ind:]
        elif cont and cp_phase != 'ARGO':
            cont_ids = []
        else:
            cont_ids = ids

        for  i, id_ in enumerate(cont_ids):

            #CHECKPOINT
            pickle.dump(['ARGO', id_],open(self.overview_folder+'/'+PICKLE_ARGONET_CHECKPOINT,'wb'))

            print('Preparing data and performing a forecasting loop for {0}'.format(id_))
            id_dir = self.output_name+'/{0}'.format(id_)

            if not os.path.exists(id_dir):
                gen_folder(id_dir)

            # Initializing predictors.
            if verbose:
                print('Autoreggressive model for {0} is: {1} \n'.format(id_, ar_model))

            for j, model in enumerate(self.models):
                if model == 'AR':
                    predictors[model] = Predictor(model=model)
                else:
                    predictors[model] = Predictor(model=model, model_funct=model_dict[model][0], \
                                                model_preproc=model_dict[model][1], mod_params=model_dict[model][5])

            if verbose:
                print('Lags found for all models= \n {0} \n AR_model:{1}'.format(which_lags, ar_model))
            self.data.generate_ar_lags(which_id= id_, which_lags = which_lags)

            #Entering data generation loop per id
            highest_ar_lag = np.max(which_lags)
            target = self.data.target[id_]
            features = self.data.features[id_]
            horizon = self.horizon - 1
            feature_preproc = self.feature_preproc

            if verbose:
                print('highest_ar_lag:{0}\n period_start:{1} \n period_end:{2} \n \
                      horizon:{3} \n feature_preproc:{4}'.format(highest_ar_lag, period_start, period_end, horizon, feature_preproc))
                time.sleep(2)

            w = self.training_window
            s = self.training_window_size

            if isinstance(s, int) and s < features.shape[0]:
                training_window_size = s
            else:
                print('WARNING! Error with training window size. Using default')
                training_window_size = TRAINING_WINDOW_SIZE_DEFAULT

            try:
                if w in training_window_handler.keys():
                    training_window = w
                    training_window_size = s

            except KeyError:
                print('WARNING! Error defining the training window type. Using default')
                training_window = TRAINING_WINDOW_TYPE_DEFAULT

            if verbose:
                print('Training window parameters:\n size:{0}, type:{1} \n\n'.format(training_window_size, training_window))

            index_start = features.index.get_loc(period_start)
            index_end = features.index.get_loc(period_end)
            first_available_index =  highest_ar_lag + horizon + training_window_size

            if index_start >= first_available_index:
                first_prediction_index = index_start
                first_window_index = index_start - training_window_size
            else:
                print('Prediction start date falls within minimum training dataset period \n \
                      period_start=[{0}, loc_in_df:{1}] \n \
                      first_available_index=[{2}, loc_in_df:{3}] \n \
                      highest_ar_lag = {4}, training_window_size {5}. \n\n \
                      Prediction should start at index {1} but model minimum\
                       data points needed for training set at start is: {6}.\n \
                       Please update the training dataset or autoreggresive parameters.\
                      '.format(period_start, index_start, first_available_index,\
                               features.index[first_available_index], highest_ar_lag,\
                               training_window_size, training_window_size + highest_ar_lag))
                return

            last_prediction_index = index_end
            dates = target.index[first_prediction_index:last_prediction_index+1]

            if verbose:
                print('Main loop parameters \n \
                      period_start=[{0}, loc_in_df:{1}] \n \
                      first_available_index=[{2}, loc_in_df:{3}] \n \
                      highest_ar_lag = {4}, training_window_size ={5}. \n \
                      last_prediction_index = [{6}, date: {7}]\n \
                      '.format(period_start, index_start, first_available_index,\
                               features.index[first_available_index], highest_ar_lag,\
                               training_window_size, last_prediction_index, period_end))
            rg = list(range(first_prediction_index, last_prediction_index+1))

            bar = ChargingBar('Processing', max=len(rg))
            for j in rg:
                ar_model_labels = ['AR{0}'.format(v) for i,v in enumerate(ar_model)]
                bar.next()
                # range(a,b) runs from a to b-1.

                '''
                Main iteration loop works as follows:
                1.- defines the training window to work on
                2.- generates next day prediction
                '''

                # Preparing training window
                training_window_size, window_indices = training_window_handler[training_window](\
                                training_window_size,first_window_index, \
                                  j, features, target)


                if verbose:
                    print('Training window indices and size: \n \
                          training_window_size:{0} \n \
                          window_indices: {1}'.format(training_window_size, window_indices))
                    time.sleep(1)

                '''
                After retrieving the data points from our training_window function,
                we receive a set of indices indicating the position of the training samples
                in our features dataframe. Now we want to create our training data features and target.
                We first generate the training target vector with the indices we received and then
                we update the values of the indices by the desired horizon.
                '''
                window_indices = np.array(window_indices)
                Y_train = target.iloc[window_indices].values
                window_indices -= horizon


                regular_input_train = copy.deepcopy(features.iloc[window_indices,:])
                regular_input_predict = copy.deepcopy(features.iloc[j,:])
                column_limit = None
                threshold = None
                n_predictors = regular_input_train.shape[1]

                #Preprocessing
                #regular_input_train, regular_input_predict, Y_train = \
                #feature_preproccessing_handler[feature_preproc](regular_input_train, \
                #                                                regular_input_predict, Y_train)

                if verbose:
                    print('Training dataset indices, size: \n \
                          window_indices:{0} \n \
                          j:{1} \n \
                          Y_train.shape:{2}'.format(window_indices, j, Y_train.shape))

                for k, model in enumerate(self.models):
                    if model != 'AR':

                        if model_dict[model][4]:
                            external_variables_train, external_variables_predict = dynamic_transformations(\
                                copy.copy(regular_input_train), regular_input_predict, Y_train)
                        else:
                            external_variables_train = copy.copy(regular_input_train)
                            external_variables_predict = copy.copy(regular_input_predict)

                        if model_dict[model][2] or model_dict[model][3]:
                            external_variables_train = filter_by_similarity(copy.copy(external_variables_train),Y_train.ravel(),\
                                                                func=None,column_limit=model_dict[model][2], threshold=model_dict[model][3])
                    X_train = []
                    X_predict = []

                    if len(which_lags) > 0:

                        ar_pd = self.data.ar_lags[id_][ar_model_labels]
                        ar_train = copy.deepcopy(ar_pd.iloc[window_indices,:])

                        if self.ar_column_filtering:
                            ar_train=filter_ar_columns(ar_train)
                        ar_predict = copy.deepcopy(ar_pd.iloc[j,:])

                        #ar_train, ar_predict, Y_train = feature_preproccessing_handler[feature_preproc](ar_train, \
                        #                                                ar_predict, Y_train)
                        X_train.append(ar_train)
                        X_predict.append(ar_predict)
                    else:
                        print('Warning! No AR data found for {0}'.format(predictor_model))

                    if model != 'AR':

                        X_train.append(external_variables_train)
                        X_predict.append(external_variables_predict)

                    X_train = np.hstack(X_train)
                    X_predict = np.hstack(X_predict)

                    if verbose:
                        print(X_train, Y_train)

                    X_train, X_predict, counter, location = check_predict_features(X_train, X_predict)
                    if counter > 0:
                        print('Warning! {0} NaN(s) found within prediction sample features for index : {1} and model {3} at location(s) {2}\n\n.'.format(\
                                            counter, target.index[j], location, model))

                    predictors[model].predict(X_train, Y_train, X_predict)
            #After finishing all the predictions, we extract all data and export it
            # to a folder generated by the code
            bar.finish()
            self.first_prediction_index = first_prediction_index
            self.last_prediction_index = last_prediction_index

            for k, model in enumerate(self.models):

                if k == 0:

                    if predictors[model].store_predictions == True:
                        print(model,np.array(predictors[model].predictions))
                        model_preds = np2dataframe(dates, model,np.array(predictors[model].predictions))

                    if predictors[model].store_coefficients == True:
                        model_coefficients = np2dataframe(dates, model,np.array(np.vstack(predictors[model].coefficients)))
                    if predictors[model].store_hyper_params == True:
                        model_hyper_params = np2dataframe(dates, model,np.array(np.vstack(predictors[model].hyper_params)))
                else:
                    if predictors[model].store_predictions == True:
                        model_preds = pd.concat([model_preds,\
                                                 np2dataframe(dates, model,np.array(predictors[model].predictions))], axis=1)
                    if predictors[model].store_coefficients == True:
                        model_coefficients = pd.concat([model_coefficients,\
                                                       np2dataframe(dates, model,np.array(np.vstack(predictors[model].coefficients)))], axis = 1)
                    if predictors[model].store_hyper_params == True:
                        model_hyper_params = pd.concat([model_hyper_params, \
                                                        np2dataframe(dates, model,np.array(np.vstack(predictors[model].hyper_params)))], axis = 1)

            #model_preds.to_csv(id_dir+'/' + ID_PREDS)
            model_coefficients.to_csv(id_dir + '/' + ID_COEFF)
            model_hyper_params.to_csv(id_dir + '/' + ID_HPARAMS)

            exportfeatures = features.copy()
            exportfeatures.insert(0, TARGET, target)

            model_preds.insert(0,TARGET, target[first_prediction_index:last_prediction_index+1])
            model_preds.to_csv(id_dir+'/' + ID_PREDS)
            exportfeatures.to_csv(id_dir + '/' + ID_DATA)

            #neighbor_ARGOdata[id_] = copy.copy(model_preds['ARGO'])
            #neighbor_ILIdata[id_] = copy.copy(target)

            print('Data for {0} generated.'.format(id_))

class ARGO_lrmse:
    '''
        ARGO net is an ensemble approach used in https://www.biorxiv.org/content/early/2018/06/14/344580
    '''

    def __init__(self, data=None, model_dict=None, output_name=None, training_window= 'static', \
                training_window_size=104, horizon=1, feature_preprocessing='zscore',\
                ar_model=52, load_folder = None, ar_column_filtering=True, out_of_sample_rmse=False):
        '''
            Parameters
            ----------
                output_name : string, optional
                    Directory path where all results will be written to

                training_window : string, optional
                    Specify the training window type ('static' or 'expanding').
                    If static, the training dataset size stays constant with size
                    training_window_size specified by the user. If 'expanding' is
                    selected, the training dataset size keeps growing as data becomes
                    available (E.G. training dataset with 200 samples predicts for
                    time t becomes a dataset of 201 samples after adding data from time t
                    date to predict t+1).

                training_window_size : int, optional
                    The training dataset size.

                horizon : int, optional
                    Defines the number of samples gap between your expected target
                    to your most recent training sample.
                    e.g. if you're forecasting weekly points, a horizon 0
                    prediction would mean you are predicting for week t with your
                    most recent training data sample being t-1, horizon 1
                    would mean you're missing the t-1 sample from your dataset
                    and are actually predicting for t with t-2 being your most
                    recent sample. This description only makes sense for sequential data.

                feature_preprocessing : string, optional
                    specify the type of data pre-proccessing to input data. pre-proccessing
                    occurs iteratively within time-windows.

                ar_model : int or list of integers
                    Specify the autoreggresive model to use for the ARGONET implementation.
                    If int, ARGONET object generates the AR models using lags from t-1 to t-ar_model.
                    If a list of integers, ARGONET generates a custom set of autoreggressive lags based
                    on the list (e.g. if [1,3,10] is passed, the autoreggresive model will only contain
                    lags from t-1, t-3 and t-10).

                model_dict : dict, optional
                    Model dict gives the parameters the class works with when testing the ARGO model.
                    You construct a model giving the following parameters in a list:

                    fitting_function: (Functions available in argo_methods_.py)
                    pre_proccessing_function: (functions available in loading_functions_.py)
                    top_n_terms: (positive integer number)
                    correlation_threshold: (filters variables based on pearson correlation threshold)
                    dynamic_transformations: (transforms variables based on different operations)
                    mod_params: model params for the function you're using

                load_folder : boolean, optional
                    Specify a folder that already contains information on an ARGONET experiment.
                period_start : str, optional
                    String with date format YYYY-MM-DD.
                period_end : str, optional
                    String with date format YYYY-MM-DD.
                data : DATA object containing information from all regions of interest.
                    See DATA documentation (TODO add link)for a detailed explanation
                ar_column_filtering : Boolean, optional
                    Sets the column filter for the autoreggressive model on or off.
                    filter_ar_columns function checks for invalid data values (NaNs and
                    a list of values to ignore specified by the user)
                out_of_sample_rmse: Boolean, optional (Default is False)
                    If set to true, model selection is made based solely on out of sample metric scores. This option is only
                    available for models fitted using only static time windows
        '''

        if model_dict is not None:
            self.model_dict = model_dict
        else:
            print('Need to specify a model dict')

        models = []
        for model, params in self.model_dict.items():
            models.append(model)

        self.models = models

        # Check if there's a name for the folder
        try:
            if output_name == None or not isinstance(output_name,str):
                now = datetime.datetime.now()
                self.output_name = 'ARGO_experiment'
            else:
                self.output_name = output_name
        except Exception as t:
            print(t)
            self.output_name = 'ARGO_experiment'
        # Checking optional values.
        try:
            self.training_window = training_window
            self.training_window_size = training_window_size
        except Exception as t:
            print('No training window information found, setting to default.')
            self.training_window = TRAINING_WINDOW_TYPE_DEFAULT
            self.training_window_size = TRAINING_WINDOW_SIZE_DEFAULT
        try:
            self.horizon = horizon
        except Exception as t:
            print('No horizon data found. Setting to default')
            self.horizon = 1
        try:
            self.feature_preproc = feature_preprocessing
        except Exception as t:
            print(t)
            print('No feature preprocessing found. Setting to default.')
            self.feature_preproc = 'zscore'
        try:
            if isinstance(ar_model,int):
                self.ar_model = list(range(1, ar_model+1))
            elif isinstance(ar_model, list) and all(isinstance(x, int) for x in ar_model):
                self.ar_model = ar_model
        except Exception as t:
            print(t)
            print('No AR_model specification found.')
            return
        try:
            self.data = data
        except Exception as t:
            print(t)
            print('No data object found')
        try:
            self.ar_column_filtering = ar_column_filtering
        except Exception as t:
            print('filter_ar_columns variable not initialized properly')


        self.out_of_sample_rmse = out_of_sample_rmse

        print('Succesfully initialized experiment object.')


    def run(self, period_start, period_end, verbose=False, data=None, cont=False):

        if cont:
            try:
                self.overview_folder = self.output_name + '/' + OVERVIEW_FOLDER
                checkpoint = pickle.load(open(self.overview_folder+'/'+PICKLE_ARGONET_CHECKPOINT,'rb'))
                cp_phase = checkpoint[0]
                cp_id =  checkpoint[1]
                print('Experiment checkpoints successfully loaded. Experiment will continue at phase:{0} on id: {1}'.format(cp_phase, cp_id))
            except Exception as t:
                print('An error ocurred while trying to continue the experiment {0}'.format(self.output_name))
                print(t)
                return
        else:
            cp_phase = False
            cp_id = False

            output_name = gen_folder(self.output_name)
            if output_name != self.output_name:
                print('Warning! A folder called {0} already exists.'.format(self.output_name))
                self.output_name = output_name
            self.overview_folder = self.output_name + '/' + OVERVIEW_FOLDER
            gen_folder(self.overview_folder)

        if data is None:
            try:
                data = self.data
            except Exception as t:
                print('Could not find any data to work with. Exiting')
                print(t)
                return

        ids = data.id
        neighbor_ARGOdata = {}
        neighbor_ILIdata = {}
        neighbor_ARdata = None
        ar_model = self.ar_model
        which_lags = list(set(ar_model))

        model_dict = self.model_dict
        predictors = {}
        models = copy.copy(self.models)

        # ARGO MODEL PART
        # ---------------------------------------------------------------------
        print('Entering main computation loop. \n')


        # CHECK IF we need to continue
        if cont and cp_phase == 'ARGO':
            id_ind = ids.index(cp_id)
            cont_ids = ids[id_ind:]
        elif cont and cp_phase != 'ARGO':
            cont_ids = []
        else:
            cont_ids = ids

        for  i, id_ in enumerate(cont_ids):

            #CHECKPOINT
            pickle.dump(['ARGO', id_],open(self.overview_folder+'/'+PICKLE_ARGONET_CHECKPOINT,'wb'))

            print('Preparing data and performing a forecasting loop for {0}'.format(id_))
            id_dir = self.output_name+'/{0}'.format(id_)

            if not os.path.exists(id_dir):
                gen_folder(id_dir)

            # Initializing predictors.
            if verbose:
                print('Autoreggressive model for {0} is: {1} \n'.format(id_, ar_model))


            coefficient_name_dict = {}

            if 'ARGO_lrmse' in models:
                models.remove('ARGO_lrmse')
                ARGO_lrmse = True
            else:
                ARGO_lrmse = False

            for j, model in enumerate(models):
                    predictors[model] = Predictor(model=model, model_funct=model_dict[model][0], \
                                                model_preproc=model_dict[model][1], mod_params=model_dict[model][5])

            if  ARGO_lrmse:
                predictors['ARGO_lrmse'] = Predictor(model='ARGO_lrmse', model_funct=True, \
                                            model_preproc=True, mod_params=True)
                winner_history = []

            if verbose:
                print('Lags found for all models= \n {0} \n AR_model:{1}'.format(which_lags, ar_model))
            self.data.generate_ar_lags(which_id= id_, which_lags = which_lags)


            #Entering data generation loop per id
            highest_ar_lag = np.max(which_lags)
            target = self.data.target[id_]
            features = self.data.features[id_]
            horizon = self.horizon - 1
            feature_preproc = self.feature_preproc

            if verbose:
                print('highest_ar_lag:{0}\n period_start:{1} \n period_end:{2} \n \
                      horizon:{3} \n feature_preproc:{4}'.format(highest_ar_lag, period_start, period_end, horizon, feature_preproc))
                time.sleep(2)

            w = self.training_window
            s = self.training_window_size

            if isinstance(s, int) and s < features.shape[0]:
                training_window_size = s
            else:
                print('WARNING! Error with training window size. Using default')
                training_window_size = TRAINING_WINDOW_SIZE_DEFAULT

            try:
                if w in training_window_handler.keys():
                    training_window = w
                    training_window_size = s

            except KeyError:
                print('WARNING! Error defining the training window type. Using default')
                training_window = TRAINING_WINDOW_TYPE_DEFAULT

            if verbose:
                print('Training window parameters:\n size:{0}, type:{1} \n\n'.format(training_window_size, training_window))

            index_start = features.index.get_loc(period_start)
            index_end = features.index.get_loc(period_end)
            first_available_index =  highest_ar_lag + horizon + training_window_size

            if index_start >= first_available_index:
                first_prediction_index = index_start
                first_window_index = index_start - training_window_size
            else:
                print('Prediction start date falls within minimum training dataset period \n \
                      period_start=[{0}, loc_in_df:{1}] \n \
                      first_available_index=[{2}, loc_in_df:{3}] \n \
                      highest_ar_lag = {4}, training_window_size {5}. \n\n \
                      Prediction should start at index {1} but model minimum\
                       data points needed for training set at start is: {6}.\n \
                       Please update the training dataset or autoreggresive parameters.\
                      '.format(period_start, index_start, first_available_index,\
                               features.index[first_available_index], highest_ar_lag,\
                               training_window_size, training_window_size + highest_ar_lag))
                return

            last_prediction_index = index_end
            dates = target.index[first_prediction_index:last_prediction_index+1]

            if verbose:
                print('Main loop parameters \n \
                      period_start=[{0}, loc_in_df:{1}] \n \
                      first_available_index=[{2}, loc_in_df:{3}] \n \
                      highest_ar_lag = {4}, training_window_size ={5}. \n \
                      last_prediction_index = [{6}, date: {7}]\n \
                      '.format(period_start, index_start, first_available_index,\
                               features.index[first_available_index], highest_ar_lag,\
                               training_window_size, last_prediction_index, period_end))
            rg = list(range(first_prediction_index, last_prediction_index+1))

            bar = ChargingBar('Processing', max=len(rg))
            for j in rg:
                ar_model_labels = ['AR{0}'.format(v) for i,v in enumerate(ar_model)]
                bar.next()
                # range(a,b) runs from a to b-1.

                '''
                Main iteration loop works as follows:
                1.- defines the training window to work on
                2.- generates next day prediction
                '''

                # Preparing training window
                training_window_size, window_indices = training_window_handler[training_window](\
                                training_window_size,first_window_index, \
                                  j, features, target)


                if verbose:
                    print('Training window indices and size: \n \
                          training_window_size:{0} \n \
                          window_indices: {1}'.format(training_window_size, window_indices))
                    time.sleep(1)

                '''
                After retrieving the data points from our training_window function,
                we receive a set of indices indicating the position of the training samples
                in our features dataframe. Now we want to create our training data features and target.
                We first generate the training target vector with the indices we received and then
                we update the values of the indices by the desired horizon.
                '''

                min_ind = np.min(window_indices)
                test_indices = list(range(highest_ar_lag+horizon, min_ind))


                window_indices = np.array(window_indices)
                test_indices = np.array(test_indices)

                Y_train = target.iloc[window_indices].values
                Y_test = target.iloc[test_indices].values


                window_indices -= horizon
                test_indices -= horizon


                regular_input_train = copy.deepcopy(features.iloc[window_indices,:])
                regular_input_predict = copy.deepcopy(features.iloc[j,:])
                column_limit = None
                threshold = None
                n_predictors = regular_input_train.shape[1]

                #Preprocessing
                #regular_input_train, regular_input_predict, Y_train = \
                #feature_preproccessing_handler[feature_preproc](regular_input_train, \
                #                                                regular_input_predict, Y_train)
                if verbose:
                    print('Training dataset indices, size: \n \
                          window_indices:{0} \n \
                          j:{1} \n \
                          Y_train.shape:{2}'.format(window_indices, j, Y_train.shape))

                winner = ''
                least_rmse = float('inf')
                for k, model in enumerate(models):
                    if model != 'AR':

                        if model_dict[model][4]:
                            external_variables_train, external_variables_predict = dynamic_transformations(\
                                copy.copy(regular_input_train), regular_input_predict, Y_train)
                        else:
                            external_variables_train = copy.copy(regular_input_train)
                            external_variables_predict = copy.copy(regular_input_predict)

                        if model_dict[model][2] or model_dict[model][3]:
                            external_variables_train = filter_by_similarity(copy.copy(external_variables_train),Y_train.ravel(),\
                                                                func=None,column_limit=model_dict[model][2], threshold=model_dict[model][3])
                    X_train = []
                    X_predict = []
                    X_test = []

                    if len(which_lags) > 0:

                        ar_pd = self.data.ar_lags[id_][ar_model_labels]
                        ar_train = copy.deepcopy(ar_pd.iloc[window_indices,:])

                        if self.ar_column_filtering:
                            ar_train=filter_ar_columns(ar_train)
                        ar_predict = copy.deepcopy(ar_pd.iloc[j,:])

                        #ar_train, ar_predict, Y_train = feature_preproccessing_handler[feature_preproc](ar_train, \
                        #                                                ar_predict, Y_train)
                        X_train.append(ar_train)
                        X_predict.append(ar_predict)

                        X_test.append(copy.deepcopy(ar_pd.iloc[test_indices,:]))
                    else:
                        print('Warning! No AR data found for {0}'.format(predictor_model))

                    if model != 'AR':

                        X_test.append(copy.deepcopy(features.iloc[test_indices,:]))
                        X_train.append(external_variables_train)
                        X_predict.append(external_variables_predict)
                    if model == 'GO':
                        X_test = copy.deepcopy(features.iloc[test_indices,:])
                        X_train = [external_variables_train]
                        X_predict = [external_variables_predict]



                    if self.out_of_sample_rmse:
                        X_test = np.hstack(X_test)
                    else:
                        X_test = None

                    if j == first_prediction_index:
                        coefficient_name_dict[model] = list(pd.concat(X_train, axis=1))

                    X_train = np.hstack(X_train)
                    X_predict = np.hstack(X_predict)



                    if verbose:
                        print(X_train, Y_train)

                    X_train, X_predict, counter, location = check_predict_features(X_train, X_predict)
                    X_train, X_predict, Y_train = feature_preproccessing_handler[feature_preproc](X_train, X_predict, Y_train)

                    if counter > 0:
                        print('Warning! {0} NaN(s) found within prediction sample features for index : {1} and model {3} at location(s) {2}\n\n.'.format(\
                                            counter, target.index[j], location, model))


                    predictors[model].fit(X_train, Y_train, X_predict, X_test, Y_test)
                    predictors[model].predict2(X_train, Y_train, X_predict)
                    if ARGO_lrmse:
                        if model not in  ['AR', 'GO']:

                            if self.out_of_sample_rmse:
                                val = predictors[model].out_of_sample_metric[-1]
                            else:
                                val = predictors[model].insample_metric[-1]

                            if val < least_rmse:

                                least_rmse = val
                                winner = model


                if ARGO_lrmse:
                    winner_history.append(winner)
                    predictors['ARGO_lrmse'].predictions.append(predictors[winner].predictions[-1])
                    predictors['ARGO_lrmse'].coefficients.append(predictors[winner].coefficients[-1])
                    predictors['ARGO_lrmse'].hyper_params.append(predictors[winner].hyper_params[-1])



            #After finishing all the predictions, we extract all data and export it
            # to a folder generated by the code
            bar.finish()
            self.first_prediction_index = first_prediction_index
            self.last_prediction_index = last_prediction_index


            if ARGO_lrmse:
                models.append('ARGO_lrmse')


            for model in models:
                model_coefficients = np2dataframe(dates, model,np.array(np.vstack(predictors[model].coefficients)))
                if model != 'ARGO_lrmse':
                    model_coefficients.columns = coefficient_name_dict[model]
                model_coefficients.to_csv(id_dir + '/' + '{0}_'.format(model)+ ID_COEFF )

            for k, model in enumerate(models):

                if k == 0:

                    if predictors[model].store_predictions == True:
                        #print(model,np.array(predictors[model].predictions))
                        model_preds = np2dataframe(dates, model,np.array(predictors[model].predictions))
                    if predictors[model].store_coefficients == True:
                        model_coefficients = np2dataframe(dates, model,np.array(np.vstack(predictors[model].coefficients)))
                    if predictors[model].store_hyper_params == True:
                        model_hyper_params = np2dataframe(dates, model,np.array(np.vstack(predictors[model].hyper_params)))
                else:
                    if predictors[model].store_predictions == True:
                        model_preds = pd.concat([model_preds,\
                                                 np2dataframe(dates, model,np.array(predictors[model].predictions))], axis=1)
                    if predictors[model].store_coefficients == True:
                        model_coefficients = pd.concat([model_coefficients,\
                                                       np2dataframe(dates, model,np.array(np.vstack(predictors[model].coefficients)))], axis = 1)
                    if predictors[model].store_hyper_params == True:
                        model_hyper_params = pd.concat([model_hyper_params, \
                                                        np2dataframe(dates, model,np.array(np.vstack(predictors[model].hyper_params)))], axis = 1)

            #model_preds.to_csv(id_dir+'/' + ID_PREDS)
            model_hyper_params.to_csv(id_dir + '/' + ID_HPARAMS)

            exportfeatures = features.copy()
            exportfeatures.insert(0, TARGET, target)

            model_preds.insert(0,TARGET, target[first_prediction_index:last_prediction_index+1])
            model_preds.to_csv(id_dir+'/' + ID_PREDS)
            exportfeatures.to_csv(id_dir + '/' + ID_DATA)

            #neighbor_ARGOdata[id_] = copy.copy(model_preds['ARGO'])
            #neighbor_ILIdata[id_] = copy.copy(target)

            print('Data for {0} generated.'.format(id_))


class VotingSelector():

    def __init__(self,main_folder='', id_='', file_name='', metric='RMSE', k=3, \
                 models=['AR52GO', 'AR52'], window=0, sdate='2014-01-05', edate='2016-12-25'):
        self.main_folder=main_folder
        self.id_ = id_
        self.file_name = file_name
        self.metric= metric
        self.k = k
        self.window = window
        self.metric = metric
        self.sdate = sdate
        self.edate = edate
        self.models = models

    def run(self):

        df = pd.read_csv('{0}/{1}/{2}'.format(self.main_folder, self.id_, self.file_name), index_col=[0])
        k = self.k
        metric = self.metric
        dates = df[self.sdate:self.edate].index.values
        models = self.models
        window = self.window
        column_name = 'VS_{1}_k{0}'.format(k, metric)
        df[column_name] = 0
        for date in dates:

            i = df.index.get_loc(date)
            sub_df = df.iloc[i-window-k:i]
            winner = voting_selector_handler[metric](sub_df, window, k, models, TARGET)
            df[column_name].iloc[i] = df[winner].iloc[i]


        df.to_csv('{0}/{1}/{2}'.format(self.main_folder, self.id_, self.file_name))



    #Makes a
class NET:
    '''
        ARGO net is an ensemble approach used in https://www.biorxiv.org/content/early/2018/06/14/344580
    '''

    def __init__(self, data=None, output_name=None, training_window= 'static', \
                training_window_size=104, horizon=1, feature_preprocessing='zscore',\
                ar_model=52, neighbor_ar_model=3, ARGONET_window=3, load_folder = None,\
                ar_column_filtering=True, filter_externalvars_by_similarity=True, \
                argonet_filter_columns=True, \
                 only_ar_models=False, filter_neighbors_by_similarity=True, least_error_syncronicity=True,
                 per_location_neighbors=None):
        '''
            Parameters
            ----------

                output_name : string, optional
                    Directory path where all results will be written to

                training_window : string, optional
                    Specify the training window type ('static' or 'expanding').
                    If static, the training dataset size stays constant with size
                    training_window_size specified by the user. If 'expanding' is
                    selected, the training dataset size keeps growing as data becomes
                    available (E.G. training dataset with 200 samples predicts for
                    time t becomes a dataset of 201 samples after adding data from time t
                    date to predict t+1).

                training_window_size : int, optional
                    The training dataset size.

                horizon : int, optional
                    Defines the number of samples gap between your expected target
                    to your most recent training sample.
                    e.g. if you're forecasting weekly points, a horizon 0
                    prediction would mean you are predicting for week t with your
                    most recent training data sample being t-1, horizon 1
                    would mean you're missing the t-1 sample from your dataset
                    and are actually predicting for t with t-2 being your most
                    recent sample. This description only makes sense for sequential data.

                feature_preprocessing : string, optional
                    specify the type of data pre-proccessing to input data. pre-proccessing
                    occurs iteratively within time-windows.

                ar_model : int or list of integers
                    Specify the autoreggresive model to use for the ARGONET implementation.
                    If int, ARGONET object generates the AR models using lags from t-1 to t-ar_model.
                    If a list of integers, ARGONET generates a custom set of autoreggressive lags based
                    on the list (e.g. if [1,3,10] is passed, the autoreggresive model will only contain
                    lags from t-1, t-3 and t-10).

                neighbor_ar_model : int or list of integers
                    Specify the autoreggressive model to use for the neighbors in ARGONET implementation.

                load_folder : boolean, optional
                    Specify a folder that already contains information on an ARGONET experiment.
                period_start : str, optional
                    String with date format YYYY-MM-DD.
                period_end : str, optional
                    String with date format YYYY-MM-DD.
                data : DATA object containing information from all regions of interest.
                    See DATA documentation (TODO add link)for a detailed explanation
                ar_column_filtering : Boolean, optional
                    Sets the column filter for the autoreggressive model on or off.
                    filter_ar_columns function checks for invalid data values (NaNs and
                    a list of values to ignore specified by the user)
                argonet_filter_columns : Boolean, optional
                    Sets the column filter for the NET model input
                only_ar_models : boolean, optional
                    Ignores external variable input (google searches, etc) and only uses
                    autoreggressive data to create the NET and ARGONET model
                least_error_syncronicity : Boolean, optional
                    If set to true, then the autoreggressive lag at t_0 for the NET model is chosen based
                    on a vote between the ARGO and AR implementation (Whichever model between ARGO and AR
                    that has the least error in the past three weeks t-1, t-2, t-3 prior to t will be decided as the
                    t0 entry for the NET model).
                per_location_neighbors : Dict, optional (default is None)
                    A dictionary that indicates the only neighbors that will interact with the current location using NET.
                    E.G.  {'USA':['CAN', 'MEX'], 'ARG':['MEX', 'CHI']}
        '''

        if only_ar_models:
            self.models = ['AR']
        else:
            self.models=['ARGO', 'AR']
        # Check if there's a name for the folder
        try:
            if output_name == None or not isinstance(output_name,str):
                now = datetime.datetime.now()
                self.output_name = 'NET_experiment'.format(now.year, now.month, now.day)
            else:
                self.output_name = output_name
        except Exception as t:
            print(t)
            self.output_name = 'NET_experiment'.format(now.year, now.month, now.day)
        # Checking optional values.
        try:
            self.training_window = training_window
            self.training_window_size = training_window_size
        except Exception as t:
            print('No training window information found, setting to default.')
            self.training_window = TRAINING_WINDOW_TYPE_DEFAULT
            self.training_window_size = TRAINING_WINDOW_SIZE_DEFAULT
        try:
            self.horizon = horizon
        except Exception as t:
            print('No horizon data found. Setting to default')
            self.horizon = 1
        try:
            self.feature_preproc = feature_preprocessing
        except Exception as t:
            print(t)
            print('No feature preprocessing found. Setting to default.')
            self.feature_preproc = 'zscore'
        try:
            if isinstance(ar_model,int):
                self.ar_model = list(range(1, ar_model+1))
            elif isinstance(ar_model, list) and all(isinstance(x, int) for x in ar_model):
                self.ar_model = ar_model
        except Exception as t:
            print(t)
            print('No AR_model specification found.')
            return
        try:
            if isinstance(neighbor_ar_model,int):
                self.neighbor_ar_model = list(range(1,neighbor_ar_model+1))
            elif isinstance(neighbor_ar_model, list) and all(isinstance(x, int) for x in neighbor_ar_model):
                self.neighbor_ar_model = neighbor_ar_model
        except Exception as t:
            print(t)
            print('No AR_model specification found for neighbors.')
            return
        try:
            self.data = data
        except Exception as t:
            print(t)
            print('No data object found')
        try:
            if isinstance(ARGONET_window,int):
                self.ARGONET_window = ARGONET_window
        except Exception as t:
            print(t)
            print('ARGONET window not initialized properly.')
            return()
        try:
            self.ar_column_filtering = ar_column_filtering
        except Exception as t:
            print('filter_ar_columns variable not initialized properly')
        try:
            self.filter_externalvars_by_similarity = filter_externalvars_by_similarity
        except Exception as t:
            print('filter_externalvars_by_similarity variable not initialized properly')
        try:
            self.filter_neighbors_by_similarity = filter_neighbors_by_similarity
        except Exception as t:
            print('filter_externalvars_by_similarity variable not initialized properly')
        try:
            self.argonet_filter_columns = argonet_filter_columns
        except Exception as t:
            print('argonet_filter_columns variable not initialized properly')
        try:
            self.only_ar_models = only_ar_models
        except Exception as t:
            print('argonet_filter_columns variable not initialized properly')
        try:
            self.least_error_syncronicity = least_error_syncronicity
        except Exception as t:
            print('argonet_filter_columns variable not initialized properly')
        try:
            self.per_location_neighbors = per_location_neighbors
        except Exception as t:
            print('per_location_neighbors variable not initialized properly')





        print('Succesfully initialized experiment object.')


    def run(self, period_start, period_end, net_model_dict=None, id_net=None, verbose=False, data=None, cont=False,\
             neighbor_treshold = .60 , externalvars_threshold=.75, column_limit=None):

        if net_model_dict is None:
            dict_net_model = {
                'NET': [lasso, preproc_rmv_values, None, None, None]
            }
        else:
            dict_net_model = copy.copy(net_model_dict)
        if cont:
            try:
                self.overview_folder = self.output_name + '/' + OVERVIEW_FOLDER
                checkpoint = pickle.load(open(self.overview_folder+'/'+PICKLE_ARGONET_CHECKPOINT,'rb'))
                cp_phase = checkpoint[0]
                cp_id =  checkpoint[1]
                print('Experiment checkpoints successfully loaded. Experiment will continue at phase:{0} on id: {1}'.format(cp_phase, cp_id))
            except Exception as t:
                print('An error ocurred while trying to continue the experiment {0}'.format(self.output_name))
                print(t)
                return
        else:
            cp_phase = False
            cp_id = False

            output_name = gen_folder(self.output_name)
            if output_name != self.output_name:
                print('Warning! A folder called {0} already exists.'.format(self.output_name))
                self.output_name = output_name
            self.overview_folder = self.output_name + '/' + OVERVIEW_FOLDER
            gen_folder(self.overview_folder)

        if data is None:
            try:
                data = self.data
            except Exception as t:
                print('Could not find any data to work with. Exiting')
                print(t)
                return

        ids = data.id
        neighbor_ARGOdata = {}
        neighbor_ILIdata = {}
        neighbor_ARdata = None
        ar_model = self.ar_model
        neighbor_ar_model = self.neighbor_ar_model
        neighbor_ar_model_labels = ['AR{0}'.format(v) for i,v in enumerate(neighbor_ar_model)]
        which_lags = list(set(ar_model+neighbor_ar_model))

        # ARGO MODEL PART
        # ---------------------------------------------------------------------
        print('Entering main computation loop. \n')


        # CHECK IF we need to continue
        if cont and cp_phase == 'ARGO':
            id_ind = ids.index(cp_id)
            cont_ids = ids[id_ind:]
        elif cont and cp_phase != 'ARGO':
            cont_ids = []
        else:
            cont_ids = ids

        for  i, id_ in enumerate(cont_ids):

            #CHECKPOINT
            pickle.dump(['ARGO', id_],open(self.overview_folder+'/'+PICKLE_ARGONET_CHECKPOINT,'wb'))

            print('Preparing data and performing a forecasting loop for {0}'.format(id_))
            id_dir = self.output_name+'/{0}'.format(id_)

            if not os.path.exists(id_dir):
                gen_folder(id_dir)

            # Initializing predictors.
            predictors = {}

            if verbose:
                print('Autoreggressive model for {0} is: {1} \n'.format(id_, ar_model))

            for j, model in enumerate(self.models):
                predictors[model] = Predictor(model=model)

            if verbose:
                print('Lags found for all models= \n {0} \n AR_model:{1}'.format(which_lags, ar_model))
            self.data.generate_ar_lags(which_id= id_, which_lags = which_lags)

            #Entering data generation loop per id
            highest_ar_lag = np.max(which_lags)
            target = self.data.target[id_]
            features = self.data.features[id_]
            horizon = self.horizon - 1
            feature_preproc = self.feature_preproc

            if verbose:
                print('highest_ar_lag:{0}\n period_start:{1} \n period_end:{2} \n \
                      horizon:{3} \n feature_preproc:{4}'.format(highest_ar_lag, period_start, period_end, horizon, feature_preproc))
                time.sleep(2)

            w = self.training_window
            s = self.training_window_size

            if isinstance(s, int) and s < features.shape[0]:
                training_window_size = s
            else:
                print('WARNING! Error with training window size. Using default')
                training_window_size = TRAINING_WINDOW_SIZE_DEFAULT

            try:
                if w in training_window_handler.keys():
                    training_window = w
                    training_window_size = s

            except KeyError:
                print('WARNING! Error defining the training window type. Using default')
                training_window = TRAINING_WINDOW_TYPE_DEFAULT

            if verbose:
                print('Training window parameters:\n size:{0}, type:{1} \n\n'.format(training_window_size, training_window))

            index_start = features.index.get_loc(period_start)
            index_end = features.index.get_loc(period_end)
            first_available_index =  highest_ar_lag + horizon + training_window_size

            if index_start >= first_available_index:
                first_prediction_index = index_start
                first_window_index = index_start - training_window_size
            else:
                print('Prediction start date falls within minimum training dataset period \n \
                      period_start=[{0}, loc_in_df:{1}] \n \
                      first_available_index=[{2}, loc_in_df:{3}] \n \
                      highest_ar_lag = {4}, training_window_size {5}. \n\n \
                      Prediction should start at index {1} but model minimum\
                       data points needed for training set at start is: {6}.\n \
                       Please update the training dataset or autoreggresive parameters.\
                      '.format(period_start, index_start, first_available_index,\
                               features.index[first_available_index], highest_ar_lag,\
                               training_window_size, training_window_size + highest_ar_lag))
                return

            last_prediction_index = index_end
            dates = target.index[first_prediction_index:last_prediction_index+1]

            if verbose:
                print('Main loop parameters \n \
                      period_start=[{0}, loc_in_df:{1}] \n \
                      first_available_index=[{2}, loc_in_df:{3}] \n \
                      highest_ar_lag = {4}, training_window_size ={5}. \n \
                      last_prediction_index = [{6}, date: {7}]\n \
                      '.format(period_start, index_start, first_available_index,\
                               features.index[first_available_index], highest_ar_lag,\
                               training_window_size, last_prediction_index, period_end))
            rg = list(range(first_prediction_index, last_prediction_index+1))

            """
            bar = ChargingBar('Processing', max=len(rg))
            for j in rg:
                ar_model_labels = ['AR{0}'.format(v) for i,v in enumerate(ar_model)]
                bar.next()
                # range(a,b) runs from a to b-1.

                '''
                Main iteration loop works as follows:
                1.- defines the training window to work on
                2.- generates next day prediction
                '''

                # Preparing training window
                training_window_size, window_indices = training_window_handler[training_window](\
                                training_window_size,first_window_index, \
                                  j, features, target)


                if verbose:
                    print('Training window indices and size: \n \
                          training_window_size:{0} \n \
                          window_indices: {1}'.format(training_window_size, window_indices))
                    time.sleep(1)

                '''
                After retrieving the data points from our training_window function,
                we receive a set of indices indicating the position of the training samples
                in our features dataframe. Now we want to create our training data features and target.
                We first generate the training target vector with the indices we received and then
                we update the values of the indices by the desired horizon.
                '''
                window_indices = np.array(window_indices)
                Y_train = target.iloc[window_indices].values
                window_indices -= horizon


                regular_input_train = copy.deepcopy(features.iloc[window_indices,:])
                regular_input_predict  = copy.deepcopy(features.iloc[j,:])

                if self.filter_externalvars_by_similarity:
                    regular_input_train = filter_by_similarity(regular_input_train,Y_train.ravel(), func=None, threshold=externalvars_threshold, column_limit=column_limit)

                #Preprocessing

                #regular_input_train, regular_input_predict, Y_train = \
                #feature_preproccessing_handler[feature_preproc](regular_input_train, \
                #                                                regular_input_predict, Y_train)

                if verbose:
                    print('Training dataset indices, size: \n \
                          window_indices:{0} \n \
                          j:{1} \n \
                          Y_train.shape:{2}'.format(window_indices, j, Y_train.shape))

                for k, model in enumerate(self.models):

                    X_train = []
                    X_predict = []

                    if len(which_lags) > 0:

                        ar_pd = self.data.ar_lags[id_][ar_model_labels]
                        ar_train = copy.deepcopy(ar_pd.iloc[window_indices,:])

                        if self.ar_column_filtering:
                            ar_train=filter_ar_columns(ar_train)
                        ar_predict = copy.deepcopy(ar_pd.iloc[j,:])

                        #ar_train, ar_predict, Y_train = feature_preproccessing_handler[feature_preproc](ar_train, \
                        #                                                ar_predict, Y_train)
                        X_train.append(ar_train)
                        X_predict.append(ar_predict)
                    else:
                        print('Warning! No AR data found for {0}'.format(predictor_model))

                    if model == 'ARGO':

                        X_train.append(regular_input_train)
                        X_predict.append(regular_input_predict)
                        #X_train = [regular_input_train]
                        #X_predict = [regular_input_predict]

                    X_train = np.hstack(X_train)
                    X_predict = np.hstack(X_predict)

                    if verbose:
                        print(X_train, Y_train)
                    X_train, X_predict, counter, location = check_predict_features(X_train, X_predict)
                    if counter > 0:
                        print('Warning! {0} NaN(s) found within prediction sample features for index : {1} and model {3} at location(s) {2}\n\n.'.format(\
                                            counter, target.index[j], location, model))

                    predictors[model].predict(X_train, Y_train, X_predict)


            #After finishing all the predictions, we extract all data and export it
            # to a folder generated by the code
            bar.finish()
            self.first_prediction_index = first_prediction_index
            self.last_prediction_index = last_prediction_index

            for k, model in enumerate(self.models):

                if k == 0:

                    if predictors[model].store_predictions == True:
                        model_preds = np2dataframe(dates, model,np.array(predictors[model].predictions))

                    if predictors[model].store_coefficients == True:
                        model_coefficients = np2dataframe(dates, model,np.array(np.vstack(predictors[model].coefficients)))
                    if predictors[model].store_hyper_params == True:
                        model_hyper_params = np2dataframe(dates, model,np.array(np.vstack(predictors[model].hyper_params)))
                else:
                    if predictors[model].store_predictions == True:
                        model_preds = pd.concat([model_preds,\
                                                 np2dataframe(dates, model,np.array(predictors[model].predictions))], axis=1)
                    if predictors[model].store_coefficients == True:
                        model_coefficients = pd.concat([model_coefficients,\
                                                       np2dataframe(dates, model,np.array(np.vstack(predictors[model].coefficients)))], axis = 1)
                    if predictors[model].store_hyper_params == True:
                        model_hyper_params = pd.concat([model_hyper_params, \
                                                        np2dataframe(dates, model,np.array(np.vstack(predictors[model].hyper_params)))], axis = 1)

            #model_preds.to_csv(id_dir+'/' + ID_PREDS)
            model_coefficients.to_csv(id_dir + '/' + ID_COEFF)
            model_hyper_params.to_csv(id_dir + '/' + ID_HPARAMS)

            exportfeatures = features.copy()
            exportfeatures.insert(0, TARGET, target)

            model_preds.insert(0,TARGET, target[first_prediction_index:last_prediction_index+1])
            model_preds.to_csv(id_dir+'/' + ID_PREDS)
            exportfeatures.to_csv(id_dir + '/' + ID_DATA)

            #neighbor_ARGOdata[id_] = copy.copy(model_preds['ARGO'])
            #neighbor_ILIdata[id_] = copy.copy(target)

            print('Data for {0} generated.'.format(id_))

        #finishing the writing of data for the NET model

        # After the ARGO phase we build up the dataframes to work with the net model

        if not cont: print('Succesfully completed prediction iteration loop for ARGO models.')
        """

        print('Starting NET phase.')

        for id_ in ids:
            if cont: self.data.generate_ar_lags(which_id= id_, which_lags = which_lags)
            if neighbor_ARdata is not None:
                neighbor_ARdata = pd.concat([neighbor_ARdata, self.data.ar_lags[id_][neighbor_ar_model_labels]], axis=1)
            else:
                neighbor_ARdata = self.data.ar_lags[id_][neighbor_ar_model_labels]
            neighbor_ILIdata[id_] = self.data.target[id_].values.ravel()
            preds_df = pd.read_csv('{0}/{1}/'.format(self.output_name, id_) + ID_PREDS, index_col=[0])

            if self.only_ar_models:
                neighbor_ARGOdata[id_] = copy.copy(preds_df['AR'])
            elif self.least_error_syncronicity:
                neighbor_ARGOdata[id_] = get_timeseries_historic_winner(preds_df, 3, ['ARGO', 'AR'], TARGET)
            else:
                neighbor_ARGOdata[id_] = copy.copy(preds_df['ARGO'])

        neighbor_ARGOdata = pd.DataFrame(neighbor_ARGOdata)
        neighbor_ILIdata = pd.DataFrame(neighbor_ILIdata, index=self.data.target[id_].index)

        labels1=[]
        for n, id_ in enumerate(ids): labels1+= [n]*len(neighbor_ar_model_labels)
        labels2=list(range(len(neighbor_ar_model_labels)))*len(ids)

        ind=pd.MultiIndex(levels=[ids, neighbor_ar_model_labels],
                   labels=[labels1, labels2],
                   names=['ID', 'LAG'])

        neighbor_ARdata.columns=ind
        neighbor_ARGOdata.to_csv(self.overview_folder+'/'+CSV_ARGONET_DATA_ARGO)

        neighbor_ARdata.to_csv(self.overview_folder+'/'+CSV_ARGONET_DATA_AR)
        neighbor_ILIdata.to_csv(self.overview_folder+'/'+CSV_ARGONET_DATA_ILI)

        print('NET data written into overview folder.')
        time.sleep(1)
        # NET MODEL PART
        # ---------------------------------------------------------------------

        #Checkpoint

        if id_net:
            cont_ids=[id_net]
        elif cont and cp_phase == 'NET':
            id_ind = ids.index(cp_id)
            cont_ids = ids[id_ind:]
        elif cont and cp_phase != 'ARGO' and cp_phase != 'NET':
            cont_ids = []
        else:
            cont_ids = ids

        for i, id_ in enumerate(cont_ids):
            # savepoint
            pickle.dump(['NET', id_],open(self.overview_folder+'/'+PICKLE_ARGONET_CHECKPOINT,'wb'))


            print('Preparing data and performing a forecasting loop for {0}'.format(id_))


            id_dir = self.output_name+'/{0}'.format(id_)
            # Initializing predictors and AR terms.
            predictors = {}
            which_lags = list(set(ar_model+neighbor_ar_model))

            if verbose:
                print('Autoreggressive model for {0} is: {1} \n \
                      Models detected: {2}'.format(id_, ar_model))
                time.sleep(1)

            net_models =[]

            mods = list(dict_net_model.keys())

            if 'NET_lrmse' in mods:
                dict_net_model.pop('NET_lrmse')
                NET_lrmse = True
            else:
                NET_lrmse = False

            for mod, params in dict_net_model.items():
                predictors[mod] = Predictor(model=mod, model_funct=params[0], \
                                              model_preproc=params[1], \
                                              mod_params=params[2])
                net_models.append(mod)


            if NET_lrmse:
                predictors['NET_lrmse'] = Predictor(model='NET_lrmse', model_funct='fake', \
                                              model_preproc='fake', \
                                              mod_params='fake')

            #Entering data generation loop per id
            highest_ar_lag = np.max(which_lags)
            target = self.data.target[id_]
            features = self.data.features[id_]
            horizon = self.horizon - 1
            feature_preproc = self.feature_preproc

            if verbose:
                print('highest_ar_lag:{0}\n period_start:{1} \n period_end:{2} \n \
                      horizon:{3} \n feature_preproc:{4}'.format(highest_ar_lag, period_start, period_end, horizon, feature_preproc))
                time.sleep(1)

            w = self.training_window
            s = self.training_window_size

            if isinstance(s, int) and s < features.shape[0]:
                training_window_size = s
            else:
                print('WARNING! Error with training window size. Using default')
                training_window_size = TRAINING_WINDOW_SIZE_DEFAULT

            try:
                if w in training_window_handler.keys():
                    training_window = w
                    training_window_size = s

            except KeyError:
                print('WARNING! Error defining the training window type. Using default')
                training_window = TRAINING_WINDOW_TYPE_DEFAULT

            if verbose:
                print('Training window parameters:\n size:{0}, type:{1} \n\n'.format(training_window_size, training_window))

            index_start = features.index.get_loc(period_start)
            index_end = features.index.get_loc(period_end)
            first_available_index =  highest_ar_lag + horizon + training_window_size

            if index_start >= first_available_index:
                first_prediction_index = index_start
                first_window_index = index_start - training_window_size
            else:
                print('Prediction start date falls within minimum training dataset period \n \
                      period_start=[{0}, loc_in_df:{1}] \n \
                      first_available_index=[{2}, loc_in_df:{3}] \n \
                      highest_ar_lag = {4}, training_window_size {5}. \n\n \
                      Prediction should start at index {1} but model minimum\
                       data points needed for training set at start is: {6}.\n \
                       Please update the training dataset or autoreggresive parameters.\
                      '.format(period_start, index_start, first_available_index,\
                               features.index[first_available_index], highest_ar_lag,\
                               training_window_size, training_window_size + highest_ar_lag))
                return

            last_prediction_index = index_end
            dates = target.index[first_prediction_index:last_prediction_index+1]

            if verbose:
                print('Main loop parameters \n \
                      period_start=[{0}, loc_in_df:{1}] \n \
                      first_available_index=[{2}, loc_in_df:{3}] \n \
                      highest_ar_lag = {4}, training_window_size ={5}. \n \
                      last_prediction_index = [{6}, date: {7}]\n \
                      '.format(period_start, index_start, first_available_index,\
                               features.index[first_available_index], highest_ar_lag,\
                               training_window_size, last_prediction_index, period_end))
            rg = list(range(first_prediction_index, last_prediction_index+1))

            bar = ChargingBar('Processing', max=len(rg))

            for j in rg:
                ar_model_labels = ['AR{0}'.format(v) for i,v in enumerate(ar_model)]
                bar.next()
                # range(a,b) runs from a to b-1.

                '''
                Main iteration loop works as follows:
                1.- defines the training window to work on
                2.- generates next day prediction
                '''

                # Preparing training window
                training_window_size, window_indices = training_window_handler[training_window](\
                                training_window_size,first_window_index, \
                                  j, features, target)


                if verbose:
                    print('Training window indices and size: \n \
                          training_window_size:{0} \n \
                          window_indices: {1}'.format(training_window_size, window_indices))
                    time.sleep(1)

                '''
                After retrieving the data points from our training_window function,
                we receive a set of indices indicating the position of the training samples
                in our features dataframe. Now we want to create our training data features and target.
                We first generate the training target vector with the indices we received and then
                we update the values of the indices by the desired horizon.
                '''
                window_indices = np.array(window_indices)
                Y_train = np.array(target)[window_indices]
                window_indices -= horizon


                regular_input_train = copy.deepcopy(features.iloc[window_indices,:])
                regular_input_predict  = copy.deepcopy(features.iloc[j,:])

                if self.filter_externalvars_by_similarity:
                    regular_input_train = filter_by_similarity(regular_input_train,Y_train.ravel(), func=None, threshold=externalvars_threshold, column_limit=column_limit)

                if verbose:
                    print('Training dataset indices, size: \n \
                          window_indices:{0} \n \
                          j:{1} \n \
                          Y_train.shape:{2}'.format(window_indices, j, Y_train.shape))

                if NET_lrmse:
                    winner = ''
                    least_rmse = float('inf')

                for mod in net_models:
                    X_train = []
                    X_predict = []

                    # AR INPUT

                    ar_pd = self.data.ar_lags[id_][ar_model_labels]
                    ar_train = copy.deepcopy(ar_pd.iloc[window_indices,:])
                    if self.ar_column_filtering:
                        ar_train= filter_ar_columns(ar_train)
                    ar_predict = copy.deepcopy(ar_pd.iloc[j,:])


                    # NEIGHBOR  AR INPUT

                    #find out neighbors
                    neighbor_ids = copy.deepcopy(ids)
                    neighbor_ids.remove(id_)


                    #First, ILI from time t-time_window to t-1 (we're not supposed to have time t yet)
                    # and AR lags based of neighbor_ar_model

                    neighbor_ili_input = copy.deepcopy(neighbor_ILIdata.iloc[window_indices, neighbor_ILIdata.columns.get_level_values(0) != id_])
                    neighbor_ar_input = neighbor_ARdata.iloc[window_indices, neighbor_ARdata.columns.get_level_values(0) != id_]

                    if self.per_location_neighbors:

                        neighbor_filter = self.per_location_neighbors[id_]
                        #Filtering ili input
                        for col in list(neighbor_ili_input):
                            if col in neighbor_filter:
                                pass
                            else:
                                neighbor_ili_input[col] = neighbor_ili_input[col].apply(lambda x : 0)
                        #Filtering ili input
                        for col in neighbor_ar_input.columns.get_level_values(0):

                            if col in neighbor_filter:
                                pass
                            else:
                                indices = [i for i, v in enumerate(neighbor_ar_input.columns.get_level_values(0) == col) if v == True]
                                for ind in indices:
                                    neighbor_ar_input.iloc[:, ind] = neighbor_ar_input.iloc[:, ind].apply(lambda x : 0)


                    if self.filter_neighbors_by_similarity:
                        neighbor_ili_input = filter_by_similarity(neighbor_ili_input, Y_train.ravel(), func=None, threshold=dict_net_model[mod][3], column_limit=dict_net_model[mod][4])
                        neighbor_ar_input =  filter_by_similarity(neighbor_ar_input, Y_train.ravel(), func=None, threshold=dict_net_model[mod][3], column_limit=dict_net_model[mod][4])




                    neighbor_ar_predict =  neighbor_ARdata.iloc[j, neighbor_ARdata.columns.get_level_values(0) != id_]

                    # Finally, we use the data from our neighbor's ARGO prediction as a proxy
                    # for the ILI data at time t. Our ARGO dataframe may not be in sync with the loop
                    # so it is neccessary to access the ARGO predictions via the date we're aiming to predictself.

                    if isinstance(target.index[j], datetime.date):
                        prediction_date_index = neighbor_ARGOdata.index.get_loc(target.index[j].strftime('%Y-%m-%d'))
                    else:
                        prediction_date_index = neighbor_ARGOdata.index.get_loc(target.index[j])

                    if self.argonet_filter_columns:
                        neighbor_ar_input = filter_ar_columns(neighbor_ar_input)
                        neighbor_ili_input = filter_ar_columns(neighbor_ili_input)

                    neighbor_ili_predict = neighbor_ARGOdata.iloc[prediction_date_index, neighbor_ARGOdata.columns.get_level_values(0) != id_] # IMPORTANT

                    # Our training dataset then is comprised of the ar_model + google search queries
                    # specified, plus the neighbo_ar_model + the ili at time to predict.
                    # Our prediction sample will be comprised of the same, but instead of ili, we use argo predictions

                    X_train = [ar_train, regular_input_train, neighbor_ili_input, neighbor_ar_input]
                    X_predict = [ar_predict, regular_input_predict, neighbor_ili_predict, neighbor_ar_predict]


                    X_train = np.hstack(X_train)
                    X_predict = np.hstack(X_predict)
                    #Preprocessing
                    #X_train, X_predict, Y_train = feature_preproccessing_handler[feature_preproc](X_train, \
                    #                                                X_predict, Y_train)

                    X_train, X_predict, counter, location = check_predict_features(X_train, X_predict)
                    if counter > 0:
                        print('Warning! {0} NaN(s) found within prediction sample features for index : {1} and model {3} at location(s) {2}\n\n.'.format(\
                                            counter, target.index[j], location, model))
                    predictors[mod].predict(X_train, Y_train, X_predict)

                    if mod != 'AR':
                        val = predictors[mod].insample_metric[-1]

                        if val < least_rmse:
                            least_rmse = val
                            winner = mod

                predictors['NET_lrmse'].predictions.append(predictors[winner].predictions[-1])
                predictors['NET_lrmse'].coefficients.append(predictors[winner].coefficients[-1])
                predictors['NET_lrmse'].hyper_params.append(predictors[winner].hyper_params[-1])

            #After finishing all the predictions, we extract all data and export it
            # to a folder generated by the code
            bar.finish()
            self.first_prediction_index = first_prediction_index
            self.last_prediction_index = last_prediction_index

            if NET_lrmse:
                mods = net_models + ['NET_lrmse']
            else:
                mods = net_models

            for k, model in enumerate(mods):

                if k == 0:

                    if predictors[model].store_predictions == True:
                        model_preds = np2dataframe(dates, model,np.array(predictors[model].predictions))

                    if predictors[model].store_coefficients == True:
                        model_coefficients = np2dataframe(dates, model,np.array(np.vstack(predictors[model].coefficients)))
                    if predictors[model].store_hyper_params == True:
                        model_hyper_params = np2dataframe(dates, model,np.array(np.vstack(predictors[model].hyper_params)))
                else:
                    if predictors[model].store_predictions == True:
                        model_preds = pd.concat([model_preds,\
                                                 np2dataframe(dates, model,np.array(predictors[model].predictions))], axis=1)
                    if predictors[model].store_coefficients == True:
                        model_coefficients = pd.concat([model_coefficients,\
                                                       np2dataframe(dates, model,np.array(np.vstack(predictors[model].coefficients)))], axis = 1)
                    if predictors[model].store_hyper_params == True:
                        model_hyper_params = pd.concat([model_hyper_params, \
                                                        np2dataframe(dates, model,np.array(np.vstack(predictors[model].hyper_params)))], axis = 1)

            #model_preds.to_csv(id_dir+'/' + ID_PREDS)
            model_coefficients.to_csv(id_dir + '/ARGONET_' + ID_COEFF)
            model_hyper_params.to_csv(id_dir + '/ARGONET_' + ID_HPARAMS)

            preds_df = pd.read_csv(id_dir+'/' + ID_PREDS, index_col=[0])
            preds_df = pd.concat([preds_df, model_preds], axis=1)
            preds_df.to_csv(id_dir+'/' + ID_PREDS)

            print('Data for {0} generated.'.format(id_))

        print('Successfully finished NET phase')
