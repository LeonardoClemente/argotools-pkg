import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_decomposition import PLSRegression
import time
from argotools.forecastlib.functions import preds2matrix
from argotools.config import MISSING_DATA
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor


# GET PARAMS
def lasso_params(model, zero_vector_indices):

    coefficients = model.coef_
    coefficients.reshape([1,-1])
    if len(zero_vector_indices) > 0:
        coefficients = reorder_coefficients(coefficients, zero_vector_indices)

    hyper_params = []
    p = model.get_params()
    for k,v in p.items():
        hyper_params.append(v)

    return coefficients, hyper_params

def bagging_lasso(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):
    n_predictors = X_train.shape[1]
    X_train, X_pred, zero_vector_indices = handle_zero_vectors(X_train, X_pred)
    lr = BaggingRegressor(base_estimator=LassoCV(cv=10, fit_intercept=False, \
                        n_alphas=1000, max_iter=20000, tol=.001, normalize=True),\
                         n_estimators=10, max_samples=.80,\
                         max_features=.90, bootstrap=True, bootstrap_features=True,\
                          oob_score=False, warm_start=False, verbose=0)
    model = lr.fit(X_train, y_train)

    if store_settings[0]:
       predictions = model.predict(X_pred.reshape(1,-1)).ravel()
    else:
       predictions = None

    if store_settings[1]:
       coefficients = np.zeros(n_predictors).ravel()

    else:
       coefficients = None
    if store_settings[2]:
       hyper_params = np.zeros(n_predictors).ravel()
    else:
       hyper_params = None
    return predictions, coefficients, hyper_params, metric_value

def bagging_tree(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):
    n_predictors = X_train.shape[1]
    X_train, X_pred, zero_vector_indices = handle_zero_vectors(X_train, X_pred)
    lr = BaggingRegressor(n_estimators=15, max_samples=.80,\
                         max_features=.90, bootstrap=True, bootstrap_features=True,\
                          oob_score=False, warm_start=False, verbose=0)

    model = lr.fit(X_train, y_train)

    if store_settings[0]:
       predictions = model.predict(X_pred.reshape(1,-1)).ravel()
    else:
       predictions = None

    if store_settings[1]:
       coefficients = np.zeros(n_predictors).ravel()

    else:
       coefficients = None
    if store_settings[2]:
       hyper_params = np.zeros(n_predictors).ravel()
    else:
       hyper_params = None
    return predictions, coefficients, hyper_params, metric_value




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

def reorder_coefficients(coefficients, zero_vector_indices):
    if np.size(coefficients) == 1:
        coefficients = [coefficients]
    n_coeff = len(coefficients) + len(zero_vector_indices)
    ordered_coeff = np.zeros(n_coeff)

    non_zero_index =0
    for i in range(n_coeff):
        if i in zero_vector_indices:
            pass
        else:
            ordered_coeff[i] = coefficients[non_zero_index]
            non_zero_index += 1

    return ordered_coeff.reshape(1,-1)

def bayesian_ridge_regression(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):

    lr = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06,\
                       compute_score=False, copy_X=True,\
                       fit_intercept=False, lambda_1=1e-06,\
                       lambda_2=1e-06, n_iter=300,\
                       normalize=True, tol=0.001,\
                        verbose=False)
    model = lr.fit(X_train, y_train)


    if store_settings[0]:
      predictions = model.predict(X_pred.reshape(1,-1))
    else:
      predictions = None


    if store_settings[1]:
        coefficients = model.coef_

        if np.size(coefficients) == len(X_pred):
            coefficients.reshape([1,-1])
        else:
            coefficients.reshape([1,-1])
    else:
        coefficients = None


    if store_settings[2]:
        hyper_params = []
        p = model.get_params()
        for k,v in p.items():
            hyper_params.append(v)
    else:
        hyper_params = None

    return predictions, coefficients, hyper_params, metric_value

def adaboost(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):

    lr = AdaBoostRegressor( n_estimators=8, learning_rate=1.0, loss='square', random_state=None)
    model = lr.fit(X_train, y_train)


    if store_settings[0]:
      predictions = model.predict(X_pred.reshape(1,-1))
    else:
      predictions = None


    if store_settings[1]:
        coefficients = model.estimator_weights_.reshape([1,-1])
    else:
        coefficients = None


    if store_settings[2]:
        hyper_params = []
        p = model.get_params()
        for k,v in p.items():
            hyper_params.append(v)
    else:
        hyper_params = None

    return predictions, coefficients, hyper_params, metric_value

def lasso_adaboost(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):

    lr = AdaBoostRegressor(base_estimator=LassoCV(cv=10, fit_intercept=False, n_alphas=1000, max_iter=10000, tol=.001, normalize=True), \
                                                  n_estimators=8, learning_rate=1.0, loss='linear', random_state=None)
    model = lr.fit(X_train, y_train)


    if store_settings[0]:
      predictions = model.predict(X_pred.reshape(1,-1))
    else:
      predictions = None


    if store_settings[1]:
        coefficients = model.estimator_weights_.reshape([1,-1])
    else:
        coefficients = None


    if store_settings[2]:
        hyper_params = []
        p = model.get_params()
        for k,v in p.items():
            hyper_params.append(v)
    else:
        hyper_params = None

    return predictions, coefficients, hyper_params, metric_value

def random_forest_with_lasso_coefficients(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):
    n_predictors = X_train.shape[1]
    X_train, X_pred, zero_vector_indices = handle_zero_vectors(X_train, X_pred)
    l = LassoCV(cv=10, fit_intercept=True, n_alphas=1000, max_iter=20000, tol=.001, normalize=True)
    mod = l.fit(X_train, y_train)
    coeffs = np.abs(mod.coef_)
    column_val = sorted(zip(coeffs, list(range(len(coeffs))) ), key = lambda x:x[0], reverse=True)
    column_ind = np.array([j for c,j in column_val])
    X_train = np.array(X_train)
    X_train = X_train[:, column_ind[0:10]]
    X_pred = np.array(X_pred)[column_ind[0:10]]
    lr = RandomForestRegressor(n_estimators=150, random_state=0)
    model = lr.fit(X_train, y_train)

    if store_settings[0]:
       predictions = model.predict(X_pred.reshape(1,-1)).ravel()
    else:
       predictions = None

    if store_settings[1]:
       coefficients = np.zeros(n_predictors).ravel()

    else:
       coefficients = None
    if store_settings[2]:
       hyper_params = np.zeros(n_predictors).ravel()
    else:
       hyper_params = None
    return predictions, coefficients, hyper_params, metric_value

def lasso_positiveCoeff(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):

        X_train, X_pred, zero_vector_indices = handle_zero_vectors(X_train, X_pred)
        lr = LassoCV(cv=10, fit_intercept=True, n_alphas=1000, max_iter=20000, tol=.001, normalize=True,\
                     positive=True)
        model = lr.fit(X_train, y_train)

        if store_settings[0]:
           predictions = model.predict(X_pred.reshape(1,-1)).ravel()
        else:
           predictions = None

        if store_settings[1]:
           coefficients = model.coef_
           if np.size(coefficients) == len(X_pred):
               coefficients.reshape([1,-1])
           else:
               coefficients.reshape([1,-1])

           if len(zero_vector_indices) > 0:
               coefficients = reorder_coefficients(coefficients, zero_vector_indices)

        else:
           coefficients = None
        if store_settings[2]:
           hyper_params = []
           p = model.get_params()
           for k,v in p.items():
               hyper_params.append(v)
        else:
           hyper_params = None

        return predictions, coefficients, hyper_params, metric_value

def random_forest(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):
    n_predictors = X_train.shape[1]
    X_train, X_pred, zero_vector_indices = handle_zero_vectors(X_train, X_pred)
    lr = RandomForestRegressor(n_estimators=150, random_state=0)
    model = lr.fit(X_train, y_train)

    if store_settings[0]:
       predictions = model.predict(X_pred.reshape(1,-1)).ravel()
    else:
       predictions = None

    if store_settings[1]:
       coefficients = np.zeros(n_predictors).ravel()

    else:
       coefficients = None
    if store_settings[2]:
       hyper_params = np.zeros(n_predictors).ravel()
    else:
       hyper_params = None

    return predictions, coefficients, hyper_params, metric_value


def lasso_family(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):


    print('INPUT prior to zero vectors',X_train)
    time.sleep(15)
    X_train, X_pred, zero_vector_indices = handle_zero_vectors(X_train, X_pred)

    model = mod_params.fit(X_train, y_train)
    if store_settings[0]:
       predictions = model.predict(X_pred.reshape(1,-1)).ravel()
    else:
       predictions = None

    if store_settings[1]:
       coefficients = model.coef_
       if np.size(coefficients) == len(X_pred):
           coefficients.reshape([1,-1])
       else:
           coefficients.reshape([1,-1])

       if len(zero_vector_indices) > 0:
           coefficients = reorder_coefficients(coefficients, zero_vector_indices)

    else:
       coefficients = None
    if store_settings[2]:
       hyper_params = []
       p = model.get_params()
       for k,v in p.items():
           hyper_params.append(v)
    else:
       hyper_params = None

    if metric:
        insample_preds = model.predict(X_train)
        metric_value = metric(insample_preds[-12:], y_train[-12:])#CHANGE
    else:
        metric_value = None

    return predictions, coefficients, hyper_params, metric_value





def lasso(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):

    X_train, X_pred, zero_vector_indices = handle_zero_vectors(X_train, X_pred)
    lr = LassoCV(cv=10, fit_intercept=False, n_alphas=1000, max_iter=20000, tol=.001, normalize=True)
    model = lr.fit(X_train, y_train)
    if store_settings[0]:
       predictions = model.predict(X_pred.reshape(1,-1)).ravel()
    else:
       predictions = None

    if store_settings[1]:
       coefficients = model.coef_

       if isinstance(coefficients, int):
           print('int a list')
           coefficients = [coefficients]

       if np.size(coefficients) == len(X_pred):
           coefficients.reshape([1,-1])
       else:
           coefficients.reshape([1,-1])

       if len(zero_vector_indices) > 0:
               coefficients = reorder_coefficients(coefficients, zero_vector_indices)


    else:
       coefficients = None
    if store_settings[2]:
       hyper_params = []
       p = model.get_params()
       for k,v in p.items():
           hyper_params.append(v)
    else:
       hyper_params = None

    if metric:
        insample_preds = model.predict(X_train)
        metric_value = metric(insample_preds, y_train)
    else:
        metric_value = None

    return predictions, coefficients, hyper_params, metric_value


def lasso_1se(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):

   Ltest = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
   Ltest.fit(X_train, y_train)

   # compute minimum cross-validation error + standard error as threshold
   alpha_index = np.where(Ltest.alphas_ == Ltest.alpha_)
   cv_means = np.mean(Ltest.mse_path_, axis=1)
   threshold = cv_means[alpha_index] + (np.std(Ltest.mse_path_[alpha_index]) / np.sqrt(10))

   # find highest alpha (sparsest model) with cross-validation error below the threshold
   alpha_new = max(Ltest.alphas_[np.where((Ltest.alphas_ >= Ltest.alpha_) & (cv_means < threshold))])

   # fit lasso with this alpha and predict training set
   Lpred = Lasso(alpha=alpha_new, max_iter=10000, tol=.0005, warm_start=True)

   return Lpred.fit(X_train, y_train).predict(X_pred)


def adaLasso(X_train, y_train, X_pred, store_settings, gamma=1.1):

   alpha_grid = np.logspace(-1.5, 1, 20)
   estimator = RidgeCV(alphas=alpha_grid, cv=5)
   betas = estimator.fit(X_train, y_train).coef_

   weights = np.power(np.abs(betas), gamma)
   lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
   return lr.fit(X_train * weights, y_train).predict(X_pred * weights)


def adaLassoCV(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):

   gamma = [0, 1.1, 2]
   alpha_grid = np.logspace(-1.5, 1, 20)
   estimator = RidgeCV(alphas=alpha_grid, cv=5)
   betas = estimator.fit(X_train, y_train).coef_

   mse_array = np.empty(len(gamma))
   for counter, g in enumerate(gamma):
       weights = np.power(np.abs(betas), g)
       lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
       training_fit = lr.fit(X_train * weights, y_train).predict(X_train * weights)
       mse_array[counter] = mse(training_fit, y_train)

   optimal_g = gamma[np.argmin(mse_array)]
   print(np.argmin(mse_array))
   weights = 1 / np.power(np.abs(betas), optimal_g)
   prediction = lr.fit(X_train / weights, y_train).predict(X_pred / weights)

   return prediction


# ------------ WEIGHTINGS ------------- #
def lasso_obs(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):
   X_train_wt = np.vstack((X_train, X_train[-4:, :]))
   y_train_wt = np.append(y_train, y_train[-4:])
   lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
   return lr.fit(X_train_wt, y_train_wt).predict(X_pred)

def lasso_obs_weighted(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):
   X_train[:, [0, 1, 2, 52, 53, 54]] = X_train[:, [0, 1, 2, 52, 53, 54]] * 10
   X_pred[:, [0, 1, 2, 52, 53, 54]] = X_pred[:, [0, 1, 2, 52, 53, 54]] * 10

   X_train_wt = np.vstack((X_train, X_train[-4:, :]))
   y_train_wt = np.append(y_train, y_train[-4:])

   lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
   return lr.fit(X_train_wt, y_train_wt).predict(X_pred)


def weighted_lasso(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):
   # AR1,2,3 + top 3 exog.
   X_train[:, [0, 1, 2, 52, 53, 54]] = X_train[:, [0, 1, 2, 52, 53, 54]] * 10
   X_pred[:, [0, 1, 2, 52, 53, 54]] = X_pred[:, [0, 1, 2, 52, 53, 54]] * 10

   lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
   return lr.fit(X_train, y_train).predict(X_pred)
   return X_pred

def weighted_lasso1(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):

   X_train[:, [0, 1, 2, 51, 52, 53, 54]] = X_train[:, [0, 1, 2, 51, 52, 53, 54]] * 15
   X_pred[:, [0, 1, 2, 51, 52, 53, 54]] = X_pred[:, [0, 1, 2, 51, 52, 53, 54]] * 15

   lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
   return lr.fit(X_train, y_train).predict(X_pred)
   return X_pred

def weighted_lasso2(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):
   # AR1,2,3,52, first 5 exog.
   X_train[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] = X_train[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] * 10
   X_pred[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] = X_pred[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] * 10

   lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
   return lr.fit(X_train, y_train).predict(X_pred)
   return X_pred

def weighted_lasso2_obs(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):
   # AR1,2,3,52, first 5 exog.
   X_train[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] = X_train[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] * 10
   X_pred[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] = X_pred[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] * 10

   X_train_wt = np.vstack((X_train, X_train[-4:, :]))
   y_train_wt = np.append(y_train, y_train[-4:])

   lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
   return lr.fit(X_train_wt, y_train_wt).predict(X_pred)
   return X_pred

def weighted_lasso3(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):
   # AR1,2,3,6,12,52, first 3 exog.
   X_train[:, [0, 1, 2, 5, 11, 51, 52, 53, 54]] = X_train[:, [0, 1, 2, 5, 11, 51, 52, 53, 54]] * 10
   X_pred[:, [0, 1, 2, 5, 11, 51, 52, 53, 54]] = X_pred[:, [0, 1, 2, 5, 11, 51, 52, 53, 54]] * 10

   lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
   return lr.fit(X_train, y_train).predict(X_pred)
   return X_pred

def sort_weight_lasso(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):
   # top 5 exog.
   X_train[:, [0, 1, 2, 3, 4]] = X_train[:, [0, 1, 2, 3, 4]] * 10
   X_pred[:, [0, 1, 2, 3, 4]] = X_pred[:, [0, 1, 2, 3, 4]] * 10

   lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
   return lr.fit(X_train, y_train).predict(X_pred)
   return X_pred


def ath_weight_lasso(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):

   X_train[:, [52, 53, 54]] = X_train[:, [52, 53, 54]] * 10
   X_pred[:, [52, 53, 54]] = X_pred[:, [52, 53, 54]] * 10

   lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
   return lr.fit(X_train, y_train).predict(X_pred)
   return X_pred

def ath_ar_weight_lasso(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):

   X_train[:, [0, 1, 2, 52, 53, 54]] = X_train[:, [0, 1, 2, 52, 53, 54]] * 10
   X_pred[:, [0, 1, 2, 52, 53, 54]] = X_pred[:, [0, 1, 2, 52, 53, 54]] * 10

   lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
   return lr.fit(X_train, y_train).predict(X_pred)
   return X_pred

def ath_deweight_lasso(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):

   X_train[:, [52, 53, 54]] = X_train[:, [52, 53, 54]] * .1
   X_pred[:, [52, 53, 54]] = X_pred[:, [52, 53, 54]] * .1

   lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
   return lr.fit(X_train, y_train).predict(X_pred)
   return X_pred


def wl1(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):

   X_train[:, [0, 1, 2, 3, 12, 25, 51]] = X_train[:, [0, 1, 2, 3, 12, 25, 51]] * 10
   X_pred[:, [0, 1, 2, 3, 12, 25, 51]] = X_pred[:, [0, 1, 2, 3, 12, 25, 51]] * 10

   lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
   return lr.fit(X_train, y_train).predict(X_pred)
   return X_pred


def partial_least_squares(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):

   lr = PLSRegression(n_components=1, max_iter=1000, tol=1e-04)
   return lr.fit(X_train, y_train).predict(X_pred)
   return X_pred


def no_method(X_train, y_train, X_pred, store_settings, mod_params=None, metric=None):
   ''' returns the input variable directly as the prediction.
   '''
   return X_pred


'''
ENSEMBLE METHODS
'''

def stack_average(X_train, Y_train, X_pred, Y_ensemble, Y_index, predictor_output, \
                     meta_features, store_settings, training_window = 26):

    most_recent_preds = []

    for model, preds in predictor_output.items():
        most_recent_preds.append(preds[-1])

    predictions = np.mean(np.vstack(most_recent_preds), axis=0)
    coefficients = np.array([0])
    hyperparams = np.array([0])

    return predictions, coefficients, hyperparams

def stack_historyVoting(X_train, Y_train, X_pred, Y_ensemble, Y_index, predictor_output, \
                     meta_features, store_settings, n_points=3, training_window = 26, alpha=.95):

    '''
    historyVoting selects a prediction from the most 'voted' model.
    Voting works as follows:

        1.- Look for the n_points last Y_train values
        2.- For each value, find the most accurate model (lowest error)
            and generate a list
        3.- Based on the list, select the most frequent model.

    alpha is an hyper-parameter used to break tries between models by giving
    a preference to more recent predictions than really old predictions
    '''


    model_names = list(predictor_output.keys())
    model_counter = dict( zip( model_names, [0]*len(predictor_output)))
    Y_ensemble = np.vstack(Y_ensemble)
    coefficients = np.array(0)
    hyper_params = np.array([alpha])

    #Checking number of predictions available:
    n_available_preds = len(predictor_output[model_names[0]]) -1

    if len(Y_ensemble)-1  < n_points:
        predictions = np.array([MISSING_DATA])
        return predictions, coefficients, hyper_params, metric_value

    for i in range(0, n_points):
        target_value = Y_ensemble[0-(i+2)]
        error = float("inf")
        best_model = ''
        best_value = ''

        for model, predictions in predictor_output.items():
            model_value = predictions[0-(i+2)]

            if not np.isnan(model_value) and np.abs(target_value - model_value) < error:
                best_model = model
                error = np.abs(model_value - target_value)

        #print('DEBUGGING: best model from iteration is {0} with error of {1}'.format(best_model,error))
        model_counter[best_model] += 1*(alpha**i)

    counts = list(model_counter.values())
    best_model = ''
    best_count = 0
    for j in range(0, len(model_names)):

        model_name = model_names.pop()
        count = counts.pop()

        if count > best_count:
            best_model = model_name
            best_count = count

    #print('DEBUGGING: winner is {0}'.format(best_model))
    predictions = predictor_output[best_model][-1]

    return predictions, coefficients, hyper_params, metric_value


def stack_linearRegression(X_train, Y_train, X_pred, Y_ensemble, Y_index, predictor_output, \
                     meta_features, store_settings, training_window = 26):

    # Turn preds into input
    model_names = list(predictor_output.keys())
    output_matrix = preds2matrix(predictor_output)
    Y_ensemble = np.hstack(Y_ensemble)

    n_preds = len(Y_ensemble) - 1

    if n_preds >= training_window:
        E_train = output_matrix[0:n_preds-1,:]
        E_pred = output_matrix[n_preds-1,:]
        E_y_train = Y_ensemble[0:n_preds-1]
        lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
        model = lr.fit(E_train, E_y_train)

        if store_settings[0]:
           predictions = model.predict(E_pred.reshape(1,-1))
        else:
           predictions = None

        if store_settings[1]:
           coefficients = model.coef_

           if np.size(coefficients) == len(E_pred):
               coefficients.reshape([1,-1])
           else:
               coefficients.reshape([1,-1])
        else:
           coefficients = None
        if store_settings[2]:
           hyper_params = []
           p = model.get_params()
           for k,v in p.items():
               hyper_params.append(v)

           hyper_params = np.array(hyper_params)
    else:
       hyper_params =np.array([MISSING_DATA]*15) # Linear regression returns 15 values
       #print('DEBUGGING: ENTERED FIRST OPTION ')
       predictions = np.array([MISSING_DATA])
       coefficients = np.array([0]*len(model_names))


    return predictions, coefficients, hyper_params, metric_value

def stack_FWLR(X_train, Y_train, X_pred, Y_ensemble, Y_index, predictor_output, \
                     meta_features, store_settings, training_window = 10):


    Y_ensemble = np.hstack(Y_ensemble)
    Y_index = np.hstack(Y_index)


    if  Y_ensemble.size - 1 > training_window:

        # Extract metafeatures that match the indices from the target
        # Extract the last training_window samples of data
        mf_matrix = meta_features.iloc[Y_index[-(training_window+1):],:]
        mf_matrix = np.hstack([np.ones([mf_matrix.shape[0],1] ),mf_matrix]) # add a feature of 1s to add prediction matrices without weighting
        predictions_matrix = preds2matrix(predictor_output)
        predictions_matrix = predictions_matrix[-(training_window+1):,:]
        weighted_matrices=[]

        # Weighting matrices
        for i in range(mf_matrix.shape[1]):
                weighted_matrices.append(predictions_matrix*(mf_matrix[:,i].reshape([mf_matrix.shape[0],1])))

        weighted_matrices = np.hstack(weighted_matrices)
        E_x_train = weighted_matrices[-(training_window+1):-1,:]
        E_y_train =  Y_ensemble[-(training_window+1):-1]
        E_pred =  weighted_matrices[-1,:]

        lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
        model = lr.fit(E_x_train, E_y_train)

        if store_settings[0]:
           predictions = model.predict(E_pred.reshape(1,-1))
        else:
           predictions = np.array([MISSING_DATA])

        if store_settings[1]:
           coefficients = model.coef_

           if np.size(coefficients) == len(E_pred):
               coefficients.reshape([1,-1])
           else:
               coefficients.reshape([1,-1])
        else:
           coefficients = np.array([MISSING_DATA])

        if store_settings[2]:
           hyper_params = []
           p = model.get_params()
           for k,v in p.items():
               hyper_params.append(v)
           hyper_params = np.array(hyper_params)

    else:
       hyper_params = np.array([MISSING_DATA]*15) # Linear regression returns 15 values
       #print('DEBUGGING: ENTERED FIRST OPTION ')
       predictions = np.array([MISSING_DATA])
       coefficients = np.array([MISSING_DATA]*(len(predictor_output)*(meta_features.shape[1]+1)))
    return predictions, coefficients, hyper_params, metric_value
