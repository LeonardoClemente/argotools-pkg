from argotools.forecastlib.functions import *
from argotools.forecastlib.argo_methods_ import *
'''
dictionaries containing all handler functions.
Handler
'''

default_ar =  s=list(range(1,53))


method_handler = {
    'NET': lasso,
    'AR': lasso,
    'AR1': lasso,
    'AR12':lasso,
    'AR52': lasso,
    'ARGO': lasso,
    'AR1GO': lasso,
    'AR12GO': lasso,
    'AR52GO': lasso,
    'AR12GO_blr': bayesian_ridge_regression,
    'adaboost':adaboost,
    'lasso_adaboost':lasso_adaboost,
   'lasso': lasso,
   'lasso_1se': lasso_1se,
   'adalasso': adaLasso,
   'adalasso-cv': adaLassoCV,
   'None': no_method,
   'weighted-lasso': weighted_lasso,
   'weighted-lasso1': weighted_lasso1,
   'weighted-lasso2': weighted_lasso2,
   'weighted-lasso3': weighted_lasso3,
   'lasso-obs': lasso_obs,
   'lasso-obs-weighted': lasso_obs_weighted,
   'weighted-lasso2-obs': weighted_lasso2_obs,
   'ath-weight': ath_weight_lasso,
   'ath-ar-weight': ath_ar_weight_lasso,
   'ath-deweight': ath_deweight_lasso,
   'wl1': wl1,
   'sort-weight': sort_weight_lasso,
   'partial-ls': partial_least_squares,
   #'nanARGO': nanARGO,
   #'f_regression': F_regression,
   #'elasticnet': Elasticnet
   }

ar_handler = {
    'adaboost': default_ar,
    'lasso_adaboost':default_ar,
    'AR' : default_ar,
    'AR1' : [1],
    'AR12' : list(range(1,13)),
    'AR52GO':list(range(1,53)),
    'AR52': list(range(1,53)),
    'default': default_ar,
    'AR1GO': [1],
    'AR12GO': default_ar,
    'AR12GO_blr': list(range(1,13)),
    'lasso': default_ar,
    'lasso_1se': default_ar,
    'adalasso': default_ar,
    'adalasso-cv': default_ar,
    'None': None,
    'weighted-lasso': default_ar,
    'weighted-lasso1': default_ar,
    'weighted-lasso2': default_ar,
    'weighted-lasso3': default_ar,
    'lasso-obs': default_ar,
    'lasso-obs-weighted':  default_ar,
    'weighted-lasso2-obs': default_ar,
    'ath-weight': default_ar,
    'ath-ar-weight': default_ar,
    'ath-deweight': default_ar,
    'wl1': default_ar,
    'sort-weight': default_ar,
    'partial-ls': default_ar,
    #'nanARGO': nanARGO,
    #'f_regression': F_regression,
    #'elasticnet': Elasticnet

}

preproc_handler = {
    'lasso_adaboost': preproc_rmv_values,
    'adaboost': preproc_rmv_values,
    'AR': preproc_rmv_values,
    'AR1': preproc_rmv_values,
    'AR12': preproc_rmv_values,
    'AR52':preproc_rmv_values,
    'default': preproc_rmv_values,
    'ARGO': preproc_rmv_values,
    'AR1GO': preproc_rmv_values,
    'AR12GO': preproc_rmv_values,
    'AR52GO': preproc_rmv_values,
    'AR12GO_blr': preproc_rmv_values,
    'lasso': preproc_rmv_values,
    'lasso_1se': preproc_rmv_values,
    'adalasso': preproc_rmv_values,
    'adalasso-cv': preproc_rmv_values,
    'weighted-lasso': preproc_rmv_values,
    'weighted-lasso1': preproc_rmv_values,
    'weighted-lasso2': preproc_rmv_values,
    'weighted-lasso3': preproc_rmv_values,
    'lasso-obs': preproc_rmv_values,
    'lasso-obs-weighted':  preproc_rmv_values,
    'weighted-lasso2-obs': preproc_rmv_values,
    'ath-weight': preproc_rmv_values,
    'ath-ar-weight': preproc_rmv_values,
    'ath-deweight': preproc_rmv_values,
    'wl1': preproc_rmv_values,
    'sort-weight': preproc_rmv_values,
    'partial-ls': preproc_rmv_values,
    #'nanARGO': nanARGO,
    #'f_regression': F_regression,
    #'elasticnet': Elasticnet

}

training_window_handler = {
    'static': static_window,
    'expanding': expanding_window,
}

feature_preproccessing_handler = {
    'unnormalized': unnormalized,
    'zscore': zscore,
    'mean_normalization': mean_normalization,
    'unit_length_normalization': unit_length_normalization,
    'rescaling': rescaling
}

metric_handler = {
    'RMSE': metric_RMSE,
    'PEARSON': metric_pearson,
    'MSE':metric_MSE,
    'NRMSE': metric_NRMSE
}

ensemble_handler = {
    'stack_average':stack_average,
    'stack_historyVoting': stack_historyVoting,
    'stack_linearRegression': stack_linearRegression,
    'stack_FWLR': stack_FWLR
}

ensemble_preproc_handler = {
    'stack_average':stack_no_preproc,
    'stack_historyVoting': stack_no_preproc,
    'stack_linearRegression': stack_no_preproc,
    'stack_FWLR': stack_no_preproc
}

has_ar = {
        'adaboost':True,
        'lasso_adaboost':True,
        'NET': True,
        'AR':True,
        'AR1' : True,
        'AR12' : True,
        'AR52' : True,
        'ARGO': True,
        'AR1GO': True,
        'AR12GO': True,
        'AR12GO_blr':True,
        'AR52GO':True,
       'lasso': True,
       'lasso_1se': True,
       'adalasso': True,
       'adalasso-cv': True,
       'None': True,
       'weighted-lasso':True,
       'weighted-lasso1': True,
       'weighted-lasso2': True,
       'weighted-lasso3': True,
       'lasso-obs': True,
       'lasso-obs-weighted': True,
       'weighted-lasso2-obs': True,
       'ath-weight': True,
       'ath-ar-weight': True,
       'ath-deweight': True,
       'wl1': True,
       'sort-weight': True,
       'partial-ls': True,
       #'nanARGO': nanARGO,
       #'f_regression': F_regression,
       #'elasticnet': Elasticnet
}

has_regularInput = {
            'adaboost':True,
            'lasso_adaboost': True,
            'AR':True,
            'AR1' : False,
            'AR12' : False,
            'AR52' : False,
            'ARGO' : True,
            'AR1GO': True,
            'AR12GO': True,
            'AR12GO_blr': True,
            'AR52GO':True,
           'lasso': True,
           'lasso_1se': True,
           'adalasso': True,
           'adalasso-cv': True,
           'None': True,
           'weighted-lasso':True,
           'weighted-lasso1': True,
           'weighted-lasso2': True,
           'weighted-lasso3': True,
           'lasso-obs': True,
           'lasso-obs-weighted': True,
           'weighted-lasso2-obs': True,
           'ath-weight': True,
           'ath-ar-weight': True,
           'ath-deweight': True,
           'wl1': True,
           'sort-weight': True,
           'partial-ls': True,
           #'nanARGO': nanARGO,
           #'f_regression': F_regression,
           #'elasticnet': Elasticnet
}


voting_selector_handler  = {
    'RMSE': rmse_selector,
    'ERROR': error_selector
}



handler_dict = {
    'methods':method_handler,
    'arModels': ar_handler,
    'preprocessingMethods': preproc_handler,
    'trainingWindows': training_window_handler,
    'featurePreproccessing': feature_preproccessing_handler,
    'metrics': metric_handler,
    'ensembles': ensemble_handler
}


def handlerKeys(handler_name, handler_dict):
    print(handler_dict[handler_name].keys())
    return

def handlerContains(handler_name, labels, handler_dict = handler_dict):
    # Receives a list and checks which elements are keys of a handler dictionary
    contained = []
    for i, label in enumerate(labels):
        if label in handler_dict[handler_name].keys():
            contained.append(label)
    print('{0} out of {1} items contained within {2} keys.'.format(len(contained), len(labels), handler_name))
    return contained, len(labels) - len(contained)
