import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#import stationary_block_bootstrap as sbb
import pandas as pd
import numpy as np
import scipy.stats
import numpy
import time
import random
#import state_variables
import os
import scipy.stats
import sklearn.feature_selection
import matplotlib.gridspec as gridspec
import copy
from argotools.config import *
from argotools.forecastlib.handlers import *
from argotools.forecastlib.functions import *
import argotools.forecastlib.stationary_block_bootstrap as sbb
from argotools.dataFormatter import *
import seaborn as sns
import matplotlib.ticker as mticker
import math
from matplotlib.ticker import MaxNLocator,IndexFormatter, FormatStrFormatter

class OutputVis:

# Variables : top_n = 3, ranking_metric = 'rmse', ranking_season ='ALL_PERIOD', preds (vector/PD containing all predictions), metrics (matrix/PD containing all metrics),
# Load predictions and csvs from file,
# get name of models,  number of models, name of metrics, table variable names (season1, season2... allPeriod).
# Get RANKING METRIC or all models in the file. Check if theres more than one first.
 # FUNC STATISTICS BETWEEN THE MODELS : MEAN, VARIANCE, BEST MODEL, WORST MODEL
# figure 1 : Time-series, error and percent error
# figure 2:  metric / plot

	def __init__(self, folder_dir=None, ids=None, overview_folder='_overview'):

		# Loading tables and files
		if folder_dir is None:
			print('WARNING!  No main folder directory specified. Add it as an attribute \
			 specify it on every function call that requires it.')

		self.folder_main = folder_dir
		self.ids = ids
		self.overview_folder = overview_folder
		print('Visualizer initialized')

		# imported VARS

	def plot_SEC(self, series_filepath=None, coeff_filepath=None, target_name='ILI', models=None, color_dict=None, start_period=None, end_period=None, alpha_dict=None, output_filename=None, ext='png', mode='save', n_coeff=20, cmap_color='RdBu_r', error_type='Error', vmin=-1, vmax=1, font_path=None):

		if font_path:
			from matplotlib import font_manager
			prop = font_manager.FontProperties(fname=font_path)

		if color_dict is None:
			color_dict = dict(zip(models, [tuple(np.random.random(3)) for mod in models]))
		if alpha_dict is None:
			alpha_dict = dict(zip(models, [1 for mod in models]))

		series_df = pd.read_csv(series_filepath, index_col=0)
		coeff_df = pd.read_csv(coeff_filepath, index_col=0)

		if start_period is None:
			start_period = series_df.index[0]
		if end_period is None:
			end_period = series_df.index[-1]

		series_df = series_df[start_period:end_period]
		coeff_df = coeff_df[start_period:end_period]

		target = series_df[target_name].values
		series = {}
		errors = {}

		for mod in models:
			series[mod] = series_df[mod].values
			errors[mod] = np.abs(target - series[mod])

		indices = list(series_df[target_name].index.values)

		#plotting target
		f, axarr = plt.subplots(3,2, gridspec_kw = {'height_ratios':[2,1,3], 'width_ratios':[16,1]})

		axarr[0,0].fill_between(x=list(range(len(indices))),y1=target, facecolor='gray', alpha=0.5, label=target_name)

		#plotting series
		for mod in models:
			axarr[0,0].plot(series[mod], label=mod, color=color_dict[mod], alpha=alpha_dict[mod])
			axarr[1,0].plot(errors[mod], color=color_dict[mod], alpha=alpha_dict[mod])

		if n_coeff is None:
			n_coeff = coeff_df.shape[1]

		means = coeff_df.mean(axis=0)
		coeff_names = list(coeff_df)
		ordered_names = [ name for v, name in sorted(zip(means, coeff_names), key=lambda x: x[0], reverse=True)]
		coeff_df = coeff_df[ordered_names[:n_coeff]]

		sns.heatmap(coeff_df.T, vmin=vmin, vmax=vmax, cmap=cmap_color, center=None, \
		robust=False, annot=None, fmt='.2g', annot_kws=None, linewidths=0,\
		linecolor='white', cbar=True, cbar_kws=None, cbar_ax=axarr[2,1], square=False,\
		xticklabels='auto', yticklabels=True, mask=None, ax=axarr[2,0])

		plt.gcf().set_size_inches([10, int(n_coeff/2)])

		plt.sca(axarr[0,0])
		plt.legend(frameon=False, ncol=len(models))
		plt.xlim([0, len(indices)])
		plt.ylim(bottom=0)
		plt.xticks(range(len(indices)),indices, rotation=0)
		plt.gca().xaxis.set_major_formatter(IndexFormatter(indices))
		plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(6))
		plt.gca().set_xticklabels([])
		plt.grid(linestyle = 'dotted', linewidth = .6)


		plt.sca(axarr[1,0])
		plt.xlim([0, len(indices)])
		plt.xticks(range(len(indices)),indices, rotation=0)
		plt.gca().xaxis.set_major_formatter(IndexFormatter(indices))
		plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(6))
		plt.gca().set_xticklabels([])
		plt.grid(linestyle = 'dotted', linewidth = .6)

		plt.sca(axarr[0,1])
		plt.axis('off')
		plt.sca(axarr[1,1])
		plt.axis('off')

		plt.sca(axarr[2,0])
		plt.xticks(range(len(indices)),indices, rotation=0)
		plt.gca().xaxis.set_major_formatter(IndexFormatter(indices))
		plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(6))
		plt.gca().set_yticklabels(ordered_names[:n_coeff], fontproperties=prop)

		# STYLE

		axarr[0,0].spines['right'].set_visible(False)
		axarr[0,0].spines['top'].set_visible(False)
		axarr[1,0].spines['right'].set_visible(False)
		axarr[1,0].spines['top'].set_visible(False)

		axarr[0,0].set_ylabel(target_name)
		axarr[1,0].set_ylabel(error_type)
		plt.subplots_adjust(left=.2, bottom=.1, right=.95, top=.9, wspace=.05, hspace=.20)



		if mode == 'show':
			plt.show()
		elif mode == 'save':
			if output_filename is None:
				output_filename = '{0}_coefficients'.format(model)
				plt.savefig('{0}/{1}/{2}.{3}'.format(self.folder_main, id_, output_filename, ext), format=ext)
			else:
				plt.savefig(output_filename+'.{0}'.format(ext), format=ext)
		plt.close()


	def plot_coefficients(self, id_=None, model=None, coefficients_filepath=None, cmap_color='RdBu_r',\
	 n_coeff=None, filename='_coefficients.csv', output_filename=None, ext='png', mode='show'):

		if coefficients_filepath:
			coefficients = pd.read_csv(coefficients_filepath, index_col=0)
		else:
			coefficients = pd.read_csv('{0}/{1}/{2}'.format(self.folder_main, id_, model), index_col=0)

		coefficients.fillna(0)




		if n_coeff is None:
			n_coeff = coefficients.shape[1]
		means = coefficients.mean(axis=0)
		coeff_names = list(coefficients)
		ordered_names = [ name for v, name in sorted(zip(means, coeff_names), key=lambda x: x[0], reverse=True)]
		coefficients = coefficients[ordered_names[:n_coeff]]

		sns.heatmap(coefficients.T, vmin=None, vmax=None, cmap=cmap_color, center=None, \
		robust=False, annot=None, fmt='.2g', annot_kws=None, linewidths=0,\
		linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False,\
		xticklabels='auto', yticklabels=True, mask=None, ax=None)


		plt.gcf().set_size_inches([10, int(n_coeff/3)])
		if mode == 'show':
			plt.show()
		elif mode == 'save':
			if output_filename is None:
				output_filename = '{0}_coefficients'.format(model)
				plt.savefig('{0}/{1}/{2}.{3}'.format(folder_main, id_, output_filename, ext), format=ext)
			else:
				plt.savefig(output_filename+'.{0}'.format(ext), format=ext)
		plt.close()

	def inter_group_lollipop_comparison(ids_dict, path_dict, metric, period, models, benchmark, color_dict=None, alpha_dict=None, metric_filename='metrics.csv', bar_separation_multiplier=1.5, mode='show', output_filename='LollipopTest', plot_domain=None, ext='png'):

	    """
	    Plots the ratio of the metric score for each of the models against a benchmark in a lollipop plot to compare between experiments.

	    Parameters
	    __________

	    ids_dict: dict
	    	Dictionary containing the list of ids for each experiment
	    path_dict: dict
	        Dictionary containing the path to the experiment folders (must coincide with the keys of ids_dict)
	    metric: str
	    	String containing the name of the metric to look for in the predictions file
	    period: str
	    	Column name containing the values to plot
	    models: List, optional (default None)
	    	String list containing the names of the models to plot
	    benchmark: str
	        The name within "models" which will serve as the benchmark
	    color_dict : dict
	    	Dictionary containing specific colors for the models to plot
	    metric_filename : str, optional (default metrics.csv)
	    mode : str, optional (default is 'save')
	     	If 'save', then function saves plot on the id_ specific folder.
	    	if 'show, then function plots and used plt.show to pop up the plot real time'
	    alpha_dict : dict, optional (default is None)
	    	dictionary specifying the opacity of the bars in the plot (alpha argument in matplotlib).
	    	If set to None, then all opacities are set to 1
	    output_filename : str, optional (default is None)
	    	If set to None, output_filename is set metricname_barplot
	    ext : str, optional (default is png)
	    	Extension formal to save the barplot.
	    plot_domain : list, optional (default is [0,1])
	    	list of two integers that sets the limits in the plot (plt.xlim)
	    bar_separation_multiplier : float, optional (default is 1)
	    	Parameter that functions as multiplier for the separion between bars in the plot.
	    	if set to 1, then bars are plotted in locations 1,2,3... if set to 2, then 2,4,6, etc
	    """

	    fig, axarr = plt.subplots(len(ids_dict.keys()),1)
	    axes = axarr.ravel()
	    if color_dict is None:
	    	color_dict = dict(zip(models, ['b']*len(models)))
	    if alpha_dict is None:
	    	alpha_dict = dict(zip(models, [1]*len(models)))

	    for i, (experiment, folder_main) in enumerate(path_dict.items()):

	        plt.sca(axes[i])
	        ids = ids_dict[experiment]

	        values_dict = dict(zip(models, [[] for mod in models]))
	        min_val = float('inf')
	        max_val = float('-inf')
	        indices = []
	        overview_path = '{0}/{1}'.format(folder_main, '_overview')
	        for i, id_ in enumerate(ids):
	            indices.append(i*bar_separation_multiplier)
	            id_path = '{0}/{1}'.format(folder_main, id_)
	            df = pd.read_csv('{0}/{1}'.format(id_path, metric_filename))
	            df = df[df['METRIC']==metric]
	            for j, mod in enumerate(models):
	                ratio =  copy.copy(df[df['MODEL']==mod][period].values[0]/df[df['MODEL']==benchmark][period].values[0])

	                if metric in ['ERROR', 'RMSE', 'NRMSE', 'MAPE']:
	                    ratio=(1/ratio)

	                values_dict[mod].append(ratio)
	                if ratio < min_val:
	                    min_val = ratio
	                if ratio > max_val:
	                    max_val = ratio

	        bar_width = 1/len(models)
	        indices = np.array(indices)

	        for i, mod in enumerate(models):
	            heights = values_dict[mod]
	            bar_positions = indices + bar_width*i
	            (markers, stemlines, baseline) = plt.stem(bar_positions, heights, linefmt='--')
	            plt.setp(markers, marker='o', markersize=7, color=color_dict[mod], alpha=alpha_dict[mod], label=mod)
	            plt.setp(stemlines, color=color_dict[mod], linewidth=1)
	            plt.setp(baseline, visible=False)

	        # Black line
	        plt.plot([0,bar_positions[-1]], [1,1],'--',color='.6', alpha=.6)
	        plt.gca().spines['top'].set_visible(False)
	        plt.gca().spines['right'].set_visible(False)

	        if experiment == 'State':
	            ids = [id_[-3:] for id_ in ids]
	        plt.xticks(indices+bar_width*((len(models)-1)/2), ids)
	        plt.ylim([min_val*.95, max_val*1.05])
	        plt.xlim([-.3, bar_positions[-1]+.3])
	        if i == 0:
	            axes[i].legend(frameon=False, ncol=len(models))
	            plt.title('{0} barplot'.format(metric))



	    if mode == 'show':
	    	plt.show()
	    elif mode == 'save':
	    	if output_filename is None:
	    		output_filename = '{0}_barplot'.format(metric)
	    	plt.gcf().set_size_inches([6,15])
	    	plt.savefig('{0}/{1}.{2}'.format(overview_path, output_filename, ext), format=ext)
	    plt.close()

	def group_lollipop_ratio(ids, metric, period, models, benchmark, folder_main = None, color_dict=None, alpha_dict=None, metric_filename='metrics.csv', bar_separation_multiplier=1.5, mode='show', output_filename='LollipopTest', plot_domain=None, ext='png'):
		"""
		Plots the ratio of the metric score for each of the models against a benchmark in a lollipop plot.


		Parameters
		__________

		id_: str
			Identifier for the region to look for
		metric: str
			String containing the name of the metric to look for in the predictions file
		period: str
			Column name containing the values to plot
		models: List, optional (default None)
			String list containing the names of the models to plot
		benchmark: str
		    The name within "models" which will serve as the benchmark
		color_dict : dict
			Dictionary containing specific colors for the models to plot
		metric_filename : str, optional (default metrics.csv)
		mode : str, optional (default is 'save')
		 	If 'save', then function saves plot on the id_ specific folder.
			if 'show, then function plots and used plt.show to pop up the plot real time'
		alpha_dict : dict, optional (default is None)
			dictionary specifying the opacity of the bars in the plot (alpha argument in matplotlib).
			If set to None, then all opacities are set to 1
		output_filename : str, optional (default is None)
			If set to None, output_filename is set metricname_barplot
		ext : str, optional (default is png)
			Extension formal to save the barplot.
		plot_domain : list, optional (default is [0,1])
			list of two integers that sets the limits in the plot (plt.xlim)
		bar_separation_multiplier : float, optional (default is 1)
			Parameter that functions as multiplier for the separion between bars in the plot.
			if set to 1, then bars are plotted in locations 1,2,3... if set to 2, then 2,4,6, etc
		"""
		if color_dict is None:
			color_dict = dict(zip(models, ['b']*len(models)))
		if alpha_dict is None:
			alpha_dict = dict(zip(models, [1]*len(models)))

		if folder_main is None:
		    folder_main = self.folder_main
		values_dict = dict(zip(models, [[] for mod in models]))
		min_val = float('inf')
		max_val = float('-inf')
		indices = []
		overview_path = '{0}/{1}'.format(folder_main, '_overview')
		for i, id_ in enumerate(ids):
			indices.append(i*bar_separation_multiplier)
			id_path = '{0}/{1}'.format(folder_main, id_)
			df = pd.read_csv('{0}/{1}'.format(id_path, metric_filename))
			df = df[df['METRIC']==metric]

			for j, mod in enumerate(models):
			    ratio =  copy.copy(df[df['MODEL']==mod][period].values[0]/df[df['MODEL']==benchmark][period].values[0])

			    if metric in ['ERROR', 'RMSE', 'NRMSE', 'MAPE']:
			        ratio=(1/ratio)

			    values_dict[mod].append(ratio)
			    if ratio < min_val:
			        min_val = ratio
			    if ratio > max_val:
			        max_val = ratio

		bar_width = 1/len(models)
		indices = np.array(indices)

		for i, mod in enumerate(models):
		    heights = values_dict[mod]
		    bar_positions = indices + bar_width*i
		    (markers, stemlines, baseline) = plt.stem(bar_positions, heights, linefmt='--')
		    plt.setp(markers, marker='o', markersize=7, color=color_dict[mod], alpha=alpha_dict[mod], label=mod)
		    plt.setp(stemlines, color=color_dict[mod], linewidth=1)
		    plt.setp(baseline, visible=False)

		# Black line
		plt.plot([0,bar_positions[-1]], [1,1],'--',color='.6', alpha=.6)
		plt.gca().spines['top'].set_visible(False)
		plt.gca().spines['right'].set_visible(False)
		plt.title('{0} barplot'.format(metric))
		plt.xticks(indices+bar_width*((len(models)-1)/2), ids)
		plt.ylim([min_val*.95, max_val*1.05])
		plt.xlim([-.3, bar_positions[-1]+.3])
		plt.legend(frameon=False, ncol=len(models))


		if plot_domain:
			plt.xlim(plot_domain)

		if mode == 'show':
			plt.show()
		elif mode == 'save':
			if output_filename is None:
				output_filename = '{0}_barplot'.format(metric)
			plt.gcf().set_size_inches([6,15])
			plt.savefig('{0}/{1}.{2}'.format(overview_path, output_filename, ext), format=ext)
		plt.close()

	def inter_season_analysis(self,ids, main_folders, periods, series_names, metric = 'RMSE', filename='metrics_condensed.csv', output_filename='season_analysis', color_dict=None, alpha_dict=None, mode='save', ext='png'):

	    '''
	    Performs seasonal analysis of data based on periods decided from the user.
		The top part of the plot shows violin plots  (https://seaborn.pydata.org/generated/seaborn.violinplot.html)
		and display the model's metric scores in a boxplot/distribution schemeself.

		Bottom part shows a heatmap representing the distribution of ranking along all periods. I.E. If Each timeseries
		case contain 4 periods and there's 4 cases, the total number of periods is 4*4 = 16. Each period has a metric for each model.
		inter_season_analysis compare this metric within the period and ranks the models from first to nth place, and each place generats a +1
		count onto the heatmap in the column representing the model and the row representing the rank.
	    __________
	    ids : dict
	        The dict of lists containing the identifiers for the regions.
	    main_folders : dict
	        The path to the experiments. Dictionary keys have to be consistent with the ids keys

	    periods : list
	        list containing the periods (should be available within the metrics table)

	    filename : str
	        String containing the filename to read the series from (using pandas).

	    start_period : str,
	        timeseries Pandas dataframe starting index.

	    end_period : str
	        timeseries ending index in the pandas dataframe.
	        n_col : int, optional (default is one)

	    series_names : list, optional (default is None)
	        Names of the timeseries to plot. If set to None, then model plots all of them.

	    output_filename : str, optional (default is series)
	        Name of the graphics file containing the plots.

	    color_dict : dict
	        Dictionary containing specific colors for the models to plot.

	    mode : str, optional (default is 'save')
	        If 'save', then function saves plot on the id_ specific folder.
	        if 'show, then function plots and used plt.show to pop up the plot real time'.

	    alpha_dict : dict, optional (default is None)
	        dictionary specifying the opacity of the bars in the plot (alpha argument in matplotlib).
	        If set to None, then all opacities are set to 1.

	    ext : str, optional (default is png)
	        Extension formal to save the graphics file.

	    Defines the number of columns to define the plotting array. Function organizes
	    plots in n_col then
	    '''

	    default_colors = ['royalblue', 'darkorange', 'forestgreen', 'firebrick']
	    if color_dict is None:
	        color_dict = dict(zip(series_names, default_colors[0:len(series_names)]))



	    score_periods = {}
	    ranks = {}
	    for title, ids_ in ids.items():

	        metrics_df = pd.read_csv(main_folders[title] + '/_overview/'+ filename)
	        score_periods[title] = []
	        ranks[title] = getRanks(metrics_df, metric, ids_, series_names, periods)

	        for mod in series_names:
	            score_periods[title].append(get_all_periods(metrics_df, mod, metric, periods))

	        score_periods[title] = pd.DataFrame(np.hstack(score_periods[title]), columns=series_names)

	    f, axarr = plt.subplots(2, len(ids.keys()))
	    axes = axarr.ravel()

	    places_dict = get_places(ranks, series_names)
	    places = ['4th', '3rd', '2nd', '1st']


	    for i, title in enumerate(ids.keys()):
	        places_list = places_dict[title]
	        sns.violinplot(data=score_periods[title], ax=axes[i], cut=0, inner='box')
	        '''
	        sns.heatmap(data=ranks[metric], ax=axes[i+len(ids.keys())], cmap='Reds', cbar=False, annot=True)
	        axes[i+len(ids.keys())].set_yticklabels(['1th', '2th', '3th', '4th', '5th'], rotation='horizontal')
	        axes[i+len(ids.keys())].set_xticklabels(series_names, rotation='horizontal')
	        '''
	        print(title, i)

	        for j, ord_list in enumerate(reversed(places_list)):

	            for (mod, height) in ord_list:
	                axes[i+len(ids.keys())].barh(j, height, color=color_dict[mod])

	        plt.sca(axes[i+len(ids.keys())])
	        plt.yticks(range(len(places_list)), places)
	        axes[i].set_title(title)
	        axes[i].set_xticklabels(series_names)

	        if i == 0:
	            axes[i+len(ids.keys())].set_xlabel('No. of States')
	        elif i == 1:
	            axes[i+len(ids.keys())].set_xlabel('No. of Regions')
	        elif i == 2:
	            axes[i+len(ids.keys())].set_xlabel('No. of Countries')

	        if i == 0:
	            axes[i].set_ylabel('{0}'.format(metric))
	            axes[+len(ids.keys())].set_ylabel('Ranking Proportions')
	    if mode == 'show':
	        plt.show()
	    if mode == 'save':
	        plt.gcf().set_size_inches([9, 5])
	        plt.subplots_adjust(left=.1, bottom=.12, right=.97, top=.91, wspace=.25, hspace=.20)
	        plt.savefig('{0}/{1}/{2}.{3}'.format(self.folder_main, OVERVIEW_FOLDER, output_filename,ext),fmt=ext)
	    plt.close()
	    return

	def group_seriesbars(self, ids=None, start_period=None, end_period=None, series_names=None, folder_dir=None, metric_filename='metrics.csv', preds_filename='preds.csv', output_filename='series', color_dict=None, alpha_dict=None, mode='show', ext='png', n_col=1, width_ratios=[6,1], metric=None, metric_period=None, target_name=None):
		default_colors = ['g', 'b', 'r', 'indigo']
		default_linewidths = [1.5,1.4,1.6,1]
		'''
		Gathers information from all region and does a group plot using matplotlib, along with a barplot, showing a metric.
		regions are ordered based on the original ordering from the ids list from left to right, top to bottom
		Parameters
		__________
		ids : list
			The list containing the identifiers for the regions.

		preds_filename : str
			String containing the preds_filename to read the series from (using pandas).

		start_period : str,
			 timeseries Pandas dataframe starting indices.

		end_period : str
			timeseries ending indices in the pandas dataframe.
		n_col : int, optional (default is one)

		series_names : list, optional (default is None)
			Names of the timeseries to plot. If set to None, then model plots all of them.

		output_preds_filename : str, optional (default is series)
			Name of the graphics file containing the plots.

		color_dict : dict
			Dictionary containing specific colors for the models to plot.

		mode : str, optional (default is 'save')
		 	If 'save', then function saves plot on the id_ specific folder.
			if 'show, then function plots and used plt.show to pop up the plot real time'.

		alpha_dict : dict, optional (default is None)
			dictionary specifying the opacity of the bars in the plot (alpha argument in matplotlib).
			If set to None, then all opacities are set to 1.

		ext : str, optional (default is png)
			Extension formal to save the graphics file.


			Defines the number of columns to define the plotting array. Function organizes
			plots in n_col then
		'''
		if not ids:
			ids = self.ids

		if folder_dir is None:
			folder_dir = self.folder_main

		n_ids = len(ids)
		n_rows = math.ceil(n_ids/n_col)


		fig, axarr = plt.subplots(n_rows,n_col*2, gridspec_kw = {'width_ratios':width_ratios*n_col})
		axes = axarr.ravel()

		if color_dict is None:
		    color_dict = {}
		    for i, mod in enumerate(series_names):
			          color_dict[mod] = default_colors[i]
		if alpha_dict is None:
		    alpha_dict = {}
		    for i, mod in enumerate(series_names):
			          alpha_dict[mod] = .8


		for i, id_ in enumerate(ids):
		    df = pd.read_csv('{0}/{1}/{2}'.format(folder_dir, id_, preds_filename), index_col=[0])
		    metric_df = pd.read_csv('{0}/{1}/{2}'.format(folder_dir, id_, metric_filename))
		    series = []


		    indices = copy.copy(df[start_period:end_period].index.values)
		    for kk in range(np.size(indices)):
		        v = indices[kk][2:7]
		        indices[kk] = v


		    col_names = list(df)
		    if target_name:
		        zeros=np.zeros(np.size(df[start_period:end_period][target_name].values))
		        curve_max = np.amax(np.size(df[start_period:end_period][target_name].values))
		        #axes[i*2].plot(df[start_period:end_period][target_name].values, label=target_name, linewidth=.1)
		        axes[i*2].fill_between(x=list(range(len(indices))),y1=df[start_period:end_period][target_name].values, facecolor='gray', alpha=0.5, label=target_name)
		    for k, col in enumerate(series_names):
		    	if col in col_names:
		    		# create top panel
		    		axes[i*2].plot(df[start_period:end_period][col].values, label=col, linewidth=default_linewidths[k])
		    	else:
		    		print('WARNING! {0} not in {1} timeseries list'.format(col, id_))

		    if color_dict:
		        for j, l in  enumerate(axes[i*2].get_lines()):
		            l.set_color(color_dict[series_names[j]])

		    if alpha_dict:
		        for j, l in  enumerate(axes[i*2].get_lines()):
		            l.set_alpha(alpha_dict[series_names[j]])



		    ######
		    metric_df = metric_df[metric_df['METRIC']==metric][['MODEL', metric_period]]

		    bar_width = .5
		    hs = []
		    for k, mod in enumerate(series_names):
		        heights = metric_df[metric_df['MODEL'] == mod][metric_period].values
		        bar_positions = k
		        rects = axes[i*2+1].bar(bar_positions, heights, bar_width, label=mod, color=color_dict[mod], alpha=alpha_dict[mod])
		        hs.append(copy.copy(heights))
		    max_height = np.amax(hs)
		    min_height = np.amin(hs)
		    axes[i*2+1].set_ylim([min_height*.90, max_height*1.1])
		    axes[i*2+1].set_yticks([min_height, max_height])
		    axes[i*2+1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
		    #####
		    if i == 0:
		        if target_name:
		            n_cols = len(series_names)+1
		        else:
		            n_cols = len(series_names)
		        axes[i*2].legend(ncol=n_cols, frameon=False, loc='upper left', \
		        bbox_to_anchor=(.0,1.20))

		    axes[i*2].text(.10,.9, id_, weight = 'bold', horizontalalignment='left', transform=axes[i*2].transAxes)
		    #axes[i*2+1].yaxis.set_major_locator(mticker.MaxNLocator(2))
		    axes[i*2].yaxis.set_major_locator(mticker.MaxNLocator(2))
		    axes[i*2+1].set_xticks([])

		    # SPINES
		    axes[i*2].spines['top'].set_visible(False)
		    axes[i*2].spines['right'].set_visible(False)

		    #axes[i*2].spines['left'].set_visible(False)
		    yticks=axes[i*2].get_yticks()
		    ylim = axes[i*2].get_ylim()
		    axes[i*2].spines['left'].set_bounds(0,yticks[2])
		    axes[i*2+1].spines['left'].set_bounds(min_height,max_height)
		    axes[i*2].set_ylim(0,ylim[1])
		    axes[i*2+1].spines['top'].set_visible(False)
		    axes[i*2+1].spines['right'].set_visible(False)
		    #axes[i*2+1].spines['left'].set_visible(False)

		    if i == 0:
		        plt.ylabel('Estimates')

		    if i > n_col*(n_rows - 1)-1:
		        axes[i*2].set_xlabel('Date')

		        plt.sca(axes[i*2])
		        plt.xticks(range(len(indices)),indices, rotation=0)
		        plt.gca().xaxis.set_major_formatter(IndexFormatter(indices))
		        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(4))
		        xticks = axes[i*2].get_xticks()
		        axes[i*2].spines['bottom'].set_bounds(xticks[1], xticks[-2])
		    else:
		        plt.sca(axes[i*2])
		        plt.xticks(range(len(indices)),indices, rotation=0)
		        plt.gca().xaxis.set_major_formatter(IndexFormatter(indices))
		        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(4))
		        xticks = axes[i*2].get_xticks()
		        axes[i*2].spines['bottom'].set_bounds(xticks[1], xticks[-2])
		        #axes[i*2].set_xticklabels([])



		if i < np.size(axes)/2-1:
		    for j in range(i+1,int(np.size(axes)/2)):
		        axes[j*2+1].spines['top'].set_visible(False)
		        axes[j*2+1].spines['right'].set_visible(False)
		        axes[j*2+1].spines['left'].set_visible(False)
		        axes[j*2+1].spines['bottom'].set_visible(False)
		        axes[j*2].spines['top'].set_visible(False)
		        axes[j*2].spines['right'].set_visible(False)
		        axes[j*2].spines['left'].set_visible(False)
		        axes[j*2].spines['bottom'].set_visible(False)
		        axes[j*2].set_yticks([])
		        axes[j*2].set_xticks([])
		        axes[j*2+1].set_yticks([])
		        axes[j*2+1].set_xticks([])
		        axes[j*2].set_title('')
		        axes[j*2+1].set_title('')



		plt.subplots_adjust(left=.03, bottom=.05, right=.99, top=.95, wspace=.25, hspace=.15)
		if mode == 'show':
			plt.show()
			plt.close()
		if mode == 'save':
			fig.set_size_inches([7*n_col,2.5*n_rows])
			plt.savefig('{0}/{1}/{2}.{3}'.format(folder_dir, OVERVIEW_FOLDER, output_filename,ext),fmt=ext)

		plt.close()

	def rank_ids_by_metric(self, ids, models, period, metric='RMSE', reverse=False, metric_filename='metrics.csv'):
		'''
		rank_ids_by_metric compares the performance of two models specified in the models list and
		the selected metric. Function substracts model[0] from model[1] (i.e. model[1]-model[0]) and orders
		the results based on decreasing order.
		Parameters
		__________

		ids : list
		List of strings containing the region identifiers to rank.

		models : list
		A list of two models to compare

		metric : str, optional (default is RMSE)
		The metric to use as comparison

		order : Boolean, optional (default is False)
		If False, orders in increasing order. If set to True, orders in decreasing order

		metric_filename : str, optionall (default is 'metric.csv')

		period : str

		Specify the period of the metric


		Returns
		_______
		ids = An ordered list of IDs based on the results of the comparison
		'''

		metric_values = []
		for id_ in ids:

			metric_df = pd.read_csv('{0}/{1}/{2}'.format(self.folder_main, id_, metric_filename))

			mod0_val = metric_df[ (metric_df['METRIC'] == metric) & (metric_df['MODEL'] == models[0])][period].values
			mod1_val = metric_df[(metric_df['METRIC'] == metric) & (metric_df['MODEL'] == models[1])][period].values

			ratio = mod0_val/mod1_val

			if metric in ['RMSE', 'NRMSE', 'ERROR', 'MAPE']:
				ratio = 1/ratio
			metric_values.append(copy.copy(ratio))

		ord_values = []
		ord_ids = []
		for id_, val in sorted(zip(ids, metric_values), key = lambda x : x[1], reverse=reverse):
			ord_values.append(val)
			ord_ids.append(id_)

		return ord_ids

	def group_weekly_winner(self, ids=None, cmap='BuPu', models=None, start_period=None, end_period=None, output_filename='weekly_winners', folder_main=None, filename='preds.csv', mode='show', ext='png'):
		"""
		Fir each ID, chooses the weekly winner out of the models list in a prediction file and plots all of them
		together in heatmap.
		Parameters
		__________

		ids : list
		The list containing the identifiers for the regions.

		filename : str
		String containing the filename to read the series from (using pandas).

		start_period : str,
		 timeseries Pandas dataframe starting index.

		end_period : str
		timeseries ending index in the pandas dataframe.

		output_filename : str, optional (default is series)
		Name of the graphics file containing the plots.

		mode : str, optional (default is 'save')
			If 'save', then function saves plot on the id_ specific folder.
		if 'show, then function plots and used plt.show to pop up the plot real time'.

		ext : str, optional (default is png)
		Extension formal to save the graphics file.

		cmap : str, optional (default is 'BuPu')
		colormap style to display in the plot. List of colormaps is provided by Matplotlib.
		folder_main : str, optiona (default is None)
		Path to folder with data. If None, uses default class attribute.
		"""
		if folder_main is None:
			folder_main = self.folder_main


		#Getting winners in each id
		winners_dict = {}
		ind = list(range(len(models)))
		map_dict =dict(zip(models, ind))

		for i, id_ in enumerate(ids):

			df = pd.read_csv('{0}/{1}/{2}'.format(folder_main, id_, filename), index_col=[0])

			if i == 0:
				if start_period is None:
					start_period = df.index[0]
				if end_period is None:
					end_period = df.index[-1]

			df = df[start_period:end_period]
			winners = get_winners_from_df(df, models=models)

			winners=winners.replace({"winners" : map_dict })
			winners_dict[id_] = winners['winners'].values

		index = df[start_period:end_period].index.values
		winners_df = pd.DataFrame(winners_dict, index=index)


		ax= sns.heatmap(data=winners_df.transpose(), linewidths=.6, yticklabels=True, cbar_kws={"ticks":ind})
		ax.collections[0].colorbar.ax.set_yticklabels(models)

		#plt.matshow(winners_df.transpose(), origin='lower', aspect='auto', cmap='BuPu')
		#cb = plt.colorbar(orientation='vertical', ticks=ind, shrink=.5)
		#cb.ax.set_yticklabels(models)

		#plt.xticks(range(len(index)),index, rotation=45)
		#plt.gca().xaxis.set_major_formatter(IndexFormatter(index))
		#plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(6))


		if mode == 'show':
			plt.show()
			plt.close()
		if mode == 'save':
			plt.gcf().set_size_inches([10,6])
			plt.subplots_adjust(left=.10, bottom = .15, right = 1, top=.95, wspace=.20, hspace=.20)
			plt.savefig('{0}/{1}/{2}.{3}'.format(self.folder_main, self.overview_folder, output_filename, ext),fmt=ext)
		plt.close()

	def plot_series(self,folder_dir=None, id_=None, filename=None, output_filename='series', series_names=None, color_dict=None, alpha_dict=None, start_period=None, end_period=None, mode='save', ext='png', add_weekly_winner=False, winner_models=None):

		if folder_dir is None:
			folder_dir = self.folder_main
		if filename is None:
			filename = ID_PREDS

		df = pd.read_csv('{0}/{1}/{2}'.format(self.folder_main, id_, filename), index_col=[0])

		if start_period is None:
			start_period = df.index[0]
		if end_period is None:
			end_period =  df.index[-2]

		series = []
		index = df.index.values

		if add_weekly_winner:
			n_rows = 2
			gridspec_kw = {'height_ratios':[6,1]}
		else:
			n_rows = 1
			gridspec_kw = None

		fig, axes = plt.subplots(n_rows, 1, gridspec_kw = gridspec_kw)
		col_names = list(df)

		if series_names is None:
			series_names = col_names

		for col in series_names:
			# create top panel
			axes[0].plot(df[start_period:end_period][col].values, label=col)
			#a = ax.plot_date(x=dates, y=ILI) # fmt="-",color='.20', linewidth=3.2, label='ILI', alpha = 1)

		if color_dict:
			for i, l in  enumerate(axes[0].get_lines()):
				l.set_color(color_dict[series_names[i]])
		if alpha_dict:
			for i, l in  enumerate(axes[0].get_lines()):
				l.set_alpha(alpha_dict[series_names[i]])


		if add_weekly_winner:
			winners = get_winners_from_df(df, models=winner_models)
			ind = list(range(len(winner_models)))
			map_dict =dict(zip(winner_models, ind))
			winners=winners.replace({"winners" : map_dict })
			im = axes[1].matshow(winners['winners'].values.reshape([1,-1]), origin='lower', aspect='auto', cmap='BuPu')
			cb = plt.colorbar(im, ax=axes[1], orientation='horizontal', ticks=ind)
			cb.ax.set_xticklabels(winner_models)

		axes[0].legend(ncol=len(series_names), frameon=False)
		axes[0].set_title('{0}'.format(id_))
		axes[0].set_ylabel('Estimates')
		axes[0].set_xlabel('Index')
		axes[0].spines['top'].set_visible(False)
		axes[0].spines['right'].set_visible(False)
		plt.xticks(range(len(index)),index, rotation=45)
		axes[0].xaxis.set_major_formatter(IndexFormatter(index))
		axes[0].xaxis.set_major_locator(mticker.MaxNLocator(6))
		axes[1].set_xticks([])
		axes[1].set_yticks([])
		axes[0].autoscale(enable=True, axis='x', tight=True)


		#plt.locator_params(nbins=8)

		if mode == 'show':
			plt.show()
			plt.close()
		if mode == 'save':
			fig.set_size_inches([10,5])
			plt.savefig('{0}/{1}/{2}.{3}'.format(self.folder_main, id_, output_filename,ext),fmt=ext)

		plt.close()

	def season_analysis(self, ids, periods, series_names, folder_main=None, metrics = ['PEARSON', 'NRMSE'], filename='metrics_condensed.csv', output_filename='season_analysis', color_dict=None, alpha_dict=None, mode='save', ext='png'):

		'''
		Gathers information from all region and does a group plot using matplotlib.
		regions are ordered in based on the original ordering from the ids list from left to right, top to bottom
		Parameters
		__________
		ids : list
			The list containing the identifiers for the regions.

		periods : list
			list containing the periods (should be available within the metrics table)

		filename : str
			String containing the filename to read the series from (using pandas).

		start_period : str,
			timeseries Pandas dataframe starting index.

		end_period : str
			timeseries ending index in the pandas dataframe.
			n_col : int, optional (default is one)

		series_names : list, optional (default is None)
			Names of the timeseries to plot. If set to None, then model plots all of them.

		output_filename : str, optional (default is series)
			Name of the graphics file containing the plots.

		color_dict : dict
			Dictionary containing specific colors for the models to plot.

		mode : str, optional (default is 'save')
			If 'save', then function saves plot on the id_ specific folder.
			if 'show, then function plots and used plt.show to pop up the plot real time'.

		alpha_dict : dict, optional (default is None)
			dictionary specifying the opacity of the bars in the plot (alpha argument in matplotlib).
			If set to None, then all opacities are set to 1.

		ext : str, optional (default is png)
			Extension formal to save the graphics file.

		Defines the number of columns to define the plotting array. Function organizes
		plots in n_col then
		'''
		if not folder_main:
		    folder_main = self.folder_main

		metrics_df = pd.read_csv(folder_main + '/_overview/'+ filename)

		score_periods = {}
		ranks = {}
		for metric in metrics:
		    score_periods[metric] = []
		    ranks[metric] = getRanks(metrics_df, metric, ids, series_names, periods)
		    for mod in series_names:
		        score_periods[metric].append(get_all_periods(metrics_df, mod, metric, periods))

		    score_periods[metric] = pd.DataFrame(np.hstack(score_periods[metric]), columns=series_names)

		f, axarr = plt.subplots(2, len(metrics))

		axes = axarr.ravel()
		for i, metric in enumerate(metrics):
		    sns.violinplot(data=score_periods[metric], ax=axes[i], cut=0)
		    sns.heatmap(data=ranks[metric], ax=axes[i+2], cmap='Reds', cbar=False, annot=True)
		    axes[i].set_title(metric)
		    axes[i+2].set_yticklabels(['1th', '2th', '3th', '4th', '5th'], rotation='horizontal')
		    axes[i+2].set_xticklabels(series_names, rotation='horizontal')
		if mode == 'show':
			plt.show()
		if mode == 'save':
			plt.gcf().set_size_inches([7, 4])
			plt.subplots_adjust(left=.08, bottom=.09, right=.97, top=.91, wspace=.25, hspace=.20)
			plt.savefig('{0}/{1}/{2}.{3}'.format(self.folder_main, OVERVIEW_FOLDER, output_filename,ext),fmt=ext)

		plt.close()

	def group_plot_series(self, ids=None, start_period=None, end_period=None, series_names=None, folder_dir=None, filename='preds.csv', output_filename='series', color_dict=None, alpha_dict=None, mode='save', ext='png', n_col=1):
		'''
		Gathers information from all region and does a group plot using matplotlib.
		regions are ordered in based on the original ordering from the ids list from left to right, top to bottom
		Parameters
		__________
		ids : list
			The list containing the identifiers for the regions.

		filename : str
			String containing the filename to read the series from (using pandas).

		start_period : str,
			 timeseries Pandas dataframe starting index.

		end_period : str
			timeseries ending index in the pandas dataframe.
		n_col : int, optional (default is one)

		series_names : list, optional (default is None)
			Names of the timeseries to plot. If set to None, then model plots all of them.

		output_filename : str, optional (default is series)
			Name of the graphics file containing the plots.

		color_dict : dict
			Dictionary containing specific colors for the models to plot.

		mode : str, optional (default is 'save')
		 	If 'save', then function saves plot on the id_ specific folder.
			if 'show, then function plots and used plt.show to pop up the plot real time'.

		alpha_dict : dict, optional (default is None)
			dictionary specifying the opacity of the bars in the plot (alpha argument in matplotlib).
			If set to None, then all opacities are set to 1.

		ext : str, optional (default is png)
			Extension formal to save the graphics file.


			Defines the number of columns to define the plotting array. Function organizes
			plots in n_col then
		'''

		if folder_dir is None:
			folder_dir = self.folder_main

		n_ids = len(ids)
		n_rows = math.ceil(n_ids/n_col)

		fig, axarr = plt.subplots(n_rows,n_col)
		axes = axarr.ravel()


		for i, id_ in enumerate(ids):
			df = pd.read_csv('{0}/{1}/{2}'.format(self.folder_main, id_, filename), index_col=[0])

			series = []
			index = df[start_period:end_period].index.values
			col_names = list(df)
			for col in series_names:
				if col in col_names:
					# create top panel
					axes[i].plot(df[start_period:end_period][col].values, label=col)
				else:
					print('WARNING! {0} not in {1} timeseries list'.format(col, id_))

			if color_dict:
				for j, l in  enumerate(axes[i].get_lines()):
					l.set_color(color_dict[series_names[j]])
			if alpha_dict:
				for j, l in  enumerate(axes[i].get_lines()):
					l.set_alpha(alpha_dict[series_names[j]])

			if i == 0:
				axes[i].legend(ncol=len(series_names), frameon=False, loc='upper left', \
				bbox_to_anchor=(.0,1.20))

			axes[i].text(.80,.9, id_, weight = 'bold', horizontalalignment='left', transform=axes[i].transAxes)
			axes[i].spines['top'].set_visible(False)
			axes[i].spines['right'].set_visible(False)

			if i%n_col == 0:
				plt.ylabel('Estimates')

			if i > n_col*(n_rows - 1)-1:
				time.sleep(3)
				axes[i].set_xlabel('Index')
				plt.sca(axes[i])
				plt.xticks(range(len(index)),index, rotation=45)
				plt.gca().xaxis.set_major_formatter(IndexFormatter(index))
				plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(6))
				#plt.locator_params(nbins=8)
			else:
				axes[i].set_xticks([])

		if mode == 'show':
			plt.show()
			plt.close()
		if mode == 'save':
			fig.set_size_inches([5*n_col,3*n_rows])
			plt.savefig('{0}/{1}/{2}.{3}'.format(self.folder_main, OVERVIEW_FOLDER, output_filename,ext),fmt=ext)

		plt.close()

	def merge_models(filename, filename2, output_filename, models=None, start_period=None, end_period=None, erase_duplicates=True):
	    """
	    merges two dataframes for an specified period of time, substracts the models, and stores them in output_filepath.

	    PARAMETERS:
	    ___________


	    filename : str,

	    Path to first dataframe (CSV)

	    filename2 : str

	    output_filename : str,

	    New absolute location of the merged dataframe.

	    Path to second dataframe

	    models : list, optional (Default is None)

	    Name of models to let into the new Dataframe. If set to None, then lets all models in

	    start_period : str, optional (default is None)

	    First index in dataframe to merge, if set to None, then grabs the first index of filename's dataframe

	    end_period : str, optional (default is None)

	    """

	    df1 = pd.read_csv(filename, index_col = [0])
	    df2 = pd.read_csv(filename2, index_col = [0])

	    df3 = pd.concat([df1,df2], axis=1)

	    if start_period and (start_period in df3.index):
	        pass
	    elif start_period is None:
	        start_period = df3.index[0]
	    else:
	        print('Unable to identify start_period {0} as valid start reference.\
			please review'.format(start_period))
	        return

	    if end_period and end_period in df3.index:
	    	pass
	    elif end_period is None:
	    	end_period = df3.index[-1]
	    else:
	    	print('Unable to identify end_period {0} as valid start reference.\
	    	please review'.format(start_period))
	    	return


	    if models is None:
	        models = df3.columns

	    df3 = df3[start_period:end_period][models]
	    if erase_duplicates:
	        df3=df3.T.drop_duplicates().T
	    df3.to_csv(output_filename)

	def group_barplot_metric(self, ids, metric, period, models, color_dict=None, alpha_dict=None, metric_filename='metrics.csv', bar_separation_multiplier=1.5, mode='save', output_filename=None, plot_domain=None, ext='png', show_values=False, ordering=None):
		"""
		Produces a bar plot of the desired metric and models for an specific ids.
		If looking to make a id group plot, please check group_metric_bargraph()
		Parameters
		__________

		id_: str
			Identifier for the region to look for
		metric: str
			String containing the name of the metric to look for in the predictions file
		period: str
			Column name containing the values to plot
		models: List, optiona (default None)
			String list containing the names of the models to plot
		color_dict : dict
			Dictionary containing specific colors for the models to plot
		metric_filename : str, optional (default metrics.csv)
		mode : str, optional (default is 'save')
		 	If 'save', then function saves plot on the id_ specific folder.
			if 'show, then function plots and used plt.show to pop up the plot real time'
		alpha_dict : dict, optional (default is None)
			dictionary specifying the opacity of the bars in the plot (alpha argument in matplotlib).
			If set to None, then all opacities are set to 1
		output_filename : str, optional (default is None)
			If set to None, output_filename is set metricname_barplot
		ext : str, optional (default is png)
			Extension formal to save the barplot.
		plot_domain : list, optional (default is [0,1])
			list of two integers that sets the limits in the plot (plt.xlim)
		show_values : boolean, optional (default is False)
			plots the values of the metric within the barplot.
		bar_separation_multiplier : float, optional (default is 1)
			Parameter that functions as multiplier for the separion between bars in the plot.
			if set to 1, then bars are plotted in locations 1,2,3... if set to 2, then 2,4,6, etc
		"""
		if color_dict is None:
			color_dict = dict(zip(models, ['b']*len(models)))
		if alpha_dict is None:
			alpha_dict = dict(zip(models, [1]*len(models)))

		values_dict = dict(zip(models, [[] for mod in models]))
		indices = []
		overview_path = '{0}/{1}'.format(self.folder_main, OVERVIEW_FOLDER)
		for i, id_ in enumerate(ids):
			indices.append(i*bar_separation_multiplier)
			id_path = '{0}/{1}'.format(self.folder_main, id_)
			df = pd.read_csv('{0}/{1}'.format(id_path, metric_filename))
			df = df[df['METRIC']==metric]
			for j, mod in enumerate(models):
				try:
					values_dict[mod].append(df[df['MODEL']==mod][period].values[0])
				except Exception as t:
					print(t)
					print('\n Missing data in model:{0} for id:{1}'.format(mod, id_))
					return

		bar_width = 1/len(models)
		indices = np.array(indices)

		for i, mod in enumerate(models):
			heights = values_dict[mod]
			bar_positions = indices + bar_width*i

			rects = plt.barh(bar_positions, heights, bar_width, label=mod, color=color_dict[mod], alpha=alpha_dict[mod])
			if show_values:
				for j, rect in enumerate(rects):
					yloc = bar_positions[j]
					clr = 'black'
					p = heights[j]
					xloc = heights[j]
					plt.gca().text(xloc, yloc, p, horizontalalignment='center', verticalalignment='center', color=clr, weight='bold')




		plt.gca().spines['top'].set_visible(False)
		plt.gca().spines['right'].set_visible(False)
		plt.title('{0} barplot'.format(metric))
		plt.yticks(indices+bar_width*(len(models)/2), ids)
		if len(models) > 5:
			plt.legend(frameon=False, ncol=1)
		else:
			plt.legend(frameon=False, ncol=len(models))

		if plot_domain:
			plt.xlim(plot_domain)

		if mode == 'show':
			plt.show()
		elif mode == 'save':
			if output_filename is None:
				output_filename = '{0}_barplot'.format(metric)
			plt.gcf().set_size_inches([6,15])
			plt.savefig('{0}/{1}.{2}'.format(overview_path, output_filename, ext), format=ext)
		plt.close()

	def barplot_metric(self, id_, metric, period, models=None, color_dict=None, alpha_dict=None, metric_filename='metrics.csv', bar_width=1, bar_separation_multiplier=1, mode='save', output_filename=None, plot_domain=None, ext='png', show_values=True):
		"""
		Produces a bar plot of the desired metric and models for an specific id.
		If looking to make a id group plot, please check group_metric_bargraph()
		Parameters
		__________

		id_: str
			Identifier for the region to look for
		metric: str
			String containing the name of the metric to look for in the predictions file
		period: str
			Column name containing the values to plot
		models: List, optiona (default None)
			String list containing the names of the models to plot
		color_dict : dict
			Dictionary containing specific colors for the models to plot
		metric_filename : str, optional (default metrics.csv)
		bar_width : float, optional (default is 1)
			Bar width in the plots (0 to 1, any more will make bars be on top of each other).
		bar_separation_multiplier : float, optional (default is 1)
			Parameter that functions as multiplier for the separion between bars in the plot.
			if set to 1, then bars are plotted in locations 1,2,3... if set to 2, then 2,4,6, etc
		mode : str, optional (default is 'save')
		 	If 'save', then function saves plot on the id_ specific folder.
			if 'show, then function plots and used plt.show to pop up the plot real time'
		alpha_dict : dict, optional (default is None)
			dictionary specifying the opacity of the bars in the plot (alpha argument in matplotlib).
			If set to None, then all opacities are set to 1
		output_filename : str, optional (default is None)
			If set to None, output_filename is set metricname_barplot
		ext : str, optional (default is png)
			Extension formal to save the barplot.
		plot_domain : list, optional (default is [0,1])
			list of two integers that sets the limits in the plot (plt.xlim)
		show_values : boolean, optional (default is False)
			plots the values of the metric within the barplot.
		"""


		if id_ in self.ids:
			id_path = '{0}/{1}'.format(self.folder_main, id_)

			df = pd.read_csv('{0}/{1}'.format(id_path, metric_filename))
			df = df[df['METRIC']==metric]

			if models is None:
				models = df['MODEL'].values
			if color_dict is None:
				color_dict = dict(zip(models, ['b']*len(models)))
			if alpha_dict is None:
				alpha_dict = dict(zip(models, [1]*len(models)))

			indices = []
			for i, mod in enumerate(models):
				height = df[df['MODEL']==mod][period].values[0]
				indices.append(i*bar_separation_multiplier)
				rects = plt.barh(indices[i], height, bar_width, color=color_dict[mod], alpha=alpha_dict[mod])

				if show_values:
					for rect in rects:
						yloc = indices[i]
						clr = 'black'
						p = height
						xloc = height
						plt.gca().text(xloc, yloc, p, horizontalalignment='center', verticalalignment='center', color=clr, weight='bold')


		else:
			print(' {1} ID is not in data. current ids : {0}'.format(self.ids, id_))


		plt.gca().spines['top'].set_visible(False)
		plt.gca().spines['right'].set_visible(False)
		plt.title('{0} barplot'.format(metric))
		plt.yticks(indices, models)
		plt.legend(frameon=False, ncol=len(models))


		if plot_domain:
			plt.xlim(plot_domain)

		if mode == 'show':
			plt.show()
		elif mode == 'save':
			if output_filename is None:
				output_filename = '{0}_barplot'.format(metric)
			plt.gcf().set_size_inches([10,7])
			plt.savefig('{0}/{1}.{2}'.format(id_path, output_filename, ext), format=ext)
		plt.close()

	def extract_metrics(self, ids = None, filename = None, folder_main = None, \
	metrics=['RMSE', 'PEARSON'], models= ['AR12GO'], seasons=None):

		if ids is None:
			ids = self.ids
		if folder_main is None:
			folder_main = self.folder_main+'/'+OVERVIEW_FOLDER
		if filename is None:
			filename = CSV_METRICS_CONDENSED

		df = pd.read_csv(folder_main+'/'+filename, index_col=0)

		if models is not None:
			df = df[main_df['MODEL'].isin(models)]
		if metrics is not None:
			df = df[main_df['METRIC'].isin(metrics)]

		extracted_df.to_csv(folder_main + '/metrics_extracted.csv')



	def group_compute_metrics(self, intervals, interval_labels, which_ids = 'all_ids', \
	which_metrics = ['PEARSON', 'RMSE'], remove_missing_values=[0.5, 0], input_file_name=None, output_file_name=None, \
	verbose=False, target = None, write_to_overview=False):
		'''
		For each of the ids analyzed, computes a set of metrics (metris available in the metric_handler variable).

		Input:
		 	intervals = a binary tuple list with start and end labels.
			interval_labels
			which_ids =  list that specifies which of the ids to compute the metrics for, if not, computes for all.
			which_metrics = (dictionary with per-id lists or list) Specifies which metrics to compute for the ids.
			remove_missing_values = list of float numbers to assig which values to ignore in the metric computation
		'''

		if input_file_name is None:
			input_file_name = ID_PREDS
		if output_file_name is None:
			output_file_name = ID_METRICS
		if target is None:
			target = TARGET

		if which_ids == 'all_ids':

			which_ids = self.ids
		else:

			for i, id_ in enumerate(which_ids):

				if id_ not in self.ids:
					which_ids.remove(id_)
					print('{0} not found in experiment object ids. Removing Please check.'.format(id_))



		if isinstance(intervals, list) and isinstance(intervals[0], tuple):
			print('Non-specified id intervals received. Using intervals in all ids')
			intervals = dict(which_ids, [intervals]*len(which_ids))

		elif isinstance(intervals, dict) and isinstance(interval_labels, dict):

			for i, id_ in enumerate(which_ids):
				try:

					if len(intervals[id_]) != len(interval_labels[id_]):
						print(' WARNING! Mismatch between interval and interval_labels in id: {0}.'.format(id_))
						interval_labels[id_] = []
						for i in range(0,len(interval_labels)):
							interval_labels[id_].append('s{0}'.format(i))

				except KeyError:

					print('ID not found within interval data. Please review.')
		else:
			print('Mismatch between intervals and interval labels types (Non-dictionaries). Please check input parameters')
			return

		if write_to_overview: id_dfs = []

		if verbose: print('Reading on {0}'.format(self.folder_main))

		for id_folder in os.listdir(self.folder_main):

			if id_folder in which_ids:

				file_preds = self.folder_main + '/'  + id_folder  + '/' + input_file_name
				if verbose: print('Checking for data in {0}'.format(file_preds))
				if os.path.exists(file_preds):

					preds_pd = pd.read_csv(file_preds, index_col = 0)

					if verbose:
						print('Successfully loaded preds \n \n {0}'.format(preds_pd))
						time.sleep(10)

					id_interval = intervals[id_folder]
					id_interval_labels = interval_labels[id_folder]
					model_list = list(preds_pd)
					model_list.remove(target)
					metric_dict = {}

					if verbose:
						print('id: {0} \nid_intervals: {1}\n id_interval_labels{2}\n Models_available{3}'.format(id_folder, id_interval, id_interval_labels, model_list))
						time.sleep(10)

					# generating multi index for pandas dataframe
					levels = [which_metrics, model_list]
					labels =  [[], []]
					names = ['METRIC', 'MODEL']

					for i, (start_interval, end_interval) in enumerate(id_interval):

						metric_dict[id_interval_labels[i]] = []
						sub_pd = copy.deepcopy(preds_pd[start_interval:end_interval])

						for j, metric in enumerate(which_metrics):

							for k, model in enumerate(model_list):

								model_timeseries = sub_pd[model].values
								target_s = sub_pd[target].values

								if remove_missing_values:
									model_timeseries, target_s = timeseries_rmv_values(model_timeseries, target_s, remove_missing_values)

								#print(preds_pd, model_timeseries, target_s)
								#time.sleep(100)
								val = metric_handler[metric](model_timeseries, target_s)
								metric_dict[id_interval_labels[i]].append(val)
								if i == 0:
									labels[0].append(j)
									labels[1].append(k)

					ind = pd.MultiIndex(levels=levels, labels=labels, names=names)
					metric_pd = pd.DataFrame(metric_dict, index = ind)
					#print(metric_pd)
					metric_pd.to_csv(self.folder_main + '/'  + id_folder  + '/' + output_file_name )

					metric_pd['ID'] = np.array([id_folder]*len(labels[0]))
					if write_to_overview : id_dfs.append( metric_pd.set_index('ID', append=True, inplace=False))



				else:
					print('Not able to find results file for {0}. Please check your folder'.format(id_folder))

		print('Finished iterating over all ids. Writing out condensed file in {0} folder'.format(OVERVIEW_FOLDER))
		if write_to_overview:
			id_dfs = pd.concat(id_dfs)
			id_dfs.to_csv(self.folder_main+ '/' + OVERVIEW_FOLDER + '/' + CSV_METRICS_CONDENSED)


	def efficiency_test(self, id_folder, periods, period_labels,\
	model_to_compare, ignore_models=['GFT'],\
	 confidence_interval=.90, samples = 10000, p=1./52, filename = None,
	 output_file_name=None, remove_values=None, write=True,  op='MSE'):

		'''
			Performs efficiency test based on politis and romano 1994
		https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870
		'''
		# create path to csv file
		if filename is None:
			file_path = id_folder +'/'+ID_PREDS
		else:
			file_path = id_folder +'/'+ filename
		if output_file_name is None:
			output_file_name = ID_EFFTEST

		# load columns
		preds_pd = pd.read_csv(file_path, index_col=0)

		if remove_values is not None:
			rmv_values(preds_pd, ignore = remove_values, verbose = True)
		bbts = {}

		model_names = list(preds_pd)
		model_names.remove(TARGET)


		# Removing movels in ignore_models

		for i, model in enumerate(ignore_models):
			if model in model_names:
				del preds_pd[model]
				model_names.remove(model)


		n_models = len(model_names)

		#multiindexing
		levels = [['BBT', 'lower_bound', 'upper_bound'], model_names]
		labels = [ [0]*n_models + [1]*n_models + [2]*n_models, list(range(n_models))*3 ]
		names =['Efficiency_test', 'Model']
		ind = pd.MultiIndex(levels = levels, labels = labels, names = names)

		print('Computing the efficiency test on {0}'.format(id_folder))
		#Main computation loop
		for  i, period in enumerate(periods):
			print('Computing the efficiency test on {0}'.format(period))
			bbts[period_labels[i]] = self.bbt(preds_pd,\
			 model_to_compare=model_to_compare,period=period, \
			 confidence_interval=confidence_interval, \
			 samples = samples, p = p, op=op)

		eff_pd = pd.DataFrame(bbts, index =ind)

		if write:
			eff_pd.to_csv(id_folder + '/' + output_file_name )

		return eff_pd






	def bbt(self, preds_pd, model_to_compare='AR12',\
	 seed='random', period=('2012-01-01', '2016-12-25'), verbose = True,\
	 samples=10000, p=1./52, confidence_interval=.90, op='MSE'):

		'''
			Performs timeseries bootstrap to calculate the ratio of MSEs between model_to_compare and all other available models
			Inputs:
				preds_pd = pandas dataframe containing the data to analyze
				model_to_compare = Str naming the model (must be inside the csv file)
				seed = specify a numpy and random seed, otherwise use 'random' to not use any
				period = a 2 tuple containing the first index and last index comprising the period to analyze

 		'''
		if isinstance(seed,int):
			random.seed(seed)
			np.random.seed(seed)

		model_names = list(preds_pd)

		if verbose == True:
			print('Successfully loaded dataframe. \n \n Target name in Config : {0}  \n \n  model_names found: {1}'.format(TARGET, model_names))

		model_preds = []
		model_names.remove(TARGET)

		# Always place target preds at start
		for i, model in enumerate(model_names):
				if i == 0:
					model_preds.append(preds_pd[TARGET][period[0]:period[1]])
				model_preds.append(preds_pd[model][period[0]:period[1]])


		methods = np.column_stack(model_preds)
		n_models = len(model_names)
		eff_obs = np.zeros(n_models)

		# calculate observed relative efficiency
		for i in range(n_models):
			eff_obs[i] = metric_handler[op](methods[:, 0], methods[:, i + 1])
		eff_obs = eff_obs / eff_obs[model_names.index(model_to_compare)]

		# perform bootstrap
		scores = np.zeros((samples, n_models))

		for iteration in range(samples):
			# construct bootstrap resample
			new, n1, n2 = sbb.resample(methods, p)

			# calculate sample relative efficiencies
			for model in range(n_models):
				scores[iteration, model] = metric_handler[op](new[:, 0], new[:, model + 1])
			scores[iteration] = scores[iteration] / scores[iteration,model_names.index(model_to_compare)]


		if op == 'PEARSON':
			eff_obs = 1/eff_obs
			scores = 1/scores

		# define the variable containing the deviations from the observed rel eff
		scores_residual = scores -  eff_obs

		# construct output array
		report_array = np.zeros(3*n_models)
		for comp in range(n_models):
			tmp = scores_residual[:, comp]

			# 95% confidence interval by sorting the deviations and choosing the endpoints of the 95% region
			tmp = np.sort(tmp)

			ignored_tail_size = (1-confidence_interval)/2
			report_array[comp] = eff_obs[comp]
			report_array[n_models*1+comp] = eff_obs[comp] + tmp[int(round(samples * (0.0+ignored_tail_size)))]
			report_array[n_models*2+comp] = eff_obs[comp] + tmp[int(round(samples * (1.0-ignored_tail_size)))]

		return report_array










	def mat_load(self, state_dir = None, filename = None,  dates_bol = True, verbose = True):

		if state_dir is not None:
			self.state_dir = state_dir
		if filename is not None:
			self.filename = filename

		self.preds_pd =  pd.read_csv(self.state_dir + self.filename + '_preds.csv')
		self.metrics_pd =  pd.read_csv(self.state_dir + self.filename + '_table.csv')

		if dates_bol == True:
			self.dates = self.preds_pd['Week'].values
			del self.preds_pd['Week']

		self.model_names = list(self.preds_pd)
		self.season_names = list(self.metrics_pd)

		# Removing Gold standard from model list and 'metrics' from metrics list
		self.target_name = self.model_names[0]
		self.model_names.remove(self.target_name)
		del self.season_names[0]
		self.ranking = self.model_names
		self.target_pred = self.preds_pd[self.target_name].values

		print('Loaded data for : {0} \n Models found : {1} \n Seasons found : {2} \n \n '.format(self.state_name, self.model_names, self.season_names))

	def mat_metricRank(self, metric = None, season = None, verbose = False):
		if metric is not None:
			self.ranking_metric = metric
		if season is not None:
			self.ranking_season = season

		if verbose  == True:
			print('Ranking models based on {0}  metric for {1} season. \n \n'.format(self.ranking_metric, self.ranking_season))

		metrics_pd = self.metrics_pd
		model_names = self.model_names
		n_models =  np.size(model_names)

		if verbose == True:
			print('Number of models found = {0}'.format(n_models))

		season_names = self.season_names

		# Check if metric is in
		metrics_and_models = list(metrics_pd['Metric'].values)
		season_vals = metrics_pd[self.ranking_season].values

		if verbose == True:
			print('Table metric and models list :  \n', metrics_and_models)
			print('Season Values : \n ', season_vals)


		if self.ranking_metric in metrics_and_models:

			i = metrics_and_models.index(self.ranking_metric)
			metric_column = season_vals[i+1:i+n_models+1]
			self.ranking_values = metric_column

		#metric_column = mat_metricColumn(metrics_pd, self.ranking_metric, self.ranking_season, n_models, verbose)

			# Sorted  default ordering is minimum to maximum. For correlations we look for highest positive (??).
			if self.ranking_metric == 'PEARSON':
				metric_column *= -1

			# To compare RMSEs values have to be normalized based on gold standard's MS value.
			if self.ranking_metric == 'RMSE':
				metric_column /= np.sqrt(np.mean(np.power(self.target_pred,2)))


			ranking = [model for (metric, model) in sorted(zip(metric_column, model_names), key=lambda pair: pair[0])]

			if verbose == True:

				print('Old Ranking: {0} \n Values for metric: {2}  \n New ranking: {1} \n \n'.format(self.ranking, ranking, self.ranking_values ))

			self.ranking = ranking

		else:
			print('Ranking metric not available. Please use another metric')

	def mat_predsAndErrors(self, which_rank = None, verbose = False, start_season = 3, n_top_models = None ,dates_bol = True, gft_bol = True):

            '''
             Makes time series, error, and % error plot for the specified number of models. Models are chosen based on the ranking.
             -If there is no ranking, it plots the models in the order they were written on the prediction csv.
            -If the specified number is bigger than the number of models available, it plots all. '''
            # Get predictions for the top n
            if n_top_models is None:
            	n_top_models = self.n_top_models
            ranking = self.ranking
            model_names = self.model_names
            preds_pd =  self.preds_pd
            gold_std = self.target_pred
            n_lags = self.n_lags
            gstd_rmse = np.sqrt(np.mean(np.power(gold_std,2)))
            season_indices = self.season_indices - n_lags
            season_indices[season_indices<0] = 0
            season_names = self.season_names
            if season_indices[1] < 1:
                print('Warning! Season indices values may conflict with code', season_indices)
            ax2_limit  = 2


            season_names.remove('ALL_PERIOD')
            if gft_bol == True:
                season_names.remove('GFT_PERIOD')
            if which_rank  is None:
            	which_rank = range(0, n_top_models)

            if np.amax(which_rank) > len(ranking)-1:
            	print('Not accessible rank detected {0}. Please modify value'.format(np.amax(which_rank)))
            	time.sleep(2)
            	return

            if len(which_rank) > n_top_models:
            	n_top_models = len(which_rank)

            if verbose == True:
            	print('Initializing predsAndErrors function with following values: \n Ranking = {0}, \n Season names = {1} \n Gold standard MS = {2} \n Start season ={3} \n which_rank  = {4}'.format(ranking, season_names, gstd_rmse, start_season, which_rank))
            	print ('Gold standard shape {0} \n, season indices'.format(np.shape(gold_std), season_indices))
            	time.sleep(2)

            # Adjusting plot ticks and plot length
            stind = self.season_indices[(start_season-1)*2]-1
            season_indices -= stind
            plot_names = [ranking[i] for i in which_rank]

            if verbose == True:
            	print('start index {0} \n season_indices {1} \n plot_names ={2} '.format(stind, season_indices, plot_names))
            	time.sleep(2)

            # Create figure and axes
            fig = plt.figure()
            ax = plt.subplot(4, 1, (1, 2))
            ax1 = plt.subplot(4, 1, 3)
            ax2 = plt.subplot(4, 1, 4)

            # Plot gold standard
            ax.plot(gold_std[stind:],color='.20', linewidth=6.0, label=self.target_name, alpha = 1)

            # Compute error and percent error then plot model % CHANGe
            for i in range(0, n_top_models):

            	series = preds_pd[plot_names[i]].values
            	series_err = series - gold_std
            	series_errp = np.divide(series_err,gold_std)

            	series_rmse = np.sqrt(np.mean(np.power(series_err,2)))
            	norm_rmse = (series_rmse/gstd_rmse)

            	series_errp[np.isinf(series_errp) ] = float('NaN')

            	if ax2_limit < norm_rmse:
            		ax2_limit = norm_rmse

            	'''
            	ax.plot(series[stind:], linewidth=2, label=plot_names[i] , alpha = .6)
            	ax1.plot(series_err[stind:], linewidth=3, label=plot_names[i] , alpha = .6)
            	ax2.plot(series_errp[stind:], linewidth=2, label=plot_names[i] , alpha = .6)
            	'''

            	# TEMP FOR TOP 3
            	if i == 0:
            		ax.plot(series[stind:],  color='b',label=plot_names[i] , alpha = .8,linewidth=2.3)
            		ax1.plot(series_err[stind:], color='b', label=plot_names[i] , alpha = .8,linewidth=2.3)
            		ax2.plot(series_errp[stind:], color='b', label=plot_names[i] , alpha = .8,linewidth=2.3)
            	if i == 1:
            		ax.plot(series[stind:],  color = 'r',label=plot_names[i] , alpha = .6,linewidth=2.5)
            		ax1.plot(series_err[stind:], color = 'r', alpha = .6,linewidth=2.5)
            		ax2.plot(series_errp[stind:], color = 'r', alpha = .6,linewidth=2.5)
            	if i == 2:
            		ax.plot(series[stind:],  color ='.75', label=plot_names[i] , alpha = .9,linewidth=1.5, linestyle = 'dashed')
            		ax1.plot(series_err[stind:],  color ='.75', alpha = .9,linewidth=1.5, linestyle = 'dashed')
            		ax2.plot(series_errp[stind:],  color ='.75', alpha = .9,linewidth=1.5, linestyle = 'dashed')


            if gft_bol == True:
            	GFTSERIES = preds_pd['GFT'].values
            	GFTSERIES = GFTSERIES[:,np.newaxis]

            	ax.plot(GFTSERIES[stind:], color='g', label = 'GFT', linewidth=2.5)

            # Add format to plots
            ax.tick_params(labelsize=14)
            ax1.tick_params(labelsize=14)
            ax2.tick_params(labelsize=14)

            ax.grid(linestyle = 'dotted', linewidth = .8)
            ax1.grid(linestyle = 'dotted', linewidth = .8)
            ax2.grid(linestyle = 'dotted', linewidth = .8)

            ax.set_ylabel('ILI activity', fontsize = 16)
            ax1.set_ylabel('Error', fontsize = 16)
            ax2.set_ylabel(' Error (Normalized)', fontsize = 14)

            x= gold_std
            x[np.isnan(x)]=0

            ax.set_ylim([-np.max(x)*.05,np.max(x)*1.05])
            ax2.set_ylim([-ax2_limit,ax2_limit]) # Change limits based on mean of error mean above and below zero.

            ax1.axhline(0, 0, 1, color='gray')
            ax2.axhline(0, 0, 1, color='gray')

            ax.set_title('{0}'.format(self.state_name),  fontsize = 16,  weight='bold')
            ax.legend( fontsize = 14)

            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax1.get_xticklabels(), visible=False)

            # Not good, change

            # Yearly
            n_seasons = len(season_indices)/2
            tick_indices = [season_indices[i*2][0] for i in range(start_season-1, n_seasons)]

            # Seasonal
            # tick_indices = [season_indices[i][0] for i in range((start_season-1)*2, len(tick_indices))]

            if dates_bol == True:
                tick_labels = self.dates[stind+tick_indices]
            else:
                tick_labels = sorted(season_names[start_season-1:]*2)


            ax.set_xticks(tick_indices, minor=False)
            ax1.set_xticks(tick_indices, minor=False)
            ax2.set_xticks(tick_indices, minor=False)
            ax2.set_xticklabels(tick_labels, minor=False, rotation=75)


            # save figure
            fig = plt.gcf()
            fig.set_size_inches(12, 9)
            fig.savefig(self.state_dir+'/pyplots/{0}/series_errs.png'.format(self.filename) , format='png', dpi=300)
            # fig.savefig(self.state_folder+'/_pyplots/_series_errs_'+ self.state_name+'_'+self.filename+ '.png', format='png', dpi=300)

	def  mat_metric_bargraph(self, verbose = False, which_metric=['RMSE', 'PEARSON']):
		# Get table dataframe
			rfile = self.svars.mat_ST_RANKS
			vfile = self.svars.mat_ST_RANKS_V
			color_list = self.svars.colors

			ranking_pd = pd.read_csv(self.state_dir + rfile, index_col =0)
			values_pd = pd.read_csv(self.state_dir + vfile,  index_col =0)


			for i, metric in enumerate(which_metric):
				r = ranking_pd[metric].values
				v = values_pd[metric].values

				if i == 0:
					#color_dict = color_assign(r,color_list)
					color_dict = self.svars.STATE_COLOR_DICT

				orderedColors = []
				for k, model in enumerate(r):
					orderedColors.append(color_dict[model])


				fig = plt.figure() # CHANGE SO IT PLOTS EVERYTHING IN ONE GRAPH, GIVE FORMAT
				objects = r
				y_pos = np.arange(len(objects))
				performance = v

				plt.bar(y_pos, performance, align='center', alpha=0.8, color= orderedColors)

				plt.xticks(y_pos, objects, rotation=90)

				if  metric == 'RMSE':
					ylab = 'RMSE / gold standard MS.'
				elif metric == 'PEARSON':
					ylab = 'Coefficient value.'
					plt.ylim([np.min(v)*.7,1])

				plt.ylabel(ylab)
				plt.title('{0} for {1}'.format(metric, self.state_name), fontsize = 16)
				plt.grid(linestyle = 'dotted', linewidth = .8)
				fig.set_size_inches(7 , 9)
				fig.savefig(self.state_dir+'/pyplots/{0}/bar_'.format(self.filename)+ metric+ '.png', format='png', dpi=300)
				#fig.savefig(self.state_folder+'/_pyplots/_bars_'+ self.state_name+'_'+metric+'_'+self.filename+ '.png', format='png', dpi=300)
				plt.close()


	def  mat_all_ranks2csv(self, verbose = False, which_metric = ['PEARSON', 'RMSE']):
		# Get table dataframe
		metrics_pd = self.metrics_pd
		model_names = self.model_names
		season = self.ranking_season
		n_models = len(model_names)
		gold_std = self.target_pred

		ranking_dict = {}
		values_dict = {}
		if verbose == True:
			print('Initializing function ranks2csv with following parameters: \n metrics_pd: {0} \n model_names: {1} \n season: {2} \n n_models: {3}'.format(metrics_pd, model_names, season, n_models))

		for i,metric in enumerate(which_metric):
			print('Looking for {0} values'.format(metric))
			metric_column = mat_metricColumn(metrics_pd, metric, season, n_models, verbose)
			ranking, values =mat_getRank(model_names, metric_column, metric, verbose = False, gold_std = gold_std)

			if verbose == True:
				print('Metric = {0} \n metric_column = {1} \n model_names {2}  \n ranking: {3} \n \n \n'.format(metric, metric_column, model_names, ranking))
				time.sleep(2)

			ranking_dict[metric] = ranking
			values_dict[metric] = values

		ranking_pd = pd.DataFrame(ranking_dict)
		values_pd = pd.DataFrame(values_dict)


		if verbose == True:
			print('Ranking table looks like this : {0}. \n'.format(ranking_pd))
			print('Writing down data')

		ranking_pd.to_csv(self.state_dir + '/mat_metric_rankings.csv')
		values_pd.to_csv(self.state_dir + '/mat_metric_values.csv')

	def mat_eff_matrix_plot(self, bbt_file_name ='BBT_mse_matrix.csv', verbose = False, save_name = None ):
		# Plots values for efficiency matrix models specified in the horizontal direction of matrix
		eff_matrix_pd = pd.read_csv(self.state_dir + bbt_file_name, index_col =0)
		model_names_columns = list(eff_matrix_pd)
		model_names_rows = list(eff_matrix_pd.transpose())
		if verbose == True:
			print(model_names_columns, model_names_rows)
			time.sleep(2)

		n_model_names_columns = len(model_names_columns)
		n_model_names_rows = len(model_names_rows)
		n_div = n_model_names_rows -1
		average_eff = []
		# Entering summing loop
		for c, model_c in enumerate(model_names_columns):
			sum_eff = 0
			if verbose == True:
				print(c, model_c)
				time.sleep(1)
			for r, model_r in enumerate(model_names_rows):
				if verbose == True:
					print(r,model_r)
					time.sleep(1)
				if model_r not in model_c:
					sum_eff += eff_matrix_pd.get_value(model_r, model_c, takeable=False)
			average_eff.append(sum_eff/n_div)


		saving_dir = self.state_dir+'/pyplots/bbt_'
		if save_name is not None:
			saving_dir += save_name
		saving_dir += self.filename

		average_eff = np.vstack(average_eff)
		average_eff[average_eff > 2] == 2
		bar_graph(model_names_columns , average_eff, '(No units)', 'Average efficiency', saving_dir )

	def mat_eff_matrix(self, filename='', period='all'):

		random.seed(2)
		np.random.seed(2)

		state_name = self.state_name
		RESULTS_DIR = self.state_folder
		gold_std_name = self.target_name
		verbose = True

		# load columns
		preds_pd = self.preds_pd

		model_names = list(preds_pd)
		if verbose == True:
			print('Successfully loaded dataframe. \n \n gold_standard_name {0}  \n \n  model_names {1}'.format(gold_std_name, model_names))

		model_preds = []

		for i, model in enumerate(model_names):
			if period == 'gft':
				model_preds.append(preds_pd[model].values[143:241])
			elif period == 'all':
				# check target for zeros
				if i == 0:
					mask = (preds_pd[model].values == 0)
				model_preds.append(preds_pd[model].values[~mask])

		model_names.remove(gold_std_name)

		# comment or uncomment the following blocks for either WEEK 1 or WEEK 2 horizon.
		horizon = 0

		#methods = np.column_stack((target_1, ar_1, argo_1))
		methods = np.column_stack(model_preds)

		# implementation
		p = 1./52

		samples = 10000
		n_models = methods.shape[1] - 1

		efficiency_matrix = np.zeros([n_models, n_models])

		# calculate observed relative efficiency
		for ii in range(n_models):
			eff_obs = np.zeros(n_models)
			for i in range(n_models):
				eff_obs[i] = mse(methods[:, 0], methods[:, i + 1])
			eff_obs = eff_obs / eff_obs[ii]

			# perform bootstrap
			scores = np.zeros((samples, n_models))
			for iteration in range(samples):
				# construct bootstrap resample
				new, n1, n2 = sbb.resample(methods, p)

				# calculate sample relative efficiencies
				for model in range(n_models):
					scores[iteration, model] = mse(new[:, 0], new[:, model + 1])
				scores[iteration] = scores[iteration] / scores[iteration, -1]

			# define the variable containing the deviations from the observed rel eff
			scores_residual = scores - eff_obs

			# construct output array
			report_array = np.zeros((3,n_models))
			for comp in range(n_models):
				tmp = scores_residual[:, comp]

				# 95% confidence interval by sorting the deviations and choosing the endpoints of the 95% region
				tmp = np.sort(tmp)
				report_array[0, comp] = eff_obs[comp]
				report_array[1, comp] = eff_obs[comp] + tmp[int(round(samples * 0.050))]
				report_array[2, comp] = eff_obs[comp] + tmp[int(round(samples * 0.950))]

			efficiency_matrix[:,ii] = report_array[0,:].reshape(n_models)

		print(efficiency_matrix)

		efficiency_matrix_pd = pd.DataFrame(data = efficiency_matrix, index=model_names, columns = model_names)
		efficiency_matrix_pd.to_csv(RESULTS_DIR + state_name + '/'  + 'BBT_mse_matrix.csv')


	def insert_gft(self):
		preds_pd =  pd.read_csv(self.state_dir + self.filename + '_preds.csv')
		gft_pd = pd.read_csv(self.state_dir+'GFT_scaled.csv')
		preds_pd['GFT'] = gft_pd['GFT']
		print(preds_pd)
		preds_pd.to_csv(self.state_dir + self.filename + '_preds.csv')

	def term_scatters(self, n_terms =  8, start_date = '2010-01-03', end_date = '2016-12-25',start_window = 26,\
	plot_date = ['2012-01-01', '2016-12-25'], window = 'extending',  verbose = False):
		# Makes a scatter and correlation plot for the top n_terms based on average correlation during whole period specified by
		# start_date and end_date. Correlation starts a number of values specified by start_window
		# 1.- Conseguir documento

		window_size =104

		terms_pd = pd.read_csv(self.state_dir + self.state_name + self.svars.TERM_FILE)
		del terms_pd['GFT']

		''' Remove terms
		terms_pd.fillna(0, inplace=True)
		print len(term_names)
		for i,name in enumerate(term_names):

			series = terms_pd[name].values
			nzeros = np.size(series[series == 0])
			zvector = np.size(np.zeros_like(series))

			print i
			if nzeros > zvector*.5:
				terms_pd =terms_pd.drop(name, axis =1)

		term_names = list(terms_pd)
		'''

		# Adding auto regression terms
		gstd = terms_pd['ILI'].values

		for i in range(0,12):
			ar_term = np.hstack([gstd[1+i:len(gstd)], np.zeros(1+i)])

			terms_pd['AR{0}'.format(1+i)] =  pd.Series(ar_term, index=terms_pd.index)

		term_names = []
		term_names = list(terms_pd)
		n_terms = len(term_names)
		date_list = list(terms_pd['Week'].values)
		start_index = date_list.index(start_date)
		finish_index = date_list.index(end_date)

		plot_indices = []
		plot_indices.append(date_list.index(plot_date[0]))
		plot_indices.append(date_list.index(plot_date[1]))
		if verbose == True:
			print(plot_indices)
			print(date_list[plot_indices[0]], date_list[plot_indices[1]])

		terms_pd['Week'] = pd.to_datetime(terms_pd['Week'])
		corr_matrix = np.ones([n_terms, len( date_list[start_index-1+start_window:finish_index+1] ) ] )
		corr_matrix *= .1

		# Getting data for both extending dataset and two year dataset

		for i, date in enumerate(date_list[start_index+start_window-1:finish_index+1]):

			# Extracting dates depending on the type of window.

			if window == 'extending' or ( window == 'two_year' and start_window+i  < window_size ):
				if verbose == True:
					print('Extending')
				mask = (terms_pd['Week'] > start_date) & (terms_pd['Week'] <= date) # Mask to extract a sub-dataframe based on the dates
			elif window == 'two_year' and start_window+i > window_size:
				date_index=date_list.index(date)

				if verbose == True:
					print('Two_year')
					print(i,(i-window_size))
					print(date_list[date_index-window_size], date)
					time.sleep(1)

				mask = (terms_pd['Week'] > date_list[date_index-window_size]) & (terms_pd['Week'] <= date) # Mask to extract a sub-dataframe based on the dates


			temp_pd = terms_pd.loc[mask]
			gstd = temp_pd['ILI'].values.astype(np.float)
			for j in range(2,n_terms):
				if verbose == True:
					print('Term name')
					print(term_names[j])
				term = temp_pd[term_names[j]].values


				#term[np.isnan(term)] = 0
				if verbose == True:
					print('Debugging p_corr')
					print(i, date)
					print('Printing term')
					print(np.shape(term))
					print(term.dtype)
					print(term)
					#time.sleep(5)
				p_corr = scipy.stats.pearsonr(gstd, term)
				if p_corr[0] > 1:
					print('Correlation > 0')
					print(p_corr[0])
					print(gstd)
					print(term)
					time.sleep(100)
				corr_matrix[j,i] = p_corr[0]


		# Extracting period to plot
		mask = ( terms_pd['Week'] >= date_list[plot_indices[0]]) & (terms_pd['Week'] <= date_list[plot_indices[1]])
		# Getting only values corresponding to the plotting period
		holder = copy.copy(corr_matrix)
		corr_matrix = corr_matrix[:,plot_indices[0]-(start_index+start_window-1):plot_indices[1]+1-(start_index+start_window-1)]
		temp_pd = copy.deepcopy(terms_pd.loc[mask])
		weeks = temp_pd['Week'].values
		gstd = temp_pd['ILI']

		if verbose == True:
			print('Corr_matrix, shape', np.shape(corr_matrix))
			print('N of columns taken by the index difference', plot_indices[1]-plot_indices[0])
			print('Weeks', weeks)
			print(np.shape(corr_matrix))

		# Find top 10 terms based on average correlation
		corrSum = np.sum(corr_matrix, axis = 1)
		n_iters = np.size(corr_matrix, axis = 1)
		average_corrs = corrSum/n_iters
		average_corrs[np.isnan(average_corrs)] = 0

		if verbose == True:
			print('Correlations')
			print (np.shape(corr_matrix), np.shape(average_corrs))
			print('N of weeks left',np.shape(holder[:,plot_indices[1]+1-(start_index+start_window-1):340]))
			print('plot_indices', plot_indices[1]+1, 'sindex',start_index+start_window-1)
			for i in range(n_terms):
				print(average_corrs[i], term_names[i], corrSum[i])
			print(n_iters)
			time.sleep(10)
		ord_corrs, ord_names = order_tuple(average_corrs, term_names)


		# plot terms

		'''
		f = plt.figure()
		f.set_size_inches(20  , 9)
		gs = gridspec.GridSpec(10, 2,
                       width_ratios=[1, 2])
		#f, axarr = plt.subplots(10,3, )
		gstd = terms_pd['ILI'].values
		for i in range(0,10):
			term_name = ord_names[(-1)-i]
			term_index = term_names.index(term_name)
			corr_series = corr_matrix[term_index, :]
			term_series = terms_pd[term_name].values

			# Plot scatter plot
			plt.subplot(gs[i*2]).scatter(term_series, gstd)
			plt.subplot(gs[i*2+1]).plot_date( x=weeks, y=corr_series, color = 'red', linewidth=1.5, label=term_name, alpha = .8 )
			plt.legend()
			plt.grid(linestyle = 'dotted', linewidth = .8)
		'''
		'''
		#plot top ten terms
		plt.figure(1, figsize=[18,18], dpi=300)
		f, axarr = plt.subplots(9,1, figsize=[18,18])

		for i in range(0,9):
			term_name = ord_names[(-1)-i]
			term_index = term_names.index(term_name)
			corr_series = corr_matrix[term_index, :]
			term_series = terms_pd[term_name].values

			# Plot scatter plot
			axarr[i].plot_date( x=weeks, y=corr_series, color = 'mediumslateblue', linewidth=1, label='"{0}"'.format(term_name), alpha = .8 )
			axarr[i].legend(fontsize=15)
			if i < 8:
				axarr[i].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
			axarr[i].grid(linestyle = 'dotted', linewidth = .8)

		plt.suptitle('Correlation History for {0}.'.format(self.state_name), fontsize = 16)
		plt.savefig(self.folder_main + self.svars.OVERVIEW_FOLDER + '/scatter_history_{0}.png'.format(self.state_name), format='png')
		plt.close()
		'''

		f =	plt.figure(1, figsize=[18,18], dpi=300)

		for i in range(0,9):
			term_name = ord_names[(-1)-i]
			term_index = term_names.index(term_name)
			corr_series = corr_matrix[term_index, :]

			# Plot scatter plot
			plt.plot_date( x=weeks, y=corr_series,fmt='-', color = self.svars.colors[i], linewidth=1, label='"{0}"'.format(term_name), alpha = .8 )
			plt.plot_date( x=weeks, y=corr_series, color = self.svars.colors[i],markersize=2, linewidth=.4, alpha = .8 )
		plt.legend(fontsize=15)
		plt.grid(linestyle = 'dotted', linewidth = .8)
		plt.title('Correlation History for {0} top 9 terms.'.format(self.state_name), fontsize = 18)
		plt.ylabel('Pearson Correlation coefficient.', fontsize=18)
		plt.xlabel('Date.', fontsize=18)
		plt.tick_params(axis='both', labelsize = 16)
		plt.savefig(self.folder_main + self.svars.OVERVIEW_FOLDER + '/scatter_history2_{0}_{1}.png'.format(self.state_name, window), format='png')
		plt.close()

		# Plotting term scatters
		plt.figure(1, figsize=[18,18], dpi=300)
		f, axarr = plt.subplots(3,3, figsize=[18,18])
		cs = [[self.svars.colors[0]], [self.svars.colors[1]], [self.svars.colors[2]], \
		  [self.svars.colors[4]], [self.svars.colors[5]]]
		year_indices = [0, 52, 53, 104, 105, 156, 157, 208, 209, 260]
		l1 = ['2012', '2013', '2014', '2015', '2016']
		termsPerRow = 3

		for c in range(0,10):
			term_name = ord_names[(-1)-c]
			term_index = term_names.index(term_name)
			corr_series = corr_matrix[term_index, :]
			term_series = terms_pd[term_name].values
			average_corr = ord_corrs[(-1)-c]

			if verbose == True:
				print('debugging')
				print(term_name, term_index)
				print('series ave max and min',np.mean(term_series), np.max(term_series), np.min(term_series))
				print('Corr', average_corr)
				print('GSTD', np.mean(gstd), np.max(gstd))
				for i in range(len(year_indices)/2):
					print('lenghts',len(term_series[year_indices[i*2]:year_indices[i*2+1]]), len(gstd[year_indices[i*2]:year_indices[i*2+1]]))


			if c < 3:
			  for i in range(len(year_indices)/2):
			  	axarr[0,c].scatter(term_series[year_indices[i*2]:year_indices[i*2+1]], gstd[year_indices[i*2]:year_indices[i*2+1]], color = cs[i], label = l1[i])

			  axarr[0,c].set_title( '"{0}", Corr = {1:.3f}'.format(term_name, average_corr),fontsize = '16')
			  axarr[0,c].tick_params(labelsize=14)
			  if c == 0:
			    axarr[0,c].legend(prop={'size':15})
			if c < 6 and c > 2:
			  for i in range(len(year_indices)/2):
			    axarr[1,c - termsPerRow].scatter(term_series[year_indices[i*2]:year_indices[i*2+1]], gstd[year_indices[i*2]:year_indices[i*2+1]], color = cs[i], label = l1[i])
			  axarr[1,c - termsPerRow].set_title('"{0}", Corr = {1:.3f}'.format(term_name, average_corr),fontsize = '16')
			  axarr[1,c - termsPerRow].tick_params(labelsize=14)
			  if c - termsPerRow==0:
			    axarr[1,c- termsPerRow].set_ylabel('Suspected number of cases', fontsize = 20)
			if c < 9 and c > 5:
			  for i in range(len(year_indices)/2):
			    axarr[2,c - termsPerRow*2].scatter(term_series[year_indices[i*2]:year_indices[i*2+1]], gstd[year_indices[i*2]:year_indices[i*2+1]], color = cs[i], label = l1[i])
			  axarr[2,c - termsPerRow*2].set_title( '"{0}", Corr = {1:.3f}'.format(term_name, average_corr),fontsize = '16')
			  axarr[2,c - termsPerRow*2].legend()
			  axarr[2,c - termsPerRow*2].tick_params(labelsize=14)
			  if c == 7:
			    axarr[2,c - termsPerRow*2].set_xlabel('Predictor',fontsize = 20)
		plt.suptitle('Top correlated terms for {0}'.format(self.state_name))
		plt.savefig(self.folder_main + self.svars.OVERVIEW_FOLDER + '/scatter_top9_{0}_{1}.png'.format(self.state_name, window), format='png')
		plt.close()

		#plt.show()

		# 2.- Cargar numero de terminos
		# Generar matriz de heatmap
		# 3.- entrar a un ciclo que correr desde el indice de inicio al indice final.
		# por cada iteracion se calcula la correlacion de cada termino y se agrega a a la matriz
		# Se hace un heatmap

	def pyplot_subdir(self):
		di = self.state_dir  + '/pyplots/' +self.filename
		if not os.path.exists(di):
		    os.makedirs(di)

class InputVis:
	""" Visualizer class that aids in the analysis of timeseries data for statistical modelling

		Parameters
		__________

		dataObject : object
			An object from the data classes (Load or LoadFromTemplate classes). Contains all information regarding
			the different cases to analyze.
		folder_name :  str, optional (default='InputVisualizer')
			A string specifying the name of the folder generated by InputVisualizer class.
			All results and images regarding visualizer class will be saved on this folder
		output_dir : str, optional (default=None)
			A string containing the absolute path to the working directory to output results.
			If no directory is specified, then InputVisualizer grabs the current working directory as the output.

		Attributes
		__________

			data : object
				LoadFromTemplate or Load object containing all data to analyze.
				See LoadFromTemplate  and Load documentation to see all attributes within these classes/
			folder_main : str
				String containing the path to the folder where all results are written to.
			overview_folder : str,
				String containing the path to the folder where condensed analysis of all areas is written to.

	"""
	def __init__(self, dataObject=None, folder_name='InputVisualizer', output_dir=None, verbose = True):


		# Checking for data
		if dataObject is None:
			print('InputVisualizer initialized without Data Object detected. Please add a data Object \
			to proceed with data analysis')
		else:
			if isinstance(dataObject, Load) or isinstance(dataObject, LoadFromTemplate):
				self.data = dataObject
			else:
				print('DataObject is not an instance of data class. Please verify your input.')
				return

		#checking for output
		if output_dir:
			self.folder_main = output_dir+'/'+folder_name
		else:
			print('Unable to find an specified output directory. Using current working directory ({0}) as output.'.format(os.getcwd()))
			self.folder_main = os.getcwd()+'/'+folder_name

		try:
			output_name = gen_folder(self.folder_main)
			if output_name != self.folder_main:
				print('Warning! A folder called {0} already exists.'.format(self.folder_main))
				self.folder_main = output_name
			else:
				print('Successfully generated class folder. All results will be written to: {0}'.format(self.folder_main))
		except Exception as t:
			print('Unable to create class folder')
			print(t)
			return

		self.overview_folder = self.folder_main + '/' + OVERVIEW_FOLDER
		gen_folder(self.overview_folder)



	def scatter_features(self, id_=None, ar_model=None, n_features=9, feature_names=None, metric=metric_pearson, mode='show',\
	start_period=None, end_period=None, color_intervals=None, output_filename='scatterplot_features', output_path=None, ext='png', font_path=None):
		"""
		Generates scatter plots for a number of features of the model in a array-like fashion

		color_intervals : dict (Optional, default None)
			Dictionary which contains as key a colo and as value the intervals to plot from the dataframe
		font_path : str (Optional, default None)
			Specify a font to use within the plot (some cases like chinese characters can't be plotted with default font
			). If set to None, uses matplotlib's default.
		"""

		if font_path:
			from matplotlib import font_manager
			prop = font_manager.FontProperties(fname=font_path)

		target = copy.copy(self.data.target[id_])
		features = copy.copy(self.data.features[id_])

		if ar_model:
			ar_lags = self.data.generate_ar_lags(id_, ar_model, verbose = False, store=False)
			features = pd.concat([ar_lags, features], axis=1)
		if start_period is None:
			start_period = features.index[0]
		if end_period is None:
			end_period = features.index[-1]

		target = target[start_period:end_period]
		features = features[start_period:end_period]

		if feature_names:
			ordered_names = feature_names
			ordered_metric = ['N/A']*len(ordered_names)
		elif metric:
			metric_vals = []
			for name in list(features):

				metric_vals.append(metric(target.values.ravel(), features[name].values.ravel()))
			ordered_metric = []
			ordered_names = []
			for v, n in sorted(zip(metric_vals, list(features)), key = lambda x: x[0], reverse=True):
				ordered_metric.append(v)
				ordered_names.append(n)
		else:
			ordered_names = sorted(list(features))
			ordered_metric = ['N/A']*len(ordered_names)


		if n_features is None:
			n_features = len(ordered_names)


		n_rows = int(np.ceil(n_features/3))
		f, axarr = plt.subplots(n_rows,3)
		axes = axarr.ravel()


		for i in range(n_features):

			if color_intervals:
				for color, interval in color_intervals.items():
					feat_series = features[ordered_names[i]][interval[0]:interval[1]].values.ravel()
					axes[i].scatter(feat_series, target[interval[0]:interval[1]].values.ravel(), s=2, label=interval[0])
			else:
				feat_series = features[ordered_names[i]].values.ravel()
				axes[i].scatter(feat_series, target.values.ravel(), s=1)


			if font_path:
				axes[i].set_title('{0} (metric={1})'.format(ordered_names[i], round(ordered_metric[i],2)), fontproperties=prop)
			else:
				axes[i].set_title('{0} (metric={1})'.format(ordered_names[i], round(ordered_metric[i],2)))
			axes[i].spines['top'].set_visible(False)
			axes[i].spines['right'].set_visible(False)

			#if i < n_rows*3 - 3:
			#	axes[i].set_xticklabels([])

			if i == n_rows*3 - 3:
				axes[i].set_xlabel('Feature Values')
			if i == 0:
				axes[i].set_ylabel('Target')

			if i%3 == 1 or i%3 == 2:
				axes[i].set_yticklabels([])

		plt.gcf().set_size_inches([9,3.4*n_rows])
		if mode == 'show':
			plt.show()
		elif mode == 'save':
			if output_path is None:
				if not os.path.isdir('{0}/{1}'.format(self.folder_main, id_)):
					gen_folder('{0}/{1}'.format(self.folder_main, id_))
				plt.savefig('{0}/{1}/{2}.{3}'.format(self.folder_main, id_, output_filename, ext), format=ext)
			else:
				plt.savefig(output_path + '/' + output_filename, format=ext)
		plt.close()
	def group_corrmap(self, ids, data = None, start_period=None, end_period=None, output_filename='corrmap', \
	ext='png', mode='show', show_values=True, clustering=True):
		"""
		Generates correlation matrix and clustering to show relationships between locations in the study
		Parameters:

		ids: str
			String tag referencing the DataObject's specific data ids.
		data : load Object, optional
			load class object containing the data to showself.
		start_period : str, optional (default is None)
			Pandas dataframe starting index. If set to none then grabs the first
			index.
		end_period : str, optional (default is None)
			Pandas dataframe ending index. If set to None then grabs the last index
			in the dataframe
		mode : str, optional (default = 'save')
			if show, then shows the plot. if save, then saves the plot on the ID
			folder.
		similarity_metric: string, optional (default='pearson')
			Choose between a similarity metric between the target and the feature
		ext : str, optional (default 'png')
			format extension for the plots
		output_filename : str, optional (default 'features')
		 	Name of the plot file if 'save' mode is set.
		show_values : Boolean, optional (defaul is True)
			Option to show the correlation coefficient within the heatmap plot.
		"""
		if ids == 'all':
			ids = self.data.id
		if data is None:
			data = self.data

		targets = {}
		for i, id_ in enumerate(ids):

			# Check periods are valid queries for the data.
			if i == 0:
				if start_period and start_period in self.data.target[id_].index:
					pass
				elif start_period is None:
					start_period = self.data.target[id_].index[0]
				else:
					print('Unable to identify start_period {0} as valid start reference.\
					please review'.format(start_period))
					return

				if end_period and end_period in self.data.target[id_].index:
					pass
				elif end_period is None:
					end_period = self.data.target[id_].index[-1]
				else:
					print('Unable to identify end_period {0} as valid end reference.\
					please review'.format(end_period))
					return

				ind = self.data.target[id_][start_period:end_period].index

			targets[id_] = data.target[id_][start_period:end_period].values.ravel()



		targets = pd.DataFrame(targets, index=ind)
		corrmatrix = targets.corr()

		#plotting heatmap

		sns.heatmap(corrmatrix, xticklabels=corrmatrix.columns, yticklabels=corrmatrix.columns,\
		 annot=show_values, cmap='coolwarm', linewidths=.5)

		#Performing Clustering

		plt.show()
		plt.close()




	def target_heatmap_plot(self, ids='all', to_overview=True, start_period=None,\
	end_period=None, filename='targetHeatmap', ignore_values=[.5], ext='png', alpha=1.5, map_nan_to=-5,\
	reorder=True):

		#First we get the ids we want to compute on
		if ids == 'all':
			ids = self.data.id
		elif isinstance(ids, list) and all(isinstance(c,str) for c in ids):
			ids = [id_ for id_ in ids if id_ in self.data.id]
		elif isinstance(ids, str):
			if ids in self.data.id:
				ids = [ids]

		# Check periods are valid queries for the data.
		if start_period and start_period in self.data.target[ids[0]].index:
			pass
		elif start_period is None:
			start_period = self.data.target[ids[0]].index[0]
		else:
			print('Unable to identify start_period {0} as valid start reference.\
			please review'.format(start_period))
			return
		if end_period and end_period in self.data.target[ids[0]].index:
			pass
		elif end_period is None:
			end_period = self.data.target[ids[0]].index[-1]
		else:
			print('Unable to identify end_period {0} as valid end reference.\
			please review'.format(end_period))
			return


		if to_overview:
			overview_df = None

		for i, id_ in enumerate(ids):
			if not os.path.isdir('{0}/{1}'.format(self.folder_main, id_, filename, ext)):
				gen_folder('{0}/{1}'.format(self.folder_main, id_, filename, ext))
			id_df = copy.copy(self.data.target[id_][start_period:end_period])
			#std = id_df.std()
			for v in ignore_values:
				id_df[id_df[id_].values == v] = float('NaN')
			id_df = id_df/id_df.max()
			id_df=id_df.fillna(value=map_nan_to)
			if i == 0:
				overview_df = copy.copy(id_df)
			else:
				overview_df = pd.concat([overview_df, copy.copy(id_df)], axis=1)



			fig = plt.gcf()
			fig.set_size_inches(12, 5)
			plt.subplots_adjust(left=.08, bottom=.28, right=.99, top=.88, wspace=.20, hspace=.20)
			plt.savefig('{0}/{1}/{2}.{3}'.format(self.folder_main, id_, filename, ext), format=ext)
			plt.close()

		if reorder:
			sums = overview_df.sum()
			names = list(overview_df)
			sorted_names = [st for st,v in sorted( zip(names, np.array(sums)), key = (lambda x: float(x[1])), reverse=True)]
			overview_df=overview_df.reindex_axis(sorted_names, axis=1)

		ax = sns.heatmap(overview_df.transpose(), linewidths=0,yticklabels=True)

		for tick in ax.get_yticklabels():
			tick.set_rotation(0)


		fig = plt.gcf()

		fig.set_size_inches(12 , (4/15)*len(ids))
		plt.subplots_adjust(left=.10, bottom=.28, right=.99, top=.88, wspace=.20, hspace=.20)
		plt.savefig('{0}/{1}.{2}'.format(self.overview_folder, filename, ext), format=ext)
		plt.close()

	def plot_features(self, id_, data=None, feature_names=None, start_period=None,\
		end_period=None, output_filename='features', ext='png', alpha=1, similarity_metric='pearson', mode='show'):
		'''
		Parameters
		__________

		id_: str
			String tag referencing the DataObject's specific data.
		data : load Object, optional
			load class object containing the data to showself.
		feature_names : list of str, optional (default is None)
			List of strings containing the name of the features to plot.
			If set to None, plots all the features
		start_period : str, optional (default is None)
			Pandas dataframe starting index. If set to none then grabs the first
			index.
		end_period : str, optional (default is None)
			Pandas dataframe ending index. If set to None then grabs the last index
			in the dataframe
		mode : str, optional (default = 'save')
			if show, then shows the plot. if save, then saves the plot on the ID
			folder.
		alpha : float, opational (default = 1)
			Parameter that defines the separation scaling between timeseries
		similarity_metric: string, optional (default='pearson')
			Choose between a similarity metric between the target and the feature
		ext : str, optional (default 'png')
			format extension for the plots
		output_filename : str, optional (default 'features')
		 	Name of the plot file if 'save' mode is set.
		'''

		if data is None:
			data = self.data
		elif isinstance(data, (Load, LoadFromTemplate)):
			pass
		else:
			print('Data object not detected. Please verify input.')

		# Checking timeseries boundaries
		if start_period and start_period in data.target[id_].index:
			pass
		elif start_period is None:
			start_period = data.target[id_].index[0]
		else:
			print('Unable to identify start_period {0} as valid start reference.\
			please review'.format(start_period))
			return
		if end_period and end_period in data.target[id_].index:
			pass
		elif end_period is None:
			end_period = self.data.target[id_].index[-1]
		else:
			print('Unable to identify end_period {0} as valid end reference.\
			please review'.format(end_period))
			return

		if not os.path.isdir('{0}/{1}'.format(self.folder_main, id_)):
			gen_folder('{0}/{1}'.format(self.folder_main, id_))
		metric_values = []
		if id_ in data.id:
			target = data.target[id_][start_period:end_period].values
			features = data.features[id_][start_period:end_period]
			indices =  data.features[id_][start_period:end_period].index.values


			if feature_names is None:
				feature_names = list(features)

			metric_values = []
			target  = target/np.max(target)
			index = []
			for i, col in enumerate(feature_names):
				series = copy.copy(features[col].values)

				nans = np.isnan(series)

				if np.sum(nans) > 0:
					print('WARNING! {0} series in id:{1} contains {2} NaNs'.format(col, id_, np.sum(nans)))

				series[nans]=0
				series /= np.max(series)
				mu = np.mean(series)
				series -= mu
				series[nans] = float('NaN')

				x, y = timeseries_rmv_values(series.ravel(), target.ravel(), [NAN_TO_VALUE], verbose=False)
				y, x = timeseries_rmv_values(y.ravel(), x.ravel(), [NAN_TO_VALUE], verbose=False)
				v = scipy.stats.pearsonr(x, y)[0]
				metric_values.append(v)
				plt.plot(series - i*alpha, label=col, alpha=.5)
				index.append(-i*alpha)
				if i%6 == 0:
					plt.plot(target-i*alpha-.5, color='black')


			plt.title('Comparison with target (black), similarity metric = {0}'.format(similarity_metric))
			plt.yticks( index,['{0} ({1})'.format(f,round(v,2)) for f,v in zip(feature_names, metric_values)])
			plt.xticks(range(len(indices)),indices, rotation=45)
			plt.gca().spines['top'].set_visible(False)
			plt.gca().spines['right'].set_visible(False)
			plt.gca().xaxis.set_major_formatter(IndexFormatter(indices))
			plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(6))
			plt.gcf().set_size_inches([11,10])
			plt.subplots_adjust(left=0.25, bottom=0.08, right=0.90, top=0.97, wspace=.20, hspace=.20)
			if mode == 'show':
				plt.show()
			if mode == 'save':
				plt.savefig('{0}/{1}/{2}.{3}'.format(self.folder_main, id_, output_filename,ext),fmt=ext)
			plt.close()

		else:
			print('ID not detected within dataObject')

	def similarity_barplot(self, id_, data=None, feature_names=None, start_period=None,\
		end_period=None, output_filename='features_barplot', ext='png', alpha=1, similarity_metric='pearson',\
		 mode='show', ordering='decreasing'):
		'''
		Parameters
		__________

		id_: str
			String tag referencing the DataObject's specific data.
		data : load Object, optional
			load class object containing the data to showself.
		feature_names : list of str, optional (default is None)
			List of strings containing the name of the features to plot.
			If set to None, plots all the features
		start_period : str, optional (default is None)
			Pandas dataframe starting index. If set to none then grabs the first
			index.
		end_period : str, optional (default is None)
			Pandas dataframe ending index. If set to None then grabs the last index
			in the dataframe
		mode : str, optional (default = 'save')
			if show, then shows the plot. if save, then saves the plot on the ID
			folder.
		alpha : float, opational (default = 1)
			Parameter that defines the separation scaling between timeseries
		similarity_metric: string, optional (default='pearson')
			Choose between a similarity metric between the target and the feature
		ext : str, optional (default 'png')
			format extension for the plots
		output_filename : str, optional (default 'features')
		 	Name of the plot file if 'save' mode is set.
		'''

		if data is None:
			data = self.data
		elif isinstance(data, (Load, LoadFromTemplate)):
			pass
		else:
			print('Data object not detected. Please verify input.')

		# Checking timeseries boundaries
		if start_period and start_period in data.target[id_].index:
			pass
		elif start_period is None:
			start_period = data.target[id_].index[0]
		else:
			print('Unable to identify start_period {0} as valid start reference.\
			please review'.format(start_period))
			return
		if end_period and end_period in data.target[id_].index:
			pass
		elif end_period is None:
			end_period = self.data.target[id_].index[-1]
		else:
			print('Unable to identify end_period {0} as valid end reference.\
			please review'.format(end_period))
			return

		if not os.path.isdir('{0}/{1}'.format(self.folder_main, id_)):
			gen_folder('{0}/{1}'.format(self.folder_main, id_))
		metric_values = []
		if id_ in data.id:
			target = data.target[id_][start_period:end_period].values
			features = data.features[id_][start_period:end_period]
			indices =  data.features[id_][start_period:end_period].index.values

			if feature_names is None:
				feature_names = list(features)

			metric_values = []
			target  = target/np.max(target)
			index = []
			for i, col in enumerate(feature_names):
				series = features[col].values

				nans = np.isnan(series)

				if np.sum(nans) > 0:
					print('WARNING! {0} series in id:{1} contains {2} NaNs'.format(col, id_, np.sum(nans)))

				x, y = timeseries_rmv_values(series.ravel(), target.ravel(), [NAN_TO_VALUE], verbose=False)
				y, x = timeseries_rmv_values(y.ravel(), x.ravel(), [NAN_TO_VALUE], verbose=False)
				v = scipy.stats.pearsonr(x, y)[0]


				metric_values.append(scipy.stats.pearsonr(x, y)[0])
				index.append(i*alpha)


			if ordering == 'unordered':
				pass
			elif ordering == 'decreasing':
				values_ordered = []
				names_ordered = []
				for v, n in sorted(zip(metric_values, feature_names), key= lambda x: x[0], reverse=True):
					values_ordered.append(v)
					names_ordered.append(n)
				metric_values = values_ordered
				feature_names = names_ordered
			elif ordering == 'increasing':
				values_ordered = []
				names_ordered = []
				for v, n in sorted(zip(metric_values, feature_names), key= lambda x: x[0]):
					values_ordered.append(v)
					names_ordered.append(n)
				metric_values = values_ordered
				feature_names = names_ordered

			plt.barh(index, metric_values, alpha/2, alpha=.6)
			plt.title('Similarity between target and features with metric = {0}'.format(similarity_metric))
			plt.yticks(index, feature_names)
			plt.grid(linewidth=.5)
			plt.gca().spines['top'].set_visible(False)
			plt.gca().spines['right'].set_visible(False)

			if int(len(feature_names)/15) == 0:
				vsize = 5
			else:
				vsize = 5*(int(len(feature_names)/15))

			plt.gcf().set_size_inches([10,vsize])

			#plt.subplots_adjust(left=0.8, bottom=0.08, right=0.90, top=0.97, wspace=.20, hspace=.20)
			if mode == 'show':
				plt.show()
			if mode == 'save':
				plt.subplots_adjust(left=0.15, bottom=0.08, right=0.90, top=0.97, wspace=.20, hspace=.20)
				plt.savefig('{0}/{1}/{2}.{3}'.format(self.folder_main, id_, output_filename,ext),fmt=ext)
			plt.close()

		else:
			print('ID not detected within dataObject')



def  mat_metricColumn(table_pd, metric, season, n_models, verbose):
	# Check if metric is in
	metrics_and_models = list(table_pd['Metric'].values)
	season_vals = table_pd[season].values

	if verbose == True:
		print('Table metric and models list :  \n', metrics_and_models)
		print('Season Values : \n ', season_vals)

	if metric in metrics_and_models:

		i = metrics_and_models.index(metric)

		metric_column = season_vals[i+1:i+n_models+1]
		if  verbose == True:
			print('Index for {0} = {1}  \n Table list = {2} \n \n '.format(metric, i, metrics_and_models))

		return metric_column

	else:
		print('Metric was not found. Please check your spelling or availability in table.')
		return []


def mat_getRank(model_names, metric_column, metric_name, verbose = False, gold_std = []):
	n_models = len(model_names)
	ind = range(0,n_models)
	orig_values = metric_column
	# Sorted  default ordering is minimum to maximum. For correlations we look for highest positive (??).
	if metric_name == 'PEARSON':
	    metric_column = np.abs(metric_column)*(-1)

	# To compare RMSEs values have to be normalized based on gold standard's MS value.
	if metric_name == 'RMSE':
	    metric_column /= np.sqrt(np.mean(np.power(gold_std,2)))


	ranking_tuple = [(i, metric) for (metric, i) in sorted(zip(metric_column, ind), key=lambda pair: pair[0])]
	indices_and_values = map(list, zip(*ranking_tuple))

	ord_ind = indices_and_values[0]

	ranking = [model_names[v] for i,v in enumerate(ord_ind)]


	if metric_name == 'PEARSON':
	    values = np.abs(orig_values[ord_ind])

	if metric_name == 'RMSE' or metric_name == 'MAPE':
		values = metric_column[ord_ind]

	return ranking, values




def bar_graph(x_names , vals, ylab, title, saving_dir ):
	fig = plt.figure() # CHANGE SO IT PLOTS EVERYTHING IN ONE GRAPH, GIVE FORMAT
	objects = x_names
	y_pos = np.arange(len(objects))
	performance = vals

	plt.bar(y_pos, performance, align='center', alpha=0.8)
	plt.xticks(y_pos, objects, rotation=90)

	plt.ylabel(ylab)
	plt.title(title, fontsize = 16)
	plt.grid(linestyle = 'dotted', linewidth = .8)
	fig.set_size_inches(12, 9)
	plt.legend()

	fig.savefig(saving_dir + '.png', format='png', dpi=300)
	plt.close()

def mse(predictions, targets):
	# mean square error
		return ((predictions - targets) ** 2).mean()

def order_tuple(v1,v2, type = 'ascending'):
	v_tuple = [(a1, a2) for (a1, a2) in sorted(zip(v1, v2), key=lambda pair: pair[0])]
	v1_and_v2_ordered = map(list, zip(*v_tuple))
	v1_ordered = v1_and_v2_ordered[0]
	v2_ordered = v1_and_v2_ordered[1]
	return v1_ordered, v2_ordered

def color_assign(name_list,color_list):

	default_color = 'blue'
	color_dict = {}
	counter = 0
	while  np.size(color_list) < np.size(name_list):
		color_list.append(default_color)
		counter += 1

	if counter > 0:
		print('Name List in color_assign() has a bigger size than color list, {0} default colors added.'.format(counter))

	for i, name in enumerate(name_list):
		color_dict[name] = color_list[i]

	return color_dict


def get_places(ranks, series_names, ordered=False):

    places_dict = {}
    for metric, matrix in ranks.items():

        places_mat = []

        for i, r in enumerate(matrix):
            row_mat = []
            if metric == 'RMSE':
                reverse = False
            if metric == 'PEARSON':
                reverse = True

            if ordered:
                ordered_places  =  list( sorted(zip(series_names, list(r)) , key = lambda x : x[1], reverse=True))
            else:
                ordered_places  =  list(zip(series_names, list(r)))


            for j, (mod, val) in enumerate(reversed(ordered_places)):
                if j == 0:
                    row_mat.append( (mod, np.sum(r)) ) # all models
                elif j > 0:
                    row_mat.append( (mod, row_mat[j-1][1]-ordered_places[-j][1]) )


            places_mat.append(copy.copy(row_mat))
        places_dict[metric] = places_mat

    return places_dict
