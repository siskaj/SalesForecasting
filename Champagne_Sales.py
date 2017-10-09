# scipy
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import statsmodels

from pandas import Series
series = Series.from_csv('~/data/champagne.csv', header=0)
split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')


