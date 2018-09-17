#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 18:04:51 2018

@author: this_guy
"""

from multiprocessing import cpu_count, Pool
import os
import pandas as pd
import numpy as np
import time


#Change directory to contest directory
os.chdir('/Users/this_guy/Documents/Kaggle_contests/RStudioGoog')

#Number of CPU cores on your system
cores = cpu_count()

# Use all but the one core for paralell computation
partitions = cores-1


# Create helper function to parse json columns
# https://stackoverflow.com/questions/20680272/parsing-a-json-string-which-was-loaded-from-a-csv-using-pandas
def CustomParser(data):
    import json
    j1 = json.loads(data)
    return j1


# Partition a pandas dataframe for multithreaded function application
# http://www.racketracer.com/2016/07/06/pandas-in-parallel/
def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

# Parallelize the unpacking of the json columns
def unpack_columns(data):        
    data[sorted(data['device'][0].keys())] = data['device'].apply(pd.Series)
    data[sorted(data['geoNetwork'][0].keys())] = data['geoNetwork'].apply(pd.Series)
    data[sorted(data['totals'][0].keys())] = data['totals'].apply(pd.Series)
    data[sorted(data['trafficSource'][0].keys())] = data['trafficeSource'].apply(pd.Series)
    return data


def clean(dataset):
    timer = time.time()
    # Read in data (explicitly read fullVisitorId as a string as specified in docs)
    df = pd.read_csv('./data/' + dataset + '.csv.zip', 
                       converters={'device':CustomParser,
                                   'geoNetwork':CustomParser,
                                   'totals':CustomParser,
                                   'trafficSource':CustomParser},
                         dtype={'fullVisitorId': 'str'})
    
        
    df = parallelize(df, unpack_columns)
    
    # Drop the columns that have been unpacked
    for column in ['device', 'geoNetwork', 'totals', 'trafficSource']:
        df = df.drop(column, axis=1)
    
    df['visitNumber'] = df['visitNumber'].astype('int16')
    df['totals_hits'] = df['totals_hits'].astype('int16')
    df['totals_pageviews'] = df['totals_pageviews'].astype('float64')
    df['trafficSource_adwordsClickInfo.page'] = df['trafficSource_adwordsClickInfo.page'].astype('float64')
    df['totals_newVisits'] = df['totals_newVisits'].astype('float64')
    df['totals_bounces'] = df['totals_bounces'].astype('float64')

    if 'totals_transactionRevenue' in df.columns:
        df['totals_transactionRevenue'] = df['totals_transactionRevenue'].fillna(0).astype('int64')

    df.to_pickle('./data/' + dataset + '_clean.pkl')
    
    print(dataset + " pipeline completed in {}s".format(time.time() - timer))
    print(dataset + " shape: {}".format(df.shape))
    
clean('train')
clean('test')