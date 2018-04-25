# automated feature engineering - functions

#=========#
# Imports #
#=========#

import pandas as pd
import numpy as np  # not sure if we will use
from fastai.structured import add_datepart
import warnings

#===========#
# Functions #
#===========#

#-----------#
# Utilities #
#-----------#

def _check_input(input):
    '''
    Ensure that input is Pandas DataFrame format ready for time series analysis.
    '''
    if not type(input) == pd.DataFrame:
        if type(input) == pd.Series:
            input = input.to_frame()
        else:
            # attempt to construct DataFrame
            input = pd.DataFrame(input)
    if not type(input.index) == pd.core.indexes.datetimes.DatetimeIndex:
        # try to coerce in case it is already dates ?
        try:
            input.index = pd.DatetimeIndex(input.index)
        except:
            warnings.warn('Unable to coerce index to DatetimeIndex; \
                           attempting to find date column to use as index')
            # use _find_date_col() to try to guess, with warning
            try:
                input.set_index(_find_date_col(input), inplace=True)
            except:
                warnings.warn('Unable to set proper DatetimeIndex')
                return None
    return input

def _find_date_col(df, lazy=True):
    '''
    Attempt to find date column to use as DatetimeIndex.
    '''
    assert type(df) == pd.DataFrame, 'something is horribly wrong'
    cand = None
    idx  = None
    for col in df.columns:
        # check some "obvious" cases to skip
        tc = df[col].iat[0]
        if type(tc) == float:
            continue
        if type(tc) == int:
            if len(str(tc)) != 8:
                continue
        # try coercion to Timestamp
        try:
            _ = pd.tslib.Timestamp(df[col].iat[0])
            cand = col
        except:
            pass
    if cand is not None:
        if lazy:
            return cand
        try:
            idx = [pd.tslib.Timestamp(i) for i in df[cand]]
        except:
            pass
    if idx is None:
        raise TypeError('No datetime-compatible columns found to use as index')
    return cand

#-------------------------------#
# Feature Engineering Functions #
#-------------------------------#

# implicit - add_datepart is present

def build_features(df):
    '''
    Wrapper to build various time series features.
    '''
    df = _check_input(df)
    idf = df.copy()

    # do things from other functions here, just do them intelligently
    idf = _fastai_dateparts(idf)


    return idf

def _fastai_dateparts(df):
    '''
    Wrapper for add_datepart since we index the date column.
    '''
    idf         = df.copy()
    idf['Date'] = idf.index
    add_datepart(idf, 'Date')
    return idf

def _calc_events(df):
    '''
    Possibly too generic - may be wrapper for other things
    '''

    return None

def _ttg(df, cols=None, percs=[.05, .1, .2, .25, .5]):
    '''
    This may be a specific case of "time since event"
    '''
    idf = df.copy()
    if cols is None:
        idf = idf._get_numeric_data()
    else:
        idf = idf[cols]
    # create new event columns for each relevant col; increment while gaining perc
    return None

def _streak(df, cols=None):
    '''
    '''
    idf = df.copy()
    if cols is None:
        cols = idf._get_numeric_data().columns  # won't be perfect; watch for datepart
    for col in cols:
        idf[col + '_streak'] = _col_streak(list(df[col]))
    return idf

'''
TODO:
try to create a numpy vectorized version of the below that uses sign of difference instead of looping and subtracting each while keeping track of direction
'''
def _col_streak(l):
    assert type(l) == list, 'l must be a list'
    tr = [0]
    sl = ['z']
    for i in range(1, len(l)):
        # up case
        if l[i] >= l[i-1]:
            if sl[i-1] in ('z', 'u'):
                tr.append(tr[i-1] + 1)
            else:
                tr.append(1)
            sl.append('u')
        # down case
        else:
            if sl[i-1] in ('z', 'd'):
                tr.append(tr[i-1] - 1)
            else:
                tr.append(-1)
            sl.append('d')
    return tr
