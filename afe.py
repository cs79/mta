# automated feature engineering - functions

#=========#
# Imports #
#=========#

import pandas as pd
import numpy as np  # not sure if we will use
from fastai.structured import add_datepart
import warnings, copy

#===========#
# Functions #
#===========#

#-----------#
# Utilities #
#-----------#

def check_input(input):
    '''
    Ensure that input is Pandas DataFrame format ready for time series analysis.
    '''
    ipt = copy.deepcopy(input)
    if not type(ipt) == pd.DataFrame:
        if type(ipt) == pd.Series:
            ipt = ipt.to_frame()
        else:
            # attempt to construct DataFrame
            ipt = pd.DataFrame(ipt)
    if not type(ipt.index) == pd.DatetimeIndex:
        try:
            ipt = set_date_index(ipt)
        except:
            warnings.warn('Unable to coerce index to DatetimeIndex; \
                           attempting to find date column to use as index')
            try:
                dc = find_date_col(ipt)
                ipt[dc] = [pd.Timestamp(i) for i in ipt[dc]]
                ipt.set_index(dc, inplace=True)
            except:
                # this is slightly redundant as find_date_col will raise
                raise TypeError('No column could be coerced to DatetimeIndex')
    ipt.index.name = 'Date'  # set consistent index name
    return ipt


def set_date_index(df):
    '''
    Attempt to find and set a date index, possibly within a MultiIndex.
    '''
    assert type(df) == pd.DataFrame, 'something is horribly wrong'
    idf = df.copy()
    cand = None
    if not type(df.index) == pd.MultiIndex:
        idf.index = pd.DatetimeIndex(df.index)  # attempt in-place coercion
    else:
        for i in range(len(df.index.levels)):
            try:
                idf.index = pd.DatetimeIndex(df.index.levels[i])
                cand = i
            except:
                pass
        if cand is None:
            raise TypeError('Index could not be coerced to DatetimeIndex')
    return idf

def find_date_col(df, lazy=True):
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
        if len(str(int(tc)))!= 8:
            continue
        # try coercion to Timestamp
        try:
            _ = pd.Timestamp(tc)
            cand = col
        except:
            pass
    if cand is not None:
        if lazy:
            return cand
        try:
            idx = [pd.Timestamp(i) for i in df[cand]]
        except:
            pass
    if idx is None:
        raise TypeError('No datetime-compatible columns found to use as index')
    return cand

#-------------------------------#
# Feature Engineering Functions #
#-------------------------------#

'''
IMPORTANT:

All _ functions called by build_features should not modify the passed df and return entirely
Rather they should only return the ADDED columns, so that these can be composed in a modular fashion
If we end up wanting to chain some of them, we can chain only those we want to chain
No need to default to "all" or to repeatedly passing the same list of columns to modify in each
This also keeps individual memory chunks smaller until joined at the end
'''

def build_features(df, **kwargs):
    '''
    Convenience wrapper to build various time series features.
    '''
    # unpack kwargs:
    cols    = kwargs.get('cols', None)
    maxlags = kwargs.get('maxlags', 5)  # match default for _add_AR

    # do things from other functions here, just do them intelligently
    cdf = check_input(df)
    idf = cdf.join(fai_dps(cdf), how='outer')
    idf = idf.join(streak(cdf, cols=cols), how='outer')
    idf = idf.join(add_AR(cdf, cols=cols, maxlags=maxlags), how='outer')
    # etc.

    return idf


def fai_dps(df):
    '''
    Wrapper for add_datepart since we index the date column.
    '''
    idf         = df.copy()
    idf['Date'] = idf.index
    add_datepart(idf, 'Date')
    return idf.drop(df.columns, axis=1)


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


def streak(df, cols=None):
    '''
    '''
    idf = pd.DataFrame(index=df.index)
    if cols is None:
        cols = df._get_numeric_data().columns  # won't be perfect; watch for datepart
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


def add_AR(df, cols=None, maxlags=5):
    '''
    Create lagged versions of each specified (numeric) column, up to maxlags.
    '''
    idf = pd.DataFrame(index=df.index)
    if cols is None:
        cols = df._get_numeric_data().columns
    for col in cols:
        for i in range(1, maxlags):
            idf[col + 'AR{}'.format(i)] = df[col].shift(i)
    return idf


def _reversals(df, cols=None, window=65):
    '''
    Rolling count of directional reversals within a sliding window.
    '''
    return None
