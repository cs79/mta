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


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def get_windows(df):
    '''
    Get default windows for rolling calcs based on time series frequency.
    '''
    f = pd.infer_freq(df.index)
    # default mapping; "natural window" heuristic which may not be great
    wmap = {'B':   [5, 20, 65, 130, 260],                   \
            'D':   [7, 30, 90, 252, 365],                   \
            'W':   [4, 13, 26, 52, 78, 104],                \
            'M':   [3, 6, 9, 12, 36, 24],                   \
            'BM':  [3, 6, 9, 12, 36, 24],                   \
            'MS':  [3, 6, 9, 12, 36, 24],                   \
            'BMS': [3, 6, 9, 12, 36, 24],                   \
            'Q':   [2, 3, 4, 6, 9, 12],                     \
            'BQ':  [2, 3, 4, 6, 9, 12],                     \
            'QS':  [2, 3, 4, 6, 9, 12],                     \
            'BQS': [2, 3, 4, 6, 9, 12],                     \
            # annual probably not advisable unless you have a really long TS...
            'A':   [2, 3, 4, 5, 10],                        \
            'BA':  [2, 3, 4, 5, 10],                        \
            'AS':  [2, 3, 4, 5, 10],                        \
            'BAS': [2, 3, 4, 5, 10],                        \
            # trickier -- not sure what a good heuristic for these is:
            'BH':  [2, 3, 4],                               \
            'H':   [2, 3, 4, 5, 6, 8, 12],                  \
            'T':   [5, 10, 15, 30, 60],                     \
            'min': [5, 10, 15, 30, 60],                     \
            'S':   [5, 10, 15, 30, 60],                     \
            'L':   [10, 100, 1000],                         \
            'ms':  [10, 100, 1000],                         \
            'U':   [10, 100, 1000, 10000, 100000, 1000000], \
            'us':  [10, 100, 1000, 10000, 100000, 1000000], \
            'N':   [10, 100, 1000, 10000, 100000, 1000000]}
    try:
        tr = wmap[f]
    except:
        tr = generic_windows(len(df))
    return tr


def generic_windows(length, n=5):
    assert type(length) == int, 'length must be an int'
    start = int(round(length * 0.001))  # kind of arbitrary; make this smarter
    stop = int(round(length * 0.05))
    w = np.log10(np.logspace(start, stop, n))
    w = [int(i) for i in w]
    # check for bad values
    if len(w) != len(set(w)):
        warnings.warn('Duplicate windows created; truncating')
        w = list(set(w))
        w.sort()
    if 1 in w:
        warnings.warn('Length 1 window detected; removing')
        w.remove(1)
    if len(w) == 0:
        raise ValueError('Insufficient length to create windows')
    return w


def dedup(df):
    assert type(df) == pd.DataFrame
    idf = df.T.drop_duplicates().T
    diff = set(df.columns) - set(idf.columns)
    if diff != set():
        warnings.warn('Dropped duplicate valued columns: \n{}'.format(diff))
    return idf


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

For windowed functions, may want to use pd.infer_freq() to set default windows
'''

def build_features(df, **kwargs):
    '''
    Convenience wrapper to build various time series features.
    '''
    # unpack kwargs:
    cols    = kwargs.get('cols', None)
    maxlags = kwargs.get('maxlags', 5)  # match _add_AR default
    window  = kwargs.get('window', [65, 130, 260])  # match reversals default

    # do things from other functions here, just do them intelligently
    cdf = check_input(df)
    idf = cdf.join(fai_dps(cdf), how='outer')
    idf = idf.join(streak(cdf, cols=cols), how='outer')
    idf = idf.join(add_AR(cdf, cols=cols, maxlags=maxlags), how='outer')
    idf = idf.join(reversals(cdf, cols=cols, window=window), how='outer')
    # etc.

    # final step should be some kind of deduplication in case we have produced multiple copies of columns somehow (e.g. constants)
    idf = dedup(idf)

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
    idf = pd.DataFrame(index=df.index)
    if cols is None:
        idf = idf._get_numeric_data().columns
    # create new event columns for each relevant col; increment while gaining perc
    return None


def streak(df, cols=None):
    '''
    Create features for consecutive streak in same direction.
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
            idf[col + '_AR{}'.format(i)] = df[col].shift(i)
    return idf


def reversals(df, cols=None, window=None):
    '''
    Rolling count of directional reversals within a sliding window.
    '''
    if window is None:
        window = get_windows(df)
    assert type(window) in (int, list), 'window must be int or list of ints'
    if type(window) == list:
        assert all([type(i) == int for i in window]), 'windows must be ints'
    else:
        window = [window]
    idf = pd.DataFrame(index=df.index)
    if cols is None:
        cols = df._get_numeric_data().columns
    for w in window:
        for col in cols:
            idf[col + '_revs{}'.format(w)] = _col_revs(list(df[col]), w)
    return idf


def _col_revs(l, w):
    assert type(l) == list, 'l must be a list'
    assert type(w) == int, 'w must be an int'
    f = [0]
    s = _col_streak(l)
    for i in range(1, len(s)):
        if sign(s[i]) != sign(s[i - 1]):
            f.append(1)
        else:
            f.append(0)
    return list(pd.Series(f).rolling(window=w).sum().values)


def _rolling_stats(df, cols=None, window=None):
    '''
    Compute various rolling window statistics.
    '''
    # experimental - once this is working, convert other windowed functions
    if cols is None:
        cols = df._get_numeric_data().columns
    if window is None:
        window = get_windows(df)
    if type(window) != list:
        window = [window]
    assert all([type(i) == int for i in window]), 'windows must be ints!'
    idf = pd.DataFrame(index=df.index)
    for w in window:
        roll = df[cols].rolling(w)
        # get other stats, name appropriately, join to idf
        stds = roll.std().rename(columns=lambda x:
                                         x + '_rolling_std_{}'.format(w))
        mns = roll.mean().rename(columns=lambda x:
                                         x + '_rolling_mean_{}'.format(w))
        # TODO: add rolling volatility, corr, pctrank, "diverging std", zscore vs trailing X, etc.
    idf = idf.join(stds, how='outer').join(mns, how='outer')
    return idf


def _expanding_stats(df, cols=None):
    '''
    Compute various expanding window statistics.
    '''
    if cols is None:
        cols = df._get_numeric_data().columns
    return None
