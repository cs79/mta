# deep AR / ratchet

#=========#
# Imports #
#=========#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
# sklearn models to include with Ratchet
from sklearn.linear_model import LinearRegression  # use this as test case
# afe functionality to create features to ratchet, maybe? or assume that was done first?

'''
NB DeepAR and Ratchet are basically different ideas - split modules probably
'''

#=========#
# Classes #
#=========#

class Ratchet:
    '''
    this should have as attributes:
        - a DecayEnvelope
        - time_step
        - possibly some sort of results storage class
        - a model choice
        - window (constant sliding, or expanding sliding -- maybe give choice)
        - other things, certainly

    should be able to do things like change time_step and rerun w/out need to change much else, and then get easily comparable results
    '''
    def __init__(self, **kwargs):
        self.model     = kwargs.get('model', None)
        self.df        = kwargs.get('df', None)
        self.y_col     = kwargs.get('y_col', None)     # col to create y data
        self.target    = kwargs.get('target', None)    # actual y data
        self.envelope  = kwargs.get('envelope', None)  # think about how to pass
        self.verbose   = kwargs.get('verbose', True)
        self.trt_set   = None
        self.val_set   = None
        self.time_grid = None
        self.res       = RatchetResults()
        # etc. etc.

    def __repr__(self):
        infostr = 'Time Series Ratchet \
                   \n{}\n \
                   \nModel:\t{} \
                   \nData:\t{} \
                   \nTrT:\t{} \
                   \nVal:\t{} \
                   \nY:\t{}\n \
                   \nData Load: \
                   \n{}\n \
                   \nDF:\t{} \
                   \nTarget:\t{} \
                   \nGrid:\t{} \
                   '.format('-'*19, \
                            self.model, \
                            self.df.shape if self.df is not None else 'N/A', \
                            self.trt_set.shape if self.trt_set is not None else 'N/A', \
                            self.val_set.shape if self.val_set is not None else 'N/A', \
                            self.target.shape if self.target is not None else 'N/A', \
                            '-'*9, \
                            True if self.df is not None else False, \
                            True if self.target is not None else False, \
                            True if self.time_grid is not None else False)
        return infostr

    def set_df(self, df):
        assert type(df) in (pd.DataFrame, np.ndarray)
        if type(df) == np.ndarray:
            self.df = pd.DataFrame(df)
        else:
            self.df = df

    # TODO: Make this a utility function rather than a method
    # Can use to create target values ahead of time
    def set_target(self, n=1):
        '''
        Generate y-values from an underlying dataset.
        '''
        assert self.df is not None
        assert self.y_col is not None
        self.target = self.df[self.y_col].shift(-n).to_frame()
        # TODO: make this able to generate multiple targets; make Ratchet able to store multiple targets, make Ratchet.ratchet() able to fit vs. multiple targets

    def get_val_set(self, val_pct=0.15):
        assert val_pct >= 0
        assert val_pct <= 1
        if self.df is None:
            warnings.warn('No data to subset! Load a dataframe to this Ratchet')
        else:
            cutoff = int(len(self.df) * val_pct)
            self.val_set = self.df[-cutoff:]
            self.trt_set = self.df[:-cutoff]

    # TODO: during actual "ratchet" process, need to do train/test split still
    def tts(self, df, train_pct=0.6):
        assert train_pct >= 0
        assert train_pct <= 1
        cutoff = int(len(df) * train_pct)
        return df[:cutoff], df[cutoff:]

    def set_decay(self, de):
        assert type(de) == DecayEnvelope
        self.decay = de

    def set_time_grid(self, n=None, step_size=None):
        '''
        Define time grid that will create the time series ensemble.
        '''
        assert self.df is not None, 'Must set df before calculating time grid!'
        run  = None
        grid = {}
        if n is None:
            assert step_size
            run = 'step'
        if step_size is None:
            assert n
            run = 'n'
        if n is not None and step_size is not None:
            raise ValueError('Pass exactly one of n, step_size')
        if run == 'n':
            step_size = int(len(self.df) / n)
            # default to linear spacing
            for i in range(len(self.df)):
                if (i % step_size == 0) & (i != 0):
                    grid[i] = self.df.index[i]
        # TODO: need to leave a buffer at front of time series, OR check it during ratchet to see if there is enough data to build from; also this currently returns n+1 grid points when passing n
        assert grid != {}, 'Failed to set grid'
        self.time_grid = grid

    def ratchet(self):
        assert self.res is not None
        assert self.model is not None
        assert self.trt_set is not None
        assert self.time_grid is not None
        if self.target is None:
            warnings.warn('No target set; attempting to set with default value')
            self.set_target()
        for i in self.time_grid:
            train, test = self.tts(self.trt_set[:i])  # not sure this will work
            fitted = self.model.fit(train, self.target)  # store fitted? or no need?
            self.res.preds[i] = fitted.predict(test)
            self.res.vpreds[i] = fitted.predict(self.val)
            if self.verbose:
                print('Fit model {} for step {} of {}'\
                      .format(self.model, i, max(self.time_grid.keys())))


# a DecayEnvelope * a TimeGrid should produce the indices for the models in the ensemble
class DecayEnvelope:
    '''
    This should have a functional form, and relevant parameters for shaping it
    '''
    allowable_spacing = ('linear', 'log')
    def __init__(self, **kwargs):
        self.form     = kwargs.get('form', None)
        self.spacing  = kwargs.get('spacing', 'linear')
        self.start    = kwargs.get('start', None)
        self.end      = kwargs.get('end', None)
        self.formname = None
        self.weights  = {}
        # etc.

    def __repr__(self):
        s = 'Decay Envelope \
             \n{}\n \
             \nForm:\t\t{} \
             \nSpacing:\t{} \
             '.format('-'*14, \
                      self.formname if self.formname is not None else 'N/A', \
                      self.spacing if self.spacing is not None else 'N/A')
        return s

    def plot(self):
        assert self.form is not None
        xs = np.linspace(self.start, self.end, 100)
        ys = [self.form(i) for i in xs]
        pd.DataFrame({'Y': ys}, index=xs).plot(grid=True)
        plt.show()

    # either have "canned" forms and pass a string as arg, or pass fn as arg
    def set_form(self, form):
        if not callable(form):
            assert type(form) == str, 'Invalid form type'
        if callable(form):
            self.form = form
            self.formname = 'Custom'
        # canned string cases:
        if form == 'linear':
            self.form = lambda x: x
            self.start = 0
            self.end = 1
            self.formname = 'Linear'
        if form == 'tanh':
            self.form = lambda x: np.tanh(x)
            self.start = 0
            self.end = 3  # visual guess
            self.formname = 'Tanh'
        if form == 'sin':
            self.form = lambda x: np.sin(x)
            self.start = 0
            self.end = 1.5
            self.formname = 'Sine'
        if form == 'hyp':
            self.form = lambda x: -x ** 2
            self.start = -1
            self.end = 0
            self.formname = 'Hyperbolic'
        # etc. - research some better options for these and use vetted numbers

    def weight_grid(self, n=None):
        '''
        Get weights based on spacing and number of grid points to match.
        '''
        # can pass len(Ratchet.grid) for n, or allow passing of a grid directly
        assert self.form is not None
        assert self.spacing in self.allowable_spacing
        # TODO: check whether y-values are negative (as they are form some forms... this will screw weights up entirely - need to shift to positive scale)


class RatchetResults:
    '''
    Maybe -- for storing and combining (ensembling) predictions from various time stepped models
    Possibly want to store other model metadata, not just predictions ?
    '''
    def __init__(self):
        self.preds = {}
        self.vpreds = {}
