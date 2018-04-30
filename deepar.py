# deep AR / ratchet

#=========#
# Imports #
#=========#

import pandas as pd
import numpy as np
import warnings
# sklearn models to include with Ratchet
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
        self.model = kwargs.get('model', None)
        self.df    = kwargs.get('df', None)
        # etc. etc.

    def __repr__(self):
        infostr = 'Time Series Ratchet \
                   \n{} \
                   \n\nModel:\t{} \
                   '.format('-'*19, self.model, self.df.shape if self.df is not None else 'N/A')
        return infostr

    def set_df(self, df):
        assert type(df) in (pd.DataFrame, np.ndarray)
        if type(df) == np.ndarray:
            self.df = pd.DataFrame(df)
        else:
            self.df = df

    def get_val_set(self, val_pct=15):
        if self.df is None:
            warnings.warn('No data to subset! Load a dataframe to this Ratchet')
        else:
            cutoff = int(len(df) * val_pct / 100)
            self.val_set = self.df[-cutoff:]
            self.trt_set = self.df[:cutoff]

    # TODO: during actual "ratchet" process, need to do train/test split still; assert self.trt_set during ratchet

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
        if run == 'step':
            pts = int(len(self.df)/step_size)
            # default to linear spacing
            for i in range(pts):
                grid[i] = df.index[i]
        else:
            for i in range(len(df)):
                if i%n == 0:
                    grid[i] = df.index[i]
        # TODO: need to leave a buffer at front of time series, OR check it during ratchet to see if there is enough data to build from
        assert grid != {}, 'Failed to set grid'
        self.time_grid = grid


# a DecayEnvelope * a TimeGrid should produce the indices for the models in the ensemble
class DecayEnvelope:
    '''
    This should have a functional form, and relevant parameters for shaping it
    '''
    def __init__(self, **kwargs):
        self.form = kwargs.get('form', None)
        # etc.

class RatchetResults:
    '''
    Maybe -- for storing and combining (ensembling) predictions from various time stepped models
    Possibly want to store other model metadata, not just predictions ?
    '''
    def __init__(self):
        self.preds = {}
