# deep AR / ratchet

#=========#
# Imports #
#=========#

import pandas as pd
# sklearn models to include with Ratchet


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
        # etc. etc.



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
