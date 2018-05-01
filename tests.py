# tests

# activate py36
import pandas_datareader.data as web
import os, datetime
from sklearn.linear_model import LinearRegression

PATH = 'C:/Users/cloud/Dropbox/Projects/mta'
os.chdir(PATH)
from afe import *
from deepar import *

start = datetime.datetime(2000,1,1)
end   = datetime.datetime(2017,12,31)
f     = web.DataReader('F', 'morningstar', start, end)
#f.index = f.index.levels[1]

c = check_input(f)

t = build_features(f)

r = Ratchet()
mod = LinearRegression()
r.model = mod
r.df = t
r.get_val_set()
r.set_time_grid(n = 10)
