# tests

# activate py36
import pandas_datareader.data as web
import os, datetime

PATH = 'C:/Users/cloud/Dropbox/Projects/mta'
os.chdir(PATH)
from afe import *

start = datetime.datetime(2000,1,1)
end   = datetime.datetime(2017,12,31)
f     = web.DataReader('F', 'morningstar', start, end)
#f.index = f.index.levels[1]

c = check_input(f)

t = build_features(f)
