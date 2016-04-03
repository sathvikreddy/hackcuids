import csv
import numpy as np
import pandas as pd
with open('train.csv','rb') as csvfile:
	df = pd.read_csv(csvfile)
	a = df.iloc[:,[1,2,3]]
	print len(a['protocol_type: symbolic.'].append(a['flag: symbolic.'])