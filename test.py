import csv
import numpy as np
import pandas as pd
with open('train.csv','rb') as csvfile:
	df = pd.read_csv(csvfile)
	#a = pd.DataFrame( np.random.randn(5,5),columns=list('ABCDE') )
	#print a
	a = df.iloc[:,[1,2,3]]
	#flag: symbolic.print a
	print a['protocol_type: symbolic.'].append(a['flag: symbolic.'])
	#saved_column = df['duration: continuous.']
	#print saved_column
	#print df.columns#print df.columns.values.tolist()
"""train = list(csv.reader(csvfile, delimiter=','))
	traindata = pd.DataFrame(train)
print traindata.groupby(['1','2'])
reader = csv.reader(csvfile)
	f = open('req.csv', 'wb')
	for row in reader:
		f.write(row[1]+','+row[2]+'\n')
	#row_count = sum(1 for row in reader)
	#print row_count
	#for row in reader:
		#print(count(row[1]))"""