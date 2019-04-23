import pandas as pd
import datetime
from sklearn.neural_network import MLPRegressor
import pickle
import matplotlib.pyplot as plt

PATH = 'predict.csv'
AC_PATH = 'data.csv'
IN = 'context.nn'

ins_names = ['Solar PV System             ']
ins_names.append('Heat Pump - Inside Unit     ')
ins_names.append('Heat Pump - Outside Unit    ')
ins_names.append('Hot Water Heater            ')
ins_names.append('Lights + fans + stair lights')
ins_names.append('Dryer                       ')
ins_names.append('Washing Machine             ')
ins_names.append('Human Generator - Kitchen   ')
ins_names.append('Remaining level 1 plus + DAS')

indices = [4, 5, 6, 9, 10, 13, 14, 15, 16]

rate = input('Enter cost per unit of consumption: ')
try:
	rate = abs(float(rate))
except:
	print('Entered value is not a valid number')
	exit()

def process(ll):
	return list(map(lambda o: list(map(abs, o)), ll))

def print_pretty(ll, means):
	for l in ll:
		print()
		print('%s \t\t %s \t %s \t %s \t\t %s' % ('Equipment Name / Instrument', 'Mean Consumption', 'Pred. Consumption', 'Mean Cost', 'Pred. Cost'))
		print('---------------------------------------------------------------------------------------------------------------------------')
		show_list = []
		for index in range(len(l)):
			show_list.append((ins_names[index], l[index]))
		show_list.sort(key=lambda o: o[1], reverse=True)
		for index in range(9):
			item = show_list[index]
			print('%s \t\t %.6f \t\t %.6f \t\t %.4f \t\t %.4f' % (item[0], means[index], item[1], means[index] * rate, item[1] * rate))

print('Using path %s for test data' % PATH)
print('Deserializing model from %s' % IN)
nn = pickle.load(open(IN, 'rb'))
print('Model deserialized and loaded...')

print('Reading test data (input) from %s' % PATH)
df = pd.read_csv(PATH, header=None)
print('Read test data')

def listify(df):
	master = []
	for i, row in df.iterrows():
		master.append(row.tolist())
	return master

def create_test(df, indices):
	master = []
	for i, row in df[indices].iterrows():
		it = []
		for index in indices:
			it.append(row[index])
		master.append(it)
	return master

print('Staging input...')
input = listify(df)
print('Staging complete')

print('Reading data for calculating mean')
adf = pd.read_csv(AC_PATH, header=None)
adf = pd.DataFrame(create_test(adf, indices))
adf = adf.mean(axis=0)

print_pretty(process(nn.predict(input)), adf)
plt.plot(nn.loss_curve_)
plt.show()