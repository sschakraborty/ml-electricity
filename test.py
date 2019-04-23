import pandas as pd
import datetime
from sklearn.neural_network import MLPRegressor
import pickle

PATH = 'data.csv'
IN = 'context.nn'

print('Using path %s for test data' % PATH)
print('Deserializing model from %s' % IN)
nn = pickle.load(open(IN, 'rb'))
print('Model deserialized and loaded...')

def flatten_date(date_str):
	dt = datetime.datetime.strptime(date_str, '%d/%m/%y %H:%M')
	return [dt.month, dt.day, dt.hour, dt.minute]

def create_test(df, indices):
	master = []
	for i, row in df[indices].iterrows():
		it = []
		for index in indices:
			it.append(row[index])
		master.append(it)
	return master

#Read the data
print('Beginning to read CSV data from %s' % PATH)
df_all = pd.read_csv(PATH, header=None)
print('Data read successfully')
indices = [4, 5, 6, 9, 10, 13, 14, 15, 16]

# Pull timestamp from extracted dataframe
print('Processing and staging dataframe...')
df_input = df_all[0].tolist()
df_input = list(map(flatten_date, df_input))
df_output = create_test(df_all, indices)
print('Staging complete')

result = nn.score(df_input, df_output)
print('------------------------------')
print('Training error (score) from generated model')
print(result)