import pandas as pd
import datetime
from sklearn.neural_network import MLPRegressor
import pickle

PATH = 'data.csv'
OUT = 'context.month.nn'

def flatten_date(date_str):
	dt = datetime.datetime.strptime(date_str, '%d/%m/%y %H:%M')
	return [dt.month, dt.day]

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

# Test code - Do not touch
# print(df_input[0], df_input[1])
# print(df_output[0], df_output[1])
# print(len(df_input))
# print(len(df_output))

print('Training a MLPRegressor - hold tight...')
nn = MLPRegressor(solver='adam',
	              alpha=1e-5,
	              learning_rate='adaptive',
	              hidden_layer_sizes=(20, 15),
	              random_state=1,
	              max_iter=1000)
nn.fit(df_input, df_output)
print('Training complete')

print('Saving model')
pickle.dump(nn, open(OUT, 'wb'))
print('Model created, trained and saved successfully')