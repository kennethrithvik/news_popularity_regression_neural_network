import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.externals import joblib


def get_data():
	df = pd.read_csv('csv/prep_data_norm.csv')
	X = df.iloc[:,:44]
	y = df.iloc[:,44:]
	
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.22,random_state=42)
	return (X_train,y_train),(X_test,y_test)

def get_folds(n=4):
	df = pd.read_csv('csv/prep_data_norm.csv')
	X = df.iloc[:,:44]
	y = df.iloc[:,44:]
	
	folds = KFold(n_splits=n,random_state=42,shuffle=True)
	for train_index, test_index in folds.split(X,y):
		X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
		y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]
		yield (X_train,y_train),(X_test,y_test)

def train_and_test(model,data):
	(X_train,y_train),(X_test,y_test) = data

	model.fit(X_train,y_train)
	y_pred = model.predict(X_test)

	mse = mean_squared_error(y_test,y_pred)
	r2 = r2_score(y_test,y_pred)
	return mse,r2

def train_and_test_on_folds(model,n=4):
	gen = get_folds()
	all_mse,all_r2 = [],[]

	for i in range(n):
		data = gen.next()
		mse,r2 = train_and_test(model,data)
		print mse,r2
		all_mse.append(mse)
		all_r2.append(r2)

	print all_mse
	print all_r2

	mse = np.mean(all_mse)
	r2 = np.mean(all_r2)
	
	print mse,r2
	return mse,r2

def save_model(model,name):
	joblib.dump(clf, 'models/'+name+'.pkl')

def load_model(name):
	clf = joblib.load('models/'+name+'.pkl')
	return clf

def regression_model_A():
	# 0.150882809428 0.0760153592224
	model = MLPRegressor(
		hidden_layer_sizes=(64,),
		activation = 'tanh',
		solver='sgd',
		alpha=0.0001,
		batch_size ='auto',
		learning_rate = 'adaptive',
		learning_rate_init = 0.01,
		max_iter=10000,
		)
	return model
	

def regression_model_B():
	# 0.150265707345 0.0798210711997
	model = MLPRegressor(
		hidden_layer_sizes=(64,64),
		activation = 'tanh',
		solver='sgd',
		alpha=0.0001,
		batch_size ='auto',
		learning_rate = 'adaptive',
		learning_rate_init = 0.01,
		max_iter=10000,
		)
	return model

def regression_model_C():
	# 0.148671391315 0.0895433165118
	model = MLPRegressor(
		hidden_layer_sizes=(64,64),
		activation = 'relu',
		solver='sgd',
		alpha=0.0001,
		batch_size ='auto',
		learning_rate = 'adaptive',
		learning_rate_init = 0.01,
		max_iter=10000,
		)
	return model

def regression_model_D():
	# 0.153410696965 0.060968906771
	model = MLPRegressor(
		hidden_layer_sizes=(64,64),
		activation = 'relu',
		solver='adam',
		alpha=0.0001,
		batch_size ='auto',
		learning_rate = 'adaptive',
		learning_rate_init = 0.01,
		max_iter=10000,
		)
	return model

def regression_model_E():
	# 0.145775867623 0.107381891391
	model = MLPRegressor(
		hidden_layer_sizes=(128,128),
		activation = 'relu',
		solver='adam',
		alpha=0.0001,
		batch_size ='auto',
		learning_rate = 'adaptive',
		learning_rate_init = 0.001,
		max_iter=10000,
		)
	return model

def regression_ensemble_model():
	pass


models = [
	regression_model_A(),
	regression_model_B(),
	regression_model_C(),
	regression_model_D(),
	regression_model_E()
	]

for model in models:
	print model
	mse,r2 = train_and_test_on_folds(model,n=4)
	print mse,r2
