from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype,is_bool

# load in data
df = pd.read_csv('challenge_train.csv', low_memory = False)
# df.astype(np.float)
y = df['verdict']
X = df.drop(labels = ['md5','verdict'],axis = 'columns')
X.astype(np.float)
#X = X.drop(X.columns[X.columns.str.contains('unnamed',case = False)],axis = 'columns')

# print(X_train.head())

def make_nums_from_bool(df):
	# convert the True/Fals boolean values to 1/0
	for n,c in df.items():
		# print(n,end=" ")
		if is_bool(c):
			df[n] = (df[n] == True).astype(int)


# replacing trojan with 1, clean with 0 in df
y.replace(to_replace={'trojan' : 1,'clean' : 0},inplace=True)
	
		

# make_nums_from_bool(X)

# print(X.head())

def main():
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
	# X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, y_train, test_size=0.2)
	rf = RandomForestRegressor(n_estimators=3, max_depth = 3)
	rf.fit(X_train,y_train)
	print("Validation: ", r2_score(y_test,rf.predict(X_test)))

if __name__ == '__main__':
	main()