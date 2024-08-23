import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

house_data = pd.read_csv('Housing_Data.csv')

#house_data.hist(bins=50,figsize=(20,25))
#plt.show()

# Splitting the data into training and testing data
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state = 42)
for train_ind,test_ind in split.split(house_data,house_data['CHAS']):
    train_set = house_data.loc[train_ind]
    test_set = house_data.loc[test_ind]

# Checking the distribution of the data
#print(train_set['CHAS'].value_counts())
#print(test_set['CHAS'].value_counts())


# Checking the correlation of the data
corr_mat = house_data.corr()
#print(corr_mat['MEDV'].sort_values(ascending=False))
from pandas.plotting import scatter_matrix
attributes = ['MEDV','RM','ZN','LSTAT']
scatter_matrix(house_data[attributes],figsize=(12,8))
#plt.show()

# Preparing the data for machine learning
house_data = train_set.drop("MEDV",axis=1) # in Pandas, axis=1 means column and axis=0 means row
house_data_labels = train_set['MEDV'].copy()

# Handling missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(house_data)
#print(imputer.statistics_)
X = imputer.transform(house_data)
housing = pd.DataFrame(X,columns=house_data.columns)
#print(housing.describe())

# Creating a pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler())
])
housing_tr = my_pipeline.fit_transform(house_data)

# Selecting a model
from sklearn.ensemble import RandomForestRegressor  
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_tr,house_data_labels)

# Evaluating the model
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_tr)
mse = mean_squared_error(house_data_labels,housing_predictions)
rmse = np.sqrt(mse)
#print(rmse)

# Using better evaluation technique - Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_tr,house_data_labels,scoring='neg_mean_squared_error',cv=10)
rmse_scores = np.sqrt(-scores)
#print(rmse_scores)

def print_scores(scores):
    print("Scores: ",scores)
    print("Mean: ",scores.mean())
    print("Standard Deviation: ",scores.std())
#print_scores(rmse_scores)

# Saving the model
from joblib import dump,load
dump(model,'House_Price_Prediction.joblib')

# Testing the model
X_test = test_set.drop("MEDV",axis=1)
Y_test = test_set['MEDV'].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
#print(final_rmse)

# Using the model
model = load('House_Price_Prediction.joblib')
print("Enter the values of the following features: ")
features = []
features_name = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

for name in features_name:
    value = float(input(f"Enter value for {name}: "))
    features.append(value)

input_features = pd.DataFrame([features],columns=features_name)
final_features = my_pipeline.transform(input_features)
Predicted_price  = model.predict(final_features)
print(f"The predicted price of the house is {Predicted_price[0]}")
