
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

dataset=pd.read_csv("EV_Dataset_Denmark.csv")

dataset.shape

df=pd.DataFrame(dataset.columns)
df

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
dataset['Fabrikant']= label_encoder.fit_transform(dataset['Fabrikant'])
dataset['Model']= label_encoder.fit_transform(dataset['Model'])
dataset['Variant']= label_encoder.fit_transform(dataset['Variant'])
dataset['Reg. nr.']= label_encoder.fit_transform(dataset['Reg. nr.'])
dataset['Første reg. dato']= label_encoder.fit_transform(dataset['Første reg. dato'])
dataset['Stelnummer']= label_encoder.fit_transform(dataset['Stelnummer'])
dataset['Status']= label_encoder.fit_transform(dataset['Status'])
dataset['Type']= label_encoder.fit_transform(dataset['Type'])
dataset['Socioeconomic Behavior Index']= label_encoder.fit_transform(dataset['Socioeconomic Behavior Index'])

dataset.head()

dataset.describe()

dataset.shape

dataset.isnull()

dataset.dropna(subset=["Reg. nr.","Første reg. dato","Stelnummer","Status","Type","Fabrikant","Model","Variant","Battery Capacity (kWh)","EV type (BEV = 1 or PHEV =2)","Avg consumption in a day (KWh)","Distance driven in a day (km)","Initial SOC (kWh)","Socioeconomic Behavior Index"],inplace=True)

dataset.isnull()

dataset.shape

dataset

sns.pairplot(dataset)

f, ax = plt.subplots(figsize=(10, 8))
corr = dataset.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
square=True, ax=ax,annot=True)
plt.show()

df=pd.DataFrame(dataset.columns)
df

#With all parameters; Ytl:Y(total parameters with Linear regression) 't' indicatestotal  'l' indicates linear regression
Yt=dataset["Avg consumption in a day (KWh)"]
X13=dataset[['Fabrikant','Model','Variant','Battery Capacity (kWh)','EV type (BEV = 1 or PHEV =2)','Distance driven in a day (km)','Initial SOC (kWh)','Reg. nr.','Første reg. dato','Socioeconomic Behavior Index','Type','Status','Stelnummer']]

from sklearn.model_selection import train_test_split
X13_train, X13_test, Yt_train, Yt_test = train_test_split(X13, Yt, test_size=0.3, random_state=100)

#LINEAR REGRESSION with all parameters::
from sklearn.linear_model import LinearRegression
lrtl=LinearRegression()
lrtl.fit(X13_train,Yt_train)

lrtl.coef_
predictions_tl=lrtl.predict(X13_test)
predictions_tl

plt.scatter(Yt_test,predictions_tl);
plt.xlabel('Yt_test (True Values)');
plt.ylabel('Predicted_tl Values');

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('MAE',metrics.mean_absolute_error(Yt_test,predictions_tl))
print('MSE',metrics.mean_squared_error(Yt_test,predictions_tl))
print('RMSE',np.sqrt(metrics.mean_squared_error(Yt_test,predictions_tl)))
print('R^2',metrics.explained_variance_score(Yt_test,predictions_tl))

cdf_tl=pd.DataFrame(lrtl.coef_,X13.columns,columns=['Coeff13'])
cdf_tl

#With following seven parameters
Y=dataset["Avg consumption in a day (KWh)"]
X7=dataset[['Fabrikant','Model','Variant','Battery Capacity (kWh)','EV type (BEV = 1 or PHEV =2)','Distance driven in a day (km)','Initial SOC (kWh)']]

df=pd.DataFrame(['Fabrikant','Model','Variant','Battery Capacity (kWh)','EV type (BEV = 1 or PHEV =2)','Distance driven in a day (km)','Initial SOC (kWh)','Avg consumption in a day (KWh)'])
df

sns.pairplot(dataset,x_vars=['Fabrikant','Model','Variant','Battery Capacity (kWh)','EV type (BEV = 1 or PHEV =2)','Distance driven in a day (km)','Initial SOC (kWh)','Avg consumption in a day (KWh)'],y_vars=['Fabrikant','Model','Variant','Battery Capacity (kWh)','EV type (BEV = 1 or PHEV =2)','Distance driven in a day (km)','Initial SOC (kWh)','Avg consumption in a day (KWh)'],kind='scatter')

f, ax = plt.subplots(figsize=(10, 8))
list= ['Fabrikant','Model','Variant','Battery Capacity (kWh)','EV type (BEV = 1 or PHEV =2)','Distance driven in a day (km)','Initial SOC (kWh)','Avg consumption in a day (KWh)']
dataframe=dataset[list]
corr=dataframe.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
square=True, ax=ax,annot=True)
plt.show()

from sklearn.model_selection import train_test_split
X7_train, X7_test, Y_train, Y_test = train_test_split(X7, Y, test_size=0.3, random_state=101)

#LINEAR REGRESSION with 7 parameters::
from sklearn.linear_model import LinearRegression
lr7=LinearRegression()
lr7.fit(X7_train,Y_train)

lr7.coef_

predictions_7=lr7.predict(X7_test)

plt.scatter(Y_test,predictions_7);
plt.xlabel('Y_test (True Values)');
plt.ylabel('Predicted Values');

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('MAE',metrics.mean_absolute_error(Y_test,predictions_7))
print('MSE',metrics.mean_squared_error(Y_test,predictions_7))
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions_7)))
print('R^2',metrics.explained_variance_score(Y_test,predictions_7))

predictions_7

cdf=pd.DataFrame(lr7.coef_,X7.columns,columns=['Coeff'])
cdf

#With Only one parameter i.e., Distrance driven in a day (km)
X1=dataset[['Distance driven in a day (km)']]
Y1=dataset['Avg consumption in a day (KWh)']

from sklearn.model_selection import train_test_split
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression
lr1=LinearRegression()
lr1.fit(X1_train,Y1_train)

lr1.coef_

predictions1=lr1.predict(X1_test)
predictions1

plt.scatter(Y1_test,predictions1);
plt.xlabel('Avg consumption in a day (KWh) (True Values)');
plt.ylabel('Avg consumption in a day (KWh) Predicted1 Values');

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('MAE1',metrics.mean_absolute_error(Y1_test,predictions1))
print('MSE1',metrics.mean_squared_error(Y1_test,predictions1))
print('RMSE1',np.sqrt(metrics.mean_squared_error(Y1_test,predictions1)))
print('R^2(1)',metrics.explained_variance_score(Y1_test,predictions1))

cdf1=pd.DataFrame(lr1.coef_,X1.columns,columns=['Coeff1'])
cdf1

#SUPPORT VECTOR MACHINE(REGRESSION) USING all PARAMETERS::
from sklearn.svm import SVR
svr_model_t = SVR()
svr_model_t.fit(X13_train, Yt_train)
svr_predictions_t=svr_model_t.predict(X13_test)
plt.scatter(Yt_test,svr_predictions_t);
plt.xlabel('Yt_test (SVR_True Values)');
plt.ylabel('SVR_Predicted Values');

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('MAE',metrics.mean_absolute_error(Yt_test,svr_predictions_t))
print('MSE',metrics.mean_squared_error(Yt_test,svr_predictions_t))
print('RMSE',np.sqrt(metrics.mean_squared_error(Yt_test,svr_predictions_t)))
print('R^2',metrics.explained_variance_score(Yt_test,svr_predictions_t))

#SUPPORT VECTOR MACHINE(REGRESSION) with 7 parameters::
from sklearn.svm import SVR
svr_model_7 = SVR()
svr_model_7.fit(X7_train, Y_train)
svr_predictions_7=svr_model.predict(X7_test)
plt.scatter(Y_test,svr_predictions_7);
plt.xlabel('Y_test (SVR_True Values)');
plt.ylabel('SVR_Predicted Values');

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('MAE',metrics.mean_absolute_error(Y_test,svr_predictions_7))
print('MSE',metrics.mean_squared_error(Y_test,svr_predictions_7))
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,svr_predictions_7)))
print('R^2',metrics.explained_variance_score(Y_test,svr_predictions_7))

#SUPPORT VECTOR MACHINE(REGRESSION) USING ONE PARAMETER::
from sklearn.svm import SVR
svr_model_1 = SVR()
svr_model_1.fit(X1_train, Y1_train)
svr_predictions_1=svr_model_1.predict(X1_test)
plt.scatter(Y1_test,svr_predictions_1);
plt.xlabel('Y1_test (SVR_True Values)');
plt.ylabel('SVR_Predicted Values');

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('MAE',metrics.mean_absolute_error(Y1_test,svr_predictions_1))
print('MSE',metrics.mean_squared_error(Y1_test,svr_predictions_1))
print('RMSE',np.sqrt(metrics.mean_squared_error(Y1_test,svr_predictions_1)))
print('R^2',metrics.explained_variance_score(Y1_test,svr_predictions_1))

#Random Forest Algorithm 't' indicates total::
from sklearn.ensemble import RandomForestRegressor
rf_model_t = RandomForestRegressor()
rf_model_t.fit(X13_train, Yt_train)
rf_predictions_t=rf_model_t.predict(X13_test)
plt.scatter(Yt_test,rf_predictions_t);
plt.xlabel('Yt_test (RF_True Values)');
plt.ylabel('RF_Predicted Values');

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('MAE',metrics.mean_absolute_error(Yt_test,rf_predictions_t))
print('MSE',metrics.mean_squared_error(Yt_test,rf_predictions_t))
print('RMSE',np.sqrt(metrics.mean_squared_error(Yt_test,rf_predictions_t)))
print('R^2',metrics.explained_variance_score(Yt_test,rf_predictions_t))

#Random Forest Algorithm with 7 parameters::
from sklearn.ensemble import RandomForestRegressor
rf_model_7 = RandomForestRegressor()
rf_model_7.fit(X7_train, Y_train)
rf_predictions_7=rf_model_7.predict(X7_test)
plt.scatter(Y_test,rf_predictions_7);
plt.xlabel('Y_test (RF_True Values)');
plt.ylabel('RF_Predicted Values');

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('MAE',metrics.mean_absolute_error(Y_test,rf_predictions_7))
print('MSE',metrics.mean_squared_error(Y_test,rf_predictions_7))
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,rf_predictions_7)))
print('R^2',metrics.explained_variance_score(Y_test,rf_predictions_7))

#Random Forest Algorithm with one parameter::
from sklearn.ensemble import RandomForestRegressor
rf_model_1 = RandomForestRegressor()
rf_model_1.fit(X1_train, Y1_train)
rf_predictions_1=rf_model_1.predict(X1_test)
plt.scatter(Y1_test,rf_predictions_1);
plt.xlabel('Y1_test (RF_True Values)');
plt.ylabel('RF_Predicted Values');

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('MAE',metrics.mean_absolute_error(Y1_test,rf_predictions_1))
print('MSE',metrics.mean_squared_error(Y1_test,rf_predictions_1))
print('RMSE',np.sqrt(metrics.mean_squared_error(Y1_test,rf_predictions_1)))
print('R^2',metrics.explained_variance_score(Y1_test,rf_predictions_1))