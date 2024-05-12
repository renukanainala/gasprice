import numpy as np import pandas as pd import os for dirname, _, filenames in os.walk('/kaggle/input'): for filename in filenames: 
print(os.path.join(dirname, filename)) 
 
dataset = pd.read_csv("daily_csv.csv") dataset.head() 
dataset.info() 
 
dataset['year'] = pd.DatetimeIndex(dataset['Date']).year dataset['month'] 
= pd.DatetimeIndex(dataset['Date']).month dataset['day'] 
= pd.DatetimeIndex(dataset['Date']).day Dataset dataset.drop('Date', axis=1, inplace=True) dataset.isnull().any() dataset.isnull().sum() dataset['Price'].fillna(dataset['Price'].mean(),inplace=True) dataset.isnull().any() 
#import the matplotlib library import matplotlib.pyplot as plt 
#plot size fig=plt.figure(figsize=(5,5)) plt.scatter(dataset['day'],dataset['Price'],color='blue' ) #Set the label for the x-axis. plt.xlabel('Day') #Set the label for the y-axis. 
plt.ylabel('Price') #Set a title for the axes. 
plt.title('PRICE OF NATURAL GAS ON THE BASIS OF DAYS OF A MONTH') 
#Place a legend on the axes. plt.legend() 
 
 
import matplotlib.pyplot as plt 
plt.bar(dataset['month'],dataset['Price'],color='green') plt.xlabel('Month') plt.ylabel('Price') 
plt.title('PRICE OF NATURAL GAS ON THE BASIS OF MONTHS OF A YEAR') plt.legend() 
 
import seaborn as sns 
sns.lineplot(x='year',y='Price',data=dataset,color='red') 
 
import seaborn as sns 
sns.lineplot(x='month',y='Price',data=dataset,color='red') 
 
import seaborn as sns 
sns.lineplot(x='day',y='Price',data=dataset,color='red') 
 
fig=plt.figure(figsize=(8,4)) 
plt.scatter(dataset['year'],dataset['Price'],color='purple') plt.xlabel('Month') plt.ylabel('Price') 
plt.title('PRICE OF NATURAL GAS ON THE BASIS OF MONTHS OF A YEAR') plt.legend() 
 
import seaborn as sns sns.pairplot(dataset) plt.show() 
 
x=dataset.iloc[:,1:4].values #inputs y=dataset.iloc[:,0:1].values 
#output price only 
X Y 
from sklearn.model_selection import train_test_split x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) x_train.shape y_train.shape 
#import decision tree regressor 
from sklearn.tree import DecisionTreeRegressor dtr=DecisionTreeRegressor() 
#fitting the model or training the model dtr.fit(x_train,y_train) y_pred=dtr.predict(x_test) y_pred 
 
from sklearn.metrics import r2_score accuracy=r2_score(y_test,y_pred) Accuracy y_p=dtr.predict([[1997,1,7]]) y_p