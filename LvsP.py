import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/MPG.csv')

print(df.columns)
print(df.info())
print(df.drop('name'  , axis =1))
print(df.drop('origin' , axis =1))
#print(df.replace({'origin':{'usa':0 , 'europe':1 , 'japan':2}}))
print(df.corr())
y=df['mpg']
X=df[[ 'weight']]
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y , train_size = 0.7 , random_state = 2425 )
from sklearn.linear_model import LinearRegression
Home=LinearRegression()
print(Home.fit(X_train , y_train))
print(Home.intercept_)
print(Home.coef_)
y_pred = Home.predict(X_test)
from sklearn.metrics import mean_absolute_error,r2_score
print(mean_absolute_error(y_test,y_pred))
print(r2_score(y_test , y_pred))
(plt.scatter(X,y))
(plt.plot(X,Home.predict(X) , "red" ))
print(plt.show())
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly= poly.fit_transform(X)  
lin_reg_2 =LinearRegression()  
lin_reg_2.fit(X_poly, y)  
(plt.scatter(X,y))
(plt.plot(X,lin_reg_2.predict(poly.fit_transform(X)) ))
print(plt.show())
