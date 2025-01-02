import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

data=fetch_california_housing(as_frame=True)
df=data.frame
df.head(5)

x=df.drop("MedHouseVal",axis=1)
y=df["MedHouseVal"]

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,Y_train)

y_pred=model.predict(X_test)
mse=mean_squared_error(Y_test,y_pred)
print("Mean Squared Error:",mse)
