import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import math

data=pd.read_csv("Forest_fire.csv")
data=np.array(data)
x=data[1:,1:-1]
y=data[1:,-1]
y=y.astype('int')
x=x.astype('int')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)
pickle.dump(log_reg,open('ritz.pkl','wb'))
model=pickle.load(open('ritz.pkl','rb'))
