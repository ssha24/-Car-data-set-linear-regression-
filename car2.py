import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
a=LinearRegression()
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
scl=StandardScaler()
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df=pd.read_excel("Craigslist Car Dataset.xlsx")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
# unique()
#print(df['MODELS'].unique())
for i in df.index:
    if df.loc[i,"make"] == 7 or df.loc[i,"make"]==76:
        df.drop(i, inplace = True)
d={'other':0}
df['cylinders']=df['cylinders'].map(d)
#_______________________________________________price
for i in df.index:
    if df.loc[i,"price"]<1000:
        df.drop(i,inplace=True)
 
#upper 
#df["make"]=df["make"].str.upper()
#print(df["make"].unique())
df[["location","condition","Listed_date","Listed_time","drive","fuel","lat","long","odometer","paint color","price","size","title status","transmission","type","year make model",'cylinders','year','make','re_model','MODELS']]=df[["location","condition","Listed_date","Listed_time","drive","fuel","lat","long","odometer","paint color","price","size","title status","transmission","type","year make model",'cylinders','year','make','re_model','MODELS']].apply(lb.fit_transform)
df["make"]=lb.fit_transform(df["make"])
df["condition"]=lb.fit_transform(df["condition"])
fe=["condition","odometer","make","year"]
x=df[fe]
y=df["price"]
scl.fit_transform(x)
#print(df["make"].unique())
# train test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)
a.fit (x_train,y_train)
y_pred=a.predict(x_test)
print(y_pred)
print('_')
print(y_test)
#--------------------------------------------------------------------
from sklearn.linear_model import Ridge, Lasso
ridge = Ridge().fit(x_train, y_train)
lasso = Lasso().fit(x_train, y_train)
#----------------------------------------------------------------------------------

#print(a.coef_)
#r2=r2_score(y_train,mymodel(x_train))
#print(r2)

from sklearn.metrics import mean_squared_error

rmse =(mean_squared_error(y_test, y_pred))
print(rmse)
print(r2_score(y_test,y_pred))
#-------------------------------------------------------------------
#import seaborn as sns
import matplotlib.pyplot as plt
#sns.heatmap(df.corr(), annot=True)
#plt.show()
#-------------------------------------------------------------------
