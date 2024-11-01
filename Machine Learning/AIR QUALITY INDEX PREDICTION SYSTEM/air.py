import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

data=pd.read_csv("air2.csv")
data.drop(["city","station","date","time","Predominant_Parameter","NH3"],axis=1,inplace=True)
dict_city={'Delhi':0,'Haryana':1,'Uttar_Pradesh':2,'Maharashtra':3,'Karnataka':4,'Madhya Pradesh':5,'West_Bengal':6,'Rajasthan':7,'Punjab':8,'Telangana':9,'Bihar':10,'TamilNadu':11,'Gujarat':12,'Andhra_Pradesh':13,'Kerala':14,'Odisha':15,'Jharkhand':16,'Assam':17,'Chandigarh':18,'Meghalaya':19}
data["state"]=data["state"].apply(lambda x:dict_city[x])
data.fillna(0,inplace=True)
x=data.drop("AQI",axis=1)
y=data["AQI"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,train_size=0.8)

randomforest_model=RandomForestRegressor()
randomforest_model.fit(x_train,y_train)
print("Accuracy of the model is:",randomforest_model.score(x_test,y_test))


pickle.dump(randomforest_model,open('air1.pkl','wb'))


