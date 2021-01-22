import pandas as pd

data = pd.read_csv("nifty.csv")
x=data.iloc[1:,0:1].values
y=data.iloc[1:,-1:].values

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x,y)


from sklearn.preprocessing import PolynomialFeatures
polynom = PolynomialFeatures(degree =2)
x_polynom = polynom.fit_transform(x)

polyreg=LinearRegression()
polyreg.fit(x_polynom,y)

print(polyreg.predict(polynom.transform([[2021]])))

import pickle
with open('model.pkl','wb') as model_pkl:
    pickle.dump(polyreg,model_pkl)