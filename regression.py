# Exercise of linear regression in python
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# Read from a CSV file
# Test git push
data= pd.read_csv("BD.csv")
print(data.head)

popSize = data.loc[:, ['population']].to_numpy()
print(popSize)
year = data.loc[:, ['year']].to_numpy()
print(year)
popSizeLog = np.log(popSize)
print(popSizeLog)

model_LR = linear_model.LinearRegression()
model_LR.fit(year, popSizeLog)
print("Slope of the linear regression line is ", model_LR.coef_)
print("Growth factor (annual) is ", np.exp(model_LR.coef_))
print("Coefficient of correlation is ", model_LR.score(year, popSizeLog))
plt.scatter(year, popSizeLog)
plt.plot(year, model_LR.predict(year))
plt.xlabel("Year")
plt.ylabel("Log PopSize")
plt.show()