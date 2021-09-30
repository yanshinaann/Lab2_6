import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas import Series
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil import parser
from dateutil.relativedelta import relativedelta


from sklearn.linear_model import LinearRegression


df = pd.read_csv('fancy.csv', sep=';')
series = Series(df.values[:, 1])
initial_time = range(len(series))


def plot_initial():  # Initial time sequence
    plt.plot(initial_time, df.values[:, 1])
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.title("Initial time sequence")
    plt.show()


def plot_trend(win_width: int = 12):
    data_rolling = df.rolling(window=win_width, center=True).mean()
    plt.plot(initial_time, data_rolling)
    plt.show()


res = seasonal_decompose(series, model='additive', period=12)
res.plot(trend=False, seasonal=False)
plt.show()

plot_initial()

regression = LinearRegression()
x, y = np.reshape(initial_time, (-1, 1)), df.values[:, 1]
regression.fit(x, y)

result = regression.predict(np.reshape(initial_time, (-1, 1)))

plt.scatter(initial_time, df.values[:, 1].flatten(),  color='black')
plt.plot(initial_time, result, color='blue', linewidth=3)
plt.title("Simple Linear Regression Model Without Prediction")
plt.show()


new_time_period = range(len(initial_time), len(initial_time) + 8)
new_result = regression.predict(np.reshape(new_time_period, (-1, 1)))

plt.scatter(initial_time, df.values[:, 1].flatten(),  color='black')
plt.plot(new_time_period, new_result, color='green', linewidth=3)
plt.title("Simple Linear Regression Model With Prediction")
plt.show()

plt.scatter(initial_time, df.values[:, 1].flatten(),  color='black')
plt.plot(initial_time, result, color='blue', linewidth=3)
plt.plot(new_time_period, new_result, color='green', linewidth=3)
plt.title("Simple Linear Regression Model With Prediction")
plt.show()


in_time = parser.parse(df.values[0, 0])
print("result\n==============\n")
for i, val in enumerate(result):
    print(f"date: {(in_time + relativedelta(months=i)).strftime('%m/%Y')}, value: {round(val, 2)}")


print("predicted\n==============")
for i, val in enumerate(new_result):
    print(f"date: {(in_time + relativedelta(months=len(initial_time) + i)).strftime('%m/%Y')}, value: {round(val, 2)}")