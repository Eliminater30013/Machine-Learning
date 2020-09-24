import quandl, math
import numpy as np
import pickle
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
style.use('ggplot')
# A feature/features are input and the label is the output
quandl.ApiConfig.api_key = 'VmiHxTrcihMnqEB-B9sr'
df = quandl.get('WIKI/GOOGL')  # get the data from the website
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]  # from that data, get the heading that
# you would like to use
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0  # Add a new row for percentage change
# between the closed price of stock and the stocks throughout the day
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df[
    'Adj. Open'] * 100.0  # And one row for the initial price of stock and the end price of the stock
# Only trained on Price
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]  # features

forecast_col = 'Adj. Close'  # if confused see https://www.youtube.com/watch?v=lN5jesocJjk comments
df.fillna(-99999, inplace=True)  # if data is not found, treat it as an outlier with -999999
forecast_out = math.ceil(0.01 * len(df))  # This predection would be len(df) = 3354, * by a fraction which would the give the predection in the future e.g. 0.01*3354 = 33.54 ceiled to 34 hence this prediction each column of label is a predction of towards the future of 34 days
#print(forecast_out) # change this to change the graph

df['Label'] = df[forecast_col].shift(-forecast_out)

#print(df.head())

X = np.array(df.drop(['Label'], 1))  # keep all the features in x except label
X = preprocessing.scale(X)
X_Lately = X[-forecast_out:] # all days after day 35
X = X[:-forecast_out] # Get the first 35 days

df.dropna(inplace= True)
y = np.array(df['Label'])
#print(len(X), len(y))
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
#Training part, comment out once in a while
#clf = LinearRegression(n_jobs= -1)
#clf.fit(X_train, y_train)
#with open('lineearregression.pickle','wb') as f:
#    pickle.dump(clf,f)
#If Trained use the following two lines
# You can scale this very well, save it as a pickle so you dont have to gtrain it again
pickle_in = open('lineearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_Lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    # loc means location # iloc position from back e.g -1 is last
#print(df.tail()) These are the predicted stock prices of Googl for the next 30 days
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()