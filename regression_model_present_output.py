#regression.py
#Python V3.6.5

#This file is for displaying the neccessary graphs in a neat format to display for regression
#Should also output the needed values, already known that p < 1E-16

#Display Graph of Tx over Time, Prices over Time, Tx vs Prices scatter + Regression


# --------------AGENDA--------------
# 1 - Give Justin Overview of why/Introduction to problem statement
# 2 - Review the data sources
# 3 - Show raw data CSV and explain variable names
# 4 - Review Section 1: Data Wrangling
# 5 - Review Section 2: Statistics
# 6 - Review Section 3: Plot Data
# 7 - Explore interpretations and hypotheses



#Imports needed to make things run
import math, scipy, csv
from scipy import stats #The primary regression model library
import pandas as pd #Used for advanced dataframes
from sklearn import linear_model
import matplotlib.pyplot as plt #Used for visual plots
from sklearn.linear_model import LinearRegression #Library used for regression to output T-Value and P-Value where scipy does not
import numpy as np #For more consistent math


#Stolen class from GitHub to expand SKLearn's Usability
class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression, self)\
                .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                    for i in range(sse.shape[0])
                    ])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self


#--------------------------------------------
#Section 1: Data Wrangling
#Grabbing raw data from .CSV files
#Source: etherscan.io
#
#all_charts_url	 = https://etherscan.io/charts 				# all/other sources of data
#number_tx_url	 = https://etherscan.io/chart/tx 			# no. per day
#ETH_price_url	 = https://etherscan.io/chart/etherprice 	# USD
#Parsing data into usuable list formats
#Computing new data sets based on grabbed data
#
#Small function to grab and parse data from the csv files
def grabRow(filename,index):
	#Open specified file
	with open(filename, newline='') as csvfile:
		#Open the csv reader
		file = csv.reader(csvfile, delimiter=' ', quotechar='|')
		#Parse the data and grab only the column we want
		column = [row[0].split(',')[index] for row in file]
		column = [float(entry.replace("\"","")) for entry in column[1:]]
	return column

#Grab ethPrice and txCount from the .csv files
ethPrice = grabRow('export-EtherPrice.csv', 2)

txCount =grabRow('export-TxGrowth.csv', 2)


#Setup days for each entry
timeDays = [i for i in range(len(txCount))]


#Create cumulative transactions instead of transaction growth
cumTx = []
i = 0
for tx in txCount:
	if i == (0):
		cumTx.append(tx)
		i += 1
		continue
	cumTx.append(tx + cumTx[i-1])


#Setup data to be compatible with SKLearn's Regression object
#Set Dictionary headers to be used for dataframe
data = {'Price': ethPrice, 'Transactions': txCount}
#Create Dataframe from dictionary
df = pd.DataFrame(data)

#Assign Variables to the dataframe sections
X = df.drop('Price', axis = 1)
Y = df.drop('Transactions', axis = 1)


#--------------------------------------------
#Section 2:
#Run Regression, form regression line, and grab relevant data
#Use Scipy's Regression model so we can grab slop, intercept and R value.
#Unused:
#	r: R Value
#	p: P Value
#	err: Error
m,b,r,p,err = scipy.stats.linregress(txCount, ethPrice)

regressionLine = [(m * point) + b for point in txCount]

#Use SKLearn's Regression model to grab T-Value and P-Value
lm = LinearRegression()
#Fit the model using the data grabbed from the dataframe established above
lm.fit(X, Y)
#Grab data from regression model
t_value = lm.t[0][0]
p_value = lm.p[0][0]

#Get the average of the transactions and price.
mean_tx = sum(txCount) / len(txCount)

mean_price = sum(ethPrice) / len(ethPrice)


#--------------------------------------------
#Section 3:
#Plot Data and Output Data


#Outputting Raw Values
print("P-Value: " + str(p_value))
print("T-Value: " + str(t_value))
print("Mean Price: " + str(mean_price))
print("Mean Tx: " + str(mean_tx))
print("Slope: " + str(m))
print("Intercept: " + str(b))


#Plotting Data
#Figure 1: Transaction per Day
plt.title("Fig. 1\nTransactions per Day")
plt.xlabel("Time (Days)")
plt.ylabel("Transactions")
plt.plot(timeDays, txCount)
plt.show()

#Figure 2: ETH Closing Price per Day
plt.title("Fig. 2\nETH Closing Price per Day")
plt.xlabel("Time (Days)")
plt.ylabel("ETH Price (USD)")
plt.plot(timeDays, ethPrice)
plt.show()


#Figure 3: Transactions vs ETH Price Regression
plt.title("Fig. 3\nTransactions vs ETH Price Regression")
plt.xlabel("Transactions")
plt.ylabel("ETH Price (USD)")
plt.scatter(txCount, ethPrice)
plt.plot(txCount, regressionLine, color="orange")
plt.show()


