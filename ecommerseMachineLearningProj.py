import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv('Ecommerce Customers')

# print(customers.head())
# print(customers.describe())

#Time on website vs Yearly Amount Spent
# sns.jointplot(data=customers,x='Time on Website', y='Yearly Amount Spent')
# plt.show()

#Time on App vs Yearly Amount Spent
# sns.jointplot(data=customers,x='Time on App', y='Yearly Amount Spent')
# plt.show()

#Time on app vs Length of membership
# sns.jointplot(x='Time on App', y='Length of Membership', kind='hex', data=customers)
# plt.show()

#Pairplot
# sns.pairplot(customers)
# plt.show()

#Longer membership shows more money spent
# sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)
# plt.show()


#Create Training and Testing Data
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

#Train Linear Regression model from the data
lm.fit(X_train,y_train)

# print(lm.coef_)

predictions = lm.predict(X_test)

# Shows how model is doing
# plt.scatter(y_test,predictions)
# plt.xlabel('Y Test (True Values)')
# plt.ylabel('Predicted Values')
# plt.show()



from sklearn import metrics
print('MAE ', metrics.mean_absolute_error(y_test, predictions))
print('MSE ', metrics.mean_squared_error(y_test, predictions))
print('RMSE ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


#How much variance?
print(metrics.explained_variance_score(y_test, predictions))



# sns.distplot((y_test-predictions),bins=50)
# plt.show()

cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])
print(cdf)

'''

This result shows that the MOST important thing that you can do for this business to keep members for longer.
The second most important thing is to work on the App as it is generating a lot more sales than the website.
This data shows that the website does not create many sales.

'''