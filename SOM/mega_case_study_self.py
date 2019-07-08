# -*- coding: utf-8 -*-

# Mega Case Study - Make a Hybrid Deep Learning Model



# Part 1 - Identify the Frauds with the Self-Organizing Map

# Importing the libraries

# Library Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Import the dataset
import os
cwd = os.getcwd()
file_name = "Credit_Card_Applications.csv"
data = pd.read_csv(cwd+"/"+file_name)

# print(data.head(10))
# print(data.describe())

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

# Traning SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15)
som.random_weights_init(X)
som.train_random(data =X , num_iteration=100)

#Finding Distance from Best Maching Units to find the Outliers(fraud Customers)

cnt=0
threshold_distance = 0.9
list_som_distance = som.distance_map()
outlier_cordinator = []
outlier_distance = []
for i in range(0,len(list_som_distance)):
    for j in range(0,len(list_som_distance[i])):
        if (list_som_distance[i][j]> threshold_distance) :
            cnt+=1
            outlier_cordinator.append((i,j))
            outlier_distance.append(list_som_distance[i][j])

print("\nCount of outliner", cnt)
print("\nOutlier distance", outlier_distance)
print("\nOutlier Cordinates", outlier_cordinator)

# Visualising Result
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()

marker = ['o', 's']
color = ['r', 'g']
dict_map=dict()
for i, x in enumerate(X):
    w = som.winner(x)
    # if w in outlier_cordinator :
    plot(w[0] + 0.5,
         w[1] +0.5,
         marker[y[i]],
         markeredgecolor = color[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2
         )
# show()

# Finding Fraud Customers
fraud_customer=[]
mapping = som.win_map(X)
cnt=0
for key, value in mapping.items():
    if key in outlier_cordinator:
        cnt+=1
        # First fraud customer attribute
        if len(fraud_customer)==0:
            fraud_customer=value
        else:
            fraud_customer=np.concatenate((fraud_customer,value), axis=0)

print("******"*100)
# print(fraud_customer)
print("cnt", cnt)

fraud_customer = sc.inverse_transform(fraud_customer)
print("\n There are {} Fraud Customers, Customer Id for the Fraud Customers \n{}\n".format(len(fraud_customer[:,0]),
                                                                                            fraud_customer[:,0]))
print("*********END**************")

class_val = {0:'Rejected',
             1:'Approved'}
print("{:>10}{:>10}".format("CustomerID","Status"))
print("\n")
for i in fraud_customer[:,0]:
    print("{:>10}{:>10}".format(i, class_val[data.loc[data['CustomerID']==i,'Class'].iloc[0]]))


# PART 2 : Creating Supervised Model from Unsupervised input

# 1. Independent Variables

customers = data.iloc[:, 1:].values

# 2. Dependent Variable

is_fraud = np.zeros(len(data))

for i in range(len(data)):
    if data.iloc[i,0] in fraud_customer[:,0]:
        is_fraud[i]= 1
#print(is_fraud)


# ANN Part


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
classifier.add(Dropout(0.2))

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 3)

# Predicting the probabilities of frauds
y_pred = classifier.predict(customers)
customer_prob = np.concatenate((data.iloc[:, 0:1].values, y_pred ), axis =1)
customer_prob = customer_prob[customer_prob[:, 1].argsort()[::-1]]


