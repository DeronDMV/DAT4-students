# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 20:35:35 2015

@author: deronhogans
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
# Loading the packages I'll need to build my models 

trips = pd.DataFrame.from_csv('2PointB_January_Trips.csv', sep=",", header=0, index_col=0)
# Reading in the trips data from the month of January 

trips.describe()
# There are only two features with values, Gross_Amount and Discount_Applied \
# I want to see what features have strong relationships with Discount_Applied, so I'll transform strings to floats

trips['Status'] = trips.Status.map({'Meter Off': 1, 'Unable to Auth': 0, 'Cancelled':-1})
# Transforming Status feature, Meter Off means read the ride was completed. So I'll value that at 1
# Unable to Auth means the ride could not be ordered, so I'll value that at 0 
# A Cancelled trip is no good, so I'll value that at -1
 
trips.Trip_Adjusted = trips.Trip_Adjusted.map({'Yes': 1, 'No': 0})
# A Trip Adjustment typically means a trip will be completed, let's transform these values then plot to be sure

trips.plot(kind='scatter', x='Trip_Adjusted', y='Status', alpha=0.5)
# Looks like there was no significant relationship between Status and Trip_Adjusted so I'll continue with transformations

trips.Existing_Passenger = trips.Existing_Passenger.map({'Yes': 1, 'No': 0})
# If a passenger is new I'll value at 0, if existing I'll value at 1

trips.Fleet = trips.Fleet.map({'Discount Cab': 1, 'Cab': 2, 'Rideshares': 3, 'Black': 4, 'SUV': 5})
# the Fleet feature describes the type of car used for a trip. 2pointb values Black Car and SUV rides highers so
# i'll value them them higher at 4 and 5 respectively.

trips.describe()
# Let's see waht the data set looks like now and what we can learn
# Rides were taken in discount cabs on average
# Avg trip cost was 12.52
# Avg discount applied was $5

FullTrips = trips.dropna()
# There were a good number of missing values so I'll drop those and see if I can picks up any useful predictors of discount value from plotting

FullTrips.plot(kind='scatter', x='Existing_Passenger', y='Discount_Applied', alpha=0.5)
# There seems to be some high correlation between Existing Passengers and Discount Applied 
# Just about all new passengers are using discounts to take care of their trips

FullTrips.plot(kind='scatter', x='Gross_Amount', y='Discount_Applied', alpha=0.5)
# Gross Amount or Trip Cost may be the most valuable prediction feature
# Most discounts were used to completely cover trips, the higher the cost of the trip the less likely it seems discounts were applied

FullTrips.plot(kind='scatter', x='Fleet', y='Discount_Applied', alpha=0.5)
# Most discounts were applied to rides in Discount Cabs or Black Cars

FullTrips.plot(kind='scatter', x='Status', y='Discount_Applied', alpha=0.5)
# A discount can only be completed if the trip is completed, right? I'll leave this out of the features used for modeling

FullTripsSelect = pd.DataFrame({'Fleet': FullTrips['Fleet'], 'Gross_Amount': FullTrips['Gross_Amount'], 'Existing_Passenger': FullTrips['Existing_Passenger']})
# I've selected Fleet, Gross Amount and Existing Passenger as features to build my model

X, y = FullTripsSelect, FullTrips.Discount_Applied # Setting X equal to my prediction features and y equal to Discount_Applied
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4) # train/test/split the data to build prediction model
X_train.shape
X_test.shape
y_train.shape
y_test.shape

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
# Starting out with K=1 
# Only 70%, let's up KNN and see if the score increases

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
# KNN score increased to 77%

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
# KNN score remeanins 77%, let's see if we can get a more accurate idea of what a better KNN would be with Cross Validation

knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
scores
np.mean(scores)
# K=5 receives a cross val accuracy score of 76%, let's try 9 

knn = KNeighborsClassifier(n_neighbors=9)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
scores
np.mean(scores)
# K=9 is slightly higher, let's plot these scores to get a visual picture

k_range = range(1, 30, 2)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores.append(np.mean(cross_val_score(knn, X, y, cv=10, scoring='accuracy')))
scores

plt.figure()
plt.plot(k_range, scores)
# Our highest score of accuracy sits at K=9, so we'll test the model out with out of sample data setting K=9

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(FullTripsSelect, FullTrips.Discount_Applied)
out_of_sample = [0, 5, 10]
knn.predict(out_of_sample)


FullTrips['discount_prediction'] = knn.predict(FullTripsSelect)
# let's add our predictions to our data and see if the predictions are accurate


FullTrips.plot(kind='scatter', x='Discount_Applied', y='discount_prediction', alpha=0.5)
# Looking at the plot we did fairly well with a few misses here and there, let's see what our accuracy tells us about the accuracy

from sklearn import metrics
metrics.accuracy_score(FullTrips.Discount_Applied, FullTrips.discount_prediction)
# Achieved 75% Accuracy 

########################################################################################

new_trips = pd.DataFrame.from_csv('2pointb_Jan_Feb_Trips.csv', sep=",", header=0, index_col=0)
# New adding a new month's worht of data, making the set more robust

new_trips['Status'] = new_trips.Status.map({'Meter Off': 1, 'Unable to Auth': 0, 'Cancelled':-1})

new_trips.TripAdjusted = new_trips.TripAdjusted.map({'Yes': 1, 'No': 0})

new_trips.ExistingPassenger = new_trips.ExistingPassenger.map({'Yes': 1, 'No': 0})

new_trips.Fleet = new_trips.Fleet.map({'Discount Cab': 1, 'Cab': 2, 'Rideshare': 3, 'Black': 4, 'SUV': 5})
# Applying the same transformations

NewFullTrips = new_trips.dropna()
# Dropping missing values

NewFullTrips.plot(kind='scatter', x='ExistingPassenger', y='DiscountApplied', alpha=0.5)

NewFullTrips.plot(kind='scatter', x='GrossAmount', y='DiscountApplied', alpha=0.5)

NewFullTrips.plot(kind='scatter', x='Fleet', y='DiscountApplied', alpha=0.5)

NewFullTrips.plot(kind='scatter', x='Status', y='DiscountApplied', alpha=0.5)
# Our relationships look alot stronger now that we have more data

NewFullTripsSelect = pd.DataFrame({'Fleet': NewFullTrips['Fleet'], 'GrossAmount': NewFullTrips['GrossAmount'], 'ExistingPassenger': NewFullTrips['ExistingPassenger']})
# Selecting features for modeling

X, y = NewFullTripsSelect, NewFullTrips.DiscountApplied # Setting X, y for train/test/split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4) # train/test/split
X_train.shape
X_test.shape
y_train.shape
y_test.shape
# our split data sets look good

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
# 77% score for K=9m let's check the accuracy with Cross Validation

knn = KNeighborsClassifier(n_neighbors=9)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
scores
np.mean(scores)
# 77% here, let's plot the k range for good measure and visual affirmation

k_range = range(1, 30, 2)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores.append(np.mean(cross_val_score(knn, X, y, cv=10, scoring='accuracy')))
scores

plt.figure()
plt.plot(k_range, scores)
# looks like 9 will still be our optimal K 

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(NewFullTripsSelect, NewFullTrips.DiscountApplied)
out_of_sample = [0, 5, 10]
knn.predict(out_of_sample)
# Let's try out the prediction model on OOS data
# We got a different prediction this go around. The new data could be the cause.

NewFullTrips['discount_prediction'] = knn.predict(NewFullTripsSelect)
#let's add these predictions to the data set

NewFullTrips.plot(kind='scatter', x='DiscountApplied', y='discount_prediction', alpha=0.5)
# Let's plot against the real discounts
# It looks like our last plot of predicted discounts versus the real thing. On point with a few misses here or there
# let's check the accuracy score

metrics.accuracy_score(NewFullTrips.DiscountApplied, NewFullTrips.discount_prediction)
# 79% this time. 

# More data, better accuracy. 

L_SVC = LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None)
L_SVC.fit(X, y)
L_SVC.decision_function(X)
L_SVC.fit_transform(X, y)
L_SVC.predict(X)
L_SVC.transform(X)