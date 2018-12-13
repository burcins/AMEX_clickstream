# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 21:47:20 2018

@author: burcin
"""
import os
os.chdir("H:\American Express ML Hackathron")
import pandas as pd
import dateutil
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold

train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)
hist = pd.read_csv("historical_user_logs.csv", index_col=0)

test.user_depth = test.user_depth.fillna(method='ffill')
test.age_level = test.age_level.fillna(method='ffill')
test.user_group_id = test.user_group_id.fillna(method='ffill')
test.gender = test.gender.fillna(method='ffill')

total = train.append(test, sort=False)

total.product_category_2 = total.product_category_2.fillna(method='bfill')
total.city_development_index = total.city_development_index.fillna(method='ffill')
total.gender = total.gender.fillna(method='ffill')
total.user_group_id = total.user_group_id.fillna(method='ffill')
total.age_level = total.age_level.fillna(total.age_level.mean())
total.user_depth = total.user_depth.fillna(total.user_depth.mean())


total.info()
total.isnull().sum()

train.is_click.mean()
train.groupby('is_click').mean()
train.groupby('product')['is_click'].mean()
train.groupby('age_level')['is_click'].mean()
train.groupby('DateTime')['is_click'].mean().max()

total.apply(lambda x: [x.unique()])

total = total[total.gender.isnull()==False]
total=total.drop('product_category_2', 1)
total = total.drop('city_development_index',1)
total = total.drop('campaign_id',1)
total = total.drop('user_id',1)

total['DateTime'] = total['DateTime'].apply(dateutil.parser.parse)

total['date'] = [d.date() for d in total['DateTime']]
total['time'] = [d.time() for d in total['DateTime']]
total = total.drop('DateTime',1)

total['product'] = total['product'].astype('category')
total['age_level'] = total['age_level'].astype(str)
total['gender'] = total['gender'].astype('category')
total['product_category_1'] = total['product_category_1'].astype(str)
total['var_1'] = total['var_1'].astype(str)


encoder = preprocessing.LabelEncoder()
columns= ['product','product_category_1','gender','age_level','user_depth','var_1','date','time']
columns= ['product','product_category_1','product_category_2','gender','age_level','user_depth','var_1','city_development_index','date','time','webpage_id','user_group_id']

labencoded = total.copy()
for col in columns:
    labencoded[col]=  encoder.fit_transform(total[col]).reshape(-1,1)


test1 = labencoded[labencoded.is_click.isnull()==True]
train1 = labencoded[labencoded.is_click.isnull()==False]
subses = test1.index
Y= train1.is_click
train2 = train1
train2 = train2.drop('is_click',1)
test2 = test1.drop('is_click',1)

X_train, X_test, Y_train, Y_test = train_test_split(train2, Y, test_size=0.2)

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
confusion_matrix(Y_test, Y_pred)

# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
confusion_matrix(Y_test, Y_pred)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
confusion_matrix(Y_test, Y_pred)

Y_forsub = knn.predict(test2)
submission = pd.DataFrame({
        "session_id": subses,
        "is_click": Y_forsub
    })
submission.to_csv('subknn.csv', index=False) 
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
confusion_matrix(Y_test, Y_pred)

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
confusion_matrix(Y_test, Y_pred)

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
confusion_matrix(Y_test, Y_pred)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
confusion_matrix(Y_test, Y_pred)


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
confusion_matrix(Y_test, Y_pred)

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
confusion_matrix(Y_test, Y_pred)


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)





pd.merge(left=train, right=hist, left_on='user_id', right_on='user_id')








