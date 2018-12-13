# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 21:47:20 2018

@author: burcin
"""
import os
os.chdir("H:\American Express ML Hackathron")
import pandas as pd
import dateutil
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)
hist = pd.read_csv("historical_user_logs.csv", index_col=0)

test.user_depth = test.user_depth.fillna(method='ffill')
test.age_level = test.age_level.fillna(method='ffill')
test.user_group_id = test.user_group_id.fillna(method='ffill')
test.gender = test.gender.fillna(method='ffill')

total = train.append(test, sort=False)


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

encoder = preprocessing.LabelEncoder()
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



# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
confusion_matrix(Y_test, Y_pred)

Y_forsub = perceptron.predict(test2)
submission = pd.DataFrame({
        "session_id": subses,
        "is_click": Y_forsub
    })
submission.to_csv('subperc.csv', index=False) 


