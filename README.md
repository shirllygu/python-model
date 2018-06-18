# Loading libraries
Firstly, we will load all the required libraries for data manipulation, visualization and machine learning.


```python
# Load libraries
import numpy as np                                # For matrix operations and numerical processing
import pandas as pd                               # For munging tabular data
import matplotlib.pyplot as plt 
import os
import xgboost as xgb
import sklearn
import pickle
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
```
# Loading the data
We will load the and explore the data set.
```python

os.chdir("S:\Global_DPA\yugu\cameo\NLD")

data = pd.read_csv('./usinterest.csv')
pd.set_option('display.max_columns', 500)     # Make sure we can see all of the columns
pd.set_option('display.max_rows', 20) 
data.describe()
```
```html
          N_V8688_F      n_ib6540      n_ib6312      n_ib7819      n_ib7795  \
count  7.966937e+06  7.966937e+06  7.966937e+06  7.966937e+06  7.966937e+06   
mean   4.107950e-01  4.922670e-02  1.168849e-01  2.659981e-02  9.062467e-02   
std    4.919782e-01  2.163410e-01  3.212832e-01  1.609107e-01  2.870746e-01   
min    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
25%    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
50%    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
75%    1.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
max    1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00   

           n_ib8653      n_ib8615      n_ib8647      n_ib7809      n_ib7720  \
count  7.966937e+06  7.966937e+06  7.966937e+06  7.966937e+06  7.966937e+06   
mean   4.227376e-01  6.452795e-01  3.840295e-01  8.957483e-02  2.871710e-01   
std    4.939945e-01  4.784286e-01  4.863650e-01  2.855717e-01  4.524421e-01   
min    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
25%    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
50%    0.000000e+00  1.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
75%    1.000000e+00  1.000000e+00  1.000000e+00  0.000000e+00  1.000000e+00   
max    1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00   

           n_ib7739      n_ib7770      n_ib7765      n_ib7820      n_ib7822  \
count  7.966937e+06  7.966937e+06  7.966937e+06  7.966937e+06  7.966937e+06   
mean   3.970572e-01  3.125360e-01  9.843909e-02  7.808660e-02  4.318705e-01   
std    4.892881e-01  4.635270e-01  2.979074e-01  2.683078e-01  4.953367e-01   
min    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
25%    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
50%    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
75%    1.000000e+00  1.000000e+00  0.000000e+00  0.000000e+00  1.000000e+00   
max    1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00   

           n_ib7823      n_ib7768      n_ib8630      n_ib7759      n_ib7788  \
count  7.966937e+06  7.966937e+06  7.966937e+06  7.966937e+06  7.966937e+06   
mean   3.798254e-01  1.664637e-01  5.139077e-01  3.283571e-02  2.650519e-01   
std    4.853433e-01  3.724964e-01  4.998066e-01  1.782064e-01  4.413609e-01   
min    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
25%    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
50%    0.000000e+00  0.000000e+00  1.000000e+00  0.000000e+00  0.000000e+00   
75%    1.000000e+00  0.000000e+00  1.000000e+00  0.000000e+00  1.000000e+00   
max    1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00   

           n_ib8257      n_ib8272      n_ib7732  
count  7.966937e+06  7.966937e+06  7.966937e+06  
mean   6.222793e-02  3.790114e-02  1.399416e-01  
std    2.415691e-01  1.909572e-01  3.469265e-01  
min    0.000000e+00  0.000000e+00  0.000000e+00  
25%    0.000000e+00  0.000000e+00  0.000000e+00  
50%    0.000000e+00  0.000000e+00  0.000000e+00  
75%    0.000000e+00  0.000000e+00  0.000000e+00  
max    1.000000e+00  1.000000e+00  1.000000e+00  
```
# Building simple models
Build a simple machine learning model to predict the class of the flower from the measurements. Before we get started, let's enumerate the steps to carry out

i. Split the datset in to  train data set & test dataset.(80%,20% in this case)

ii. Build models on the train dataset.

iii. Basis one of the chosen metrics as accuracy or AUC (area under the curve),the best performing model will be selected.


Step i : Splitting the data
```python
X, Y = data.iloc[:,1:], data.iloc[:,0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
```


Step ii. Building Models
We will build 5 models using our train data.

## Logistic Regression (LR)
```python
lr = LogisticRegression()
lr.fit(X_train,Y_train)
predictions_lr = lr.predict(X_test)
print("LOGISTIC REGRESSION")
print(accuracy_score(Y_test,predictions_lr))
print(classification_report(Y_test,predictions_lr))
Y_pred_proba_lr = lr.predict_proba(X_test)[::,1]
fpr_lr, tpr_lr, _ =roc_curve(Y_test,  Y_pred_proba_lr)
auc_lr = roc_auc_score(Y_test,Y_pred_proba_lr)

```
## Linear Discriminant Analysis (LDA)
```python
lda = LinearDiscriminantAnalysis()
lda.fit(X_train,Y_train)
predictions_lda = lda.predict(X_test)
print("Linear Discriminant Analysis")
print(accuracy_score(Y_test,predictions_lda))
print(classification_report(Y_test, predictions_lda))

Y_pred_proba_lda = lda.predict_proba(X_test)[::,1]
fpr_lda, tpr_lda, _ =roc_curve(Y_test,  Y_pred_proba_lda)
auc_lda = roc_auc_score(Y_test,Y_pred_proba_lda)
```
## Extreme Gradient Boosting (XGB)
```python

xgb = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)
xgb.fit(X_train,Y_train)
predictions_xgb = xgb.predict(X_test)
print("xgboost")
print(accuracy_score(Y_test,predictions_xgb))
print(classification_report(Y_test,predictions_xgb))
Y_pred_proba_xgb = xgb.predict_proba(X_test)[::,1]
fpr_xgb, tpr_xgb, _ =roc_curve(Y_test,  Y_pred_proba_xgb)
auc_xgb = roc_auc_score(Y_test,Y_pred_proba_xgb)
```

## Classification and Regression Trees (CART)
```python


cart =  DecisionTreeClassifier()
cart.fit(X_train,Y_train)
predictions_cart = cart.predict(X_test)
print("DECISION TREE")
print(accuracy_score(Y_test,predictions_cart))
print(classification_report(Y_test,predictions_cart))
Y_pred_proba_cart = cart.predict_proba(X_test)[::,1]
fpr_cart, tpr_cart, _ =roc_curve(Y_test,  Y_pred_proba_cart)
auc_cart = roc_auc_score(Y_test,Y_pred_proba_cart)

```
## Gaussian Naive Bayes (NB)
```python
nb = GaussianNB()
nb.fit(X_train,Y_train)
predictions_nb = nb.predict(X_test)
print("GaussianNB")
print(accuracy_score(Y_test, predictions_nb))
print(classification_report(Y_test, predictions_nb))
print(confusion_matrix(Y_test,predictions_nb))
Y_pred_proba_nb = nb.predict_proba(X_test)[::,1]
fpr_nb, tpr_nb, _ =roc_curve(Y_test,  Y_pred_proba_nb)
auc_nb = roc_auc_score(Y_test,Y_pred_proba_nb)
```

## Neutral Network:Multi-Layer Perceptron Classifier model(NN)
```python


nn=MLPClassifier()
nn.fit(X_train,Y_train)
predictions_nn = nn.predict(X_test)
print("neural_network")
print(accuracy_score(Y_test, predictions_nn))
print(classification_report(Y_test, predictions_nn))
print(confusion_matrix(Y_test,predictions_nn))
Y_pred_proba_nn = nn.predict_proba(X_test)[::,1]
fpr_nn, tpr_nn, _ =roc_curve(Y_test,  Y_pred_proba_nn)
auc_nn = roc_auc_score(Y_test,Y_pred_proba_nn)

```

## Support Vector Machines (SVM)

## K-Nearest Neighbors (KNN)


Step III. Choosing the best model
We can observe that the best performing model is the LDA baed on the accuracy. We will use this model to against the tet data set. We can visually compare the model performance plotting the auc as well.


![alt text](https://github.com/shirllygu/python-model/blob/master/Screen%20Shot%202018-06-18%20at%204.17.11%20PM.png)

```
