
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[ ]:


#Importing train and test data from PB_1 and PB_2
pb1_train = pd.read_csv(r'C:\Users\Deepika\Downloads\studies\data mining\homework 5\PB1_train.csv', header = None)
pb1_test = pd.read_csv(r'C:\Users\Deepika\Downloads\studies\data mining\homework 5\PB1_test.csv', header = None)
X_train = pb1_train.iloc[:, 0:3].values
Y_train = pb1_train.iloc[:, 3].values
X_test = pb1_test.iloc[:, 0:3].values
Y_test = pb1_test.iloc[:, 3].values
print(pb1_train)
print(X_train)
print(Y_train)
pb2_train = pd.read_csv(r'C:\Users\Deepika\Downloads\studies\data mining\homework 5\PB2_train.csv', header = None)
pb2_test = pd.read_csv(r'C:\Users\Deepika\Downloads\studies\data mining\homework 5\PB2_test.csv', header = None)

X_train2 = pb2_train.iloc[:, 0:3].values
Y_train2 = pb2_train.iloc[:, 3].values
X_test2 = pb2_test.iloc[:, 0:3].values
Y_test2 = pb2_test.iloc[:, 3].values


# In[ ]:


#Featuring Scaling the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
sc2 = StandardScaler()
X_train2 = sc2.fit_transform(X_train2)
X_test2 = sc2.transform(X_test2)


# In[ ]:


# SVM with linear kernel
SV1 = SVC(kernel = 'linear', random_state = 0)
SV1.fit(X_train, Y_train)
Y_predSV1 = SV1.predict(X_test)
cm1 = confusion_matrix(Y_test, Y_predSV1)
acc=metrics.accuracy_score(Y_test, Y_predSV1)*100


# In[ ]:


print("After Linear:",Y_predSV1)
print("After Linear:",acc)



# In[ ]:


#cm1


# In[ ]:


accSV1 = (cm1[0][0] + cm1[1][1])/(cm1[0][0] + cm1[0][1] + cm1[1][0] + cm1[1][1])
#accSV1


# In[ ]:


SV2 = SVC(kernel = 'poly',degree=5, random_state = 0)
SV2.fit(X_train, Y_train)
Y_predSV2 = SV2.predict(X_test)
cm2 = confusion_matrix(Y_test, Y_predSV2)
acc=metrics.accuracy_score(Y_test, Y_predSV2)*100

# In[ ]:

print("After Poly:",Y_predSV2)
print("After Poly:",acc)


# In[ ]:


#cm2


# In[ ]:


accSV2 = (cm2[0][0] + cm2[1][1])/(cm2[0][0] + cm2[0][1] + cm2[1][0] + cm2[1][1])
#accSV2


# In[ ]:


SV3 = SVC(kernel = 'rbf', random_state = 0)
SV3.fit(X_train, Y_train)
Y_predSV3 = SV3.predict(X_test)
cm3 = confusion_matrix(Y_test, Y_predSV3)
acc=metrics.accuracy_score(Y_test, Y_predSV3)*100

# In[ ]:


print("After rbf:",Y_predSV3)
print("After rbf:",acc)



# In[ ]:


#Y_test
#
#
## In[ ]:
#
#
#cm3


# In[ ]:


accSV3 = (cm3[0][0] + cm3[1][1])/(cm3[0][0] + cm3[0][1] + cm3[1][0] + cm3[1][1])
#accSV3


# In[ ]:


SV21 = SVC(kernel = 'linear', random_state = 0)
SV21.fit(X_train2, Y_train2)
Y_pred2SV21 = SV21.predict(X_test2)
cm21 = confusion_matrix(Y_test2, Y_pred2SV21)
acc=metrics.accuracy_score(Y_test2, Y_pred2SV21)*100


# In[ ]:


print("After Linear:",Y_pred2SV21)
print("After Linear:",acc)

# In[ ]:

#
#Y_pred2SV21


# In[ ]:


#cm21


# In[ ]:


accSV21 = (cm21[0][0] + cm21[1][1])/(cm21[0][0] + cm21[0][1] + cm21[1][0] + cm21[1][1])
#accSV21


# In[ ]:


SV22 = SVC(kernel = 'poly',degree=7, random_state = 0)
SV22.fit(X_train2, Y_train2)
Y_pred2SV22 = SV22.predict(X_test2)
cm22 = confusion_matrix(Y_test2, Y_pred2SV22)
acc=metrics.accuracy_score(Y_test2, Y_pred2SV22)*100


# In[ ]:


print("After Poly:",Y_pred2SV22)
print("After Poly:",acc)

# In[ ]:


#Y_pred2SV22
#
#
## In[ ]:
#
#
#cm22


# In[ ]:


accSV22 = (cm22[0][0] + cm22[1][1])/(cm22[0][0] + cm22[0][1] + cm22[1][0] + cm22[1][1])
#accSV22


# In[ ]:


#SV23 = SVC(kernel = 'rbf', random_state = 0)
SV3.fit(X_train2, Y_train2)
Y_pred2SV23 = SV3.predict(X_test2)
cm23 = confusion_matrix(Y_test2, Y_pred2SV23)
acc=metrics.accuracy_score(Y_test2, Y_pred2SV23)*100


# In[ ]:


print("After rbf:",Y_pred2SV23)
print("After rbf:",acc)


# In[ ]:


#Y_pred2SV23
#
#
## In[ ]:
#
#
#cm23


# In[ ]:


accSV23= (cm23[0][0] + cm23[1][1])/(cm23[0][0] + cm23[0][1] + cm23[1][0] + cm23[1][1])
#accSV23

