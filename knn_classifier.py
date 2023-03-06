#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[21]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


# In[22]:


###dataset에 열의 이름을 부여해준다.
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']


# In[23]:


### pandas 데이터프레임으로 데이터 읽기
dataset = pd.read_csv(url, names = names)

dataset.head()


# In[24]:


###전처리
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# In[25]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)


# In[27]:


###정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[28]:


###훈련 및 예측
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors = 5)
kn.fit(x_train, y_train)

y_pred = kn.predict(x_test)


# In[29]:


###알고리즘 평가
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:




