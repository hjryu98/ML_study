#!/usr/bin/env python
# coding: utf-8

# In[14]:


fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9] ###fish의 정보


# In[15]:


fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)] ###정보들을 2차원 리스트로 변환
fish_target = [1] * 35 + [0] * 14 ###결과값


# In[16]:


from sklearn.neighbors import KNeighborsClassifier ###knn 사용을 위한 import
kn = KNeighborsClassifier() ###knn


# In[17]:


print(fish_data[4]) ###배열의 5번째 값 출력


# In[18]:


print(fish_data[0:4]) ###슬라이싱으로 출력 가능


# In[19]:


train_input = fish_data[:35]
train_target = fish_target[:35]
test_input = fish_data[35:]
test_target = fish_target[35:]


# In[20]:


kn.fit(train_input, train_target) ###knn 훈련 시키기


# In[21]:


kn.score(test_input, test_target) ###test data로 knn 모델 평가


# In[27]:


import numpy as np ###넘파이 사용

input_arr = np.array(fish_data)
target_arr = np.array(fish_target) ###파이썬 리스트를 넘파이 배열로 변환 >> np.array(리스트)


# In[28]:


print(input_arr)


# In[29]:


print(input_arr.shape) ###shape는 샘플 수, 특성 수를 튜플형으로 출력한다.


# In[30]:


np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)


# In[31]:


train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]


# In[32]:


import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


# In[33]:


kn = kn.fit(train_input, train_target)


# In[34]:


kn.score(test_input, test_target)


# In[35]:


kn.predict(test_input)


# In[36]:


test_target


# In[ ]:




