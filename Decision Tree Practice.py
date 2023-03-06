#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Decision Tree
#지도 학습 기법, 분류와 회귀가 모두 가능하다.
#알고리즘 : 엔트로피, 불순도

#엔트로피(Entropy) : 불순도를 수치적으로 나타낸 척도(1 : 최대, 0: 최소)
#불순도(Impurity) : 해당 범주 안에 서로 다른 데이터가 얼마나 섞여있는가?
#정보 획득(Information gain) : 분기 이전의 엔트로피 - 분기 이후의 엔트로피 값
#결정 트리 알고리즘은 정보 획득을 최대화 하는 방향으로 학습이 진행된다.


# In[1]:


###implementation

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# In[6]:


cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state = 42)
tree = DecisionTreeClassifier(random_state = 10) ###한 종류의 데이터가 남을때까지 가지치기
tree.fit(x_train, y_train) ##훈련시키고 정확도 평가

print("훈련 세트 정확도: {:.3f}".format(tree.score(x_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(x_test, y_test)))


# In[7]:


tree = DecisionTreeClassifier(max_depth=4, random_state=10) ##최대 높이를 설정해서 과적합 방지, 훈련세트 정확도는 감소하나 테스트 세트는 증가
tree.fit(x_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(tree.score(x_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(x_test, y_test)))


# In[ ]:




