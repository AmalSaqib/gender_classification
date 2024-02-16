#!/usr/bin/env python
# coding: utf-8

# In[15]:


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split


# In[16]:


# [height, weight, shoe_size]


X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)


# In[20]:


classifiers = [tree.DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier()]


# In[23]:


scores = []
prediction = []

for classifier in classifiers:
    clf = classifier.fit(X_train, Y_train)
    scores.append(clf.score(X_test, Y_test))
    prediction.append(clf.predict([[190, 70, 43]]))


# In[24]:


scores


# In[25]:


prediction

