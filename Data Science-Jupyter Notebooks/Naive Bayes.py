#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB # Gaussians for normal distributions


# In[2]:


dataset= datasets.load_iris() 
model = GaussianNB()

model.fit(dataset.data, dataset.target)
expected = dataset.target


predicted = model.predict(dataset.data)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[ ]:





# In[ ]:





# print(metrics.confusion_matrix(expected, predicted))

# In[ ]:




