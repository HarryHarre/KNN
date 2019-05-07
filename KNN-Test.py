
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('Classified Data',index_col=0)


# In[4]:


df.head()


# In[5]:


from sklearn.preprocessing import StandardScaler


# In[6]:


scaler = StandardScaler()


# In[7]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[8]:


scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[9]:


scaled_features


# In[10]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])


# In[11]:


df_feat.head()


# In[12]:


from sklearn.model_selection import train_test_split


# In[14]:


X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[15]:


from sklearn.neighbors import KNeighborsClassifier


# In[18]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[19]:


knn.fit(X_train,y_train)


# In[21]:


pred = knn.predict(X_test)


# In[24]:


from sklearn.metrics import classification_report,confusion_matrix


# In[26]:


print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[27]:


error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[31]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',
        markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[32]:


knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

