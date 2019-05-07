
# coding: utf-8

# In[12]:


#%matplotlib tk


# In[1]:


import pandas as pd, numpy as np


# In[2]:


import seaborn as sns, matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('KNN_Project_Data')


# In[4]:


df.head()


# In[23]:


sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


scaler = StandardScaler()


# In[8]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[9]:


scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[10]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[11]:


from sklearn.model_selection import train_test_split


# In[24]:


X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[25]:


from sklearn.neighbors import KNeighborsClassifier


# In[26]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[27]:


knn.fit(X_train,y_train)


# In[28]:


pred = knn.predict(X_test)


# In[29]:


from sklearn.metrics import classification_report,confusion_matrix


# In[30]:


print(confusion_matrix(y_test,pred))


# In[31]:


print(classification_report(y_test,pred))


# In[32]:


error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[33]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',
        markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[39]:


error_rate = []

for i in range(1,60):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[40]:


plt.figure(figsize=(10,6))
plt.plot(range(1,60),error_rate,color='blue',linestyle='dashed',marker='o',
        markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[41]:


knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

