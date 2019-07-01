#!/usr/bin/env python
# coding: utf-8

# In[49]:


from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



# In[50]:


X,Y=make_circles(n_samples=500,noise=0.02)


# In[51]:


print(X.shape,Y.shape)


# In[52]:


plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[53]:


def non_linearity(X):
    X1=X[:,0]
    X2=X[:,1]
    X3=X1**2+X2**2
    X_new=np.zeros((X.shape[0],3))
    X_new[:,:-1]=X
    X_new[:,-1]=X3
    return X_new


# In[54]:


X_new=non_linearity(X)


# In[55]:


print(X_new.shape)


# In[73]:


def plot3d(X,show=True):
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection='3d')
    X1=X[:,0]
    X2=X[:,1]
    X3=X[:,2]
    
    ax.scatter(X1,X2,X3,zdir='z',s=20,c=Y,depthshade=True)
    if (show==True):
        plt.show()
    
    return ax
    


# In[74]:


plot3d(X_new)


# In[60]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# In[61]:


lr=LogisticRegression()


# In[62]:


acc=cross_val_score(lr,X,Y,cv=5).mean()
print("Accuracy on X(2d) {}".format(acc*100))


# In[63]:


acc=cross_val_score(lr,X_new,Y,cv=5).mean()
print("Accuracy on X(2d) {}".format(acc*100))


# In[64]:


lr.fit(X_new,Y)


# In[68]:


weights=lr.coef_


# In[69]:


bias=lr.intercept_


# In[70]:


xx,yy=np.meshgrid(range(-2,2),range(-2,2))


# In[72]:


z=-(weights[0,0]*xx+weights[0,1]*yy+bias)/weights[0,2]


# In[80]:


ax=plot3d(X_new,False)
ax.plot_surface(xx,yy,z,alpha=0.2)
plt.show()


# In[ ]:




