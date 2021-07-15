#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[3]:


X_train = pd.read_csv('C:/Users/dines/OneDrive/Documents/Training Data/Linear_X_Train.csv')
Y_train = pd.read_csv('C:/Users/dines/OneDrive/Documents/Training Data/Linear_Y_Train.csv')


# In[4]:


X_train.head(5)


# In[5]:


Y_train


# In[6]:


X_train.describe()


# #### Normalise the dataset
# 

# In[7]:


u = X_train.mean()
std = X_train.std()
X_train = (X_train - u)/std
X_train.describe()


# In[8]:


X = X_train.values
Y = Y_train.values


# Visualising the data

# In[9]:


plt.style.use('seaborn')
plt.scatter(X,Y,color = 'red')
plt.xlabel("Hours")
plt.ylabel('Marks gain')
plt.title("Marks on the basis of Hours of Hardwork")
plt.show()


# In[10]:


def hypothesis(x,theta):
    y_= theta[0]+theta[1]*x
    return y_
def gradient(X,Y,theta):
    grad = np.zeros((2,))
    m = X.shape[0]
    for i in range(m):
        x= X[i]
        y_ = hypothesis(x,theta)
        grad[0] += (y_-Y[i])
        grad[1] += (y_-Y[i])*x
    return grad/m
        
def error(X,Y,theta):
    m = X.shape[0]
    total_error = 0.0
    for i in range(m):
        y_ = hypothesis(X[i],theta)
        total_error += (y_ - Y[i])**2
    return total_error/m
        
def gradientDescent(X,Y,max_steps = 100,learning_rate = 0.1):
   
    theta = np.zeros((2,))
    error_list = []
    theta_list = []
    for i in range(max_steps):
        grad = gradient(X,Y,theta)
        e = error(X,Y,theta)
        theta[0] = theta[0] - learning_rate* grad[0]
        theta[1] = theta[1] - learning_rate*grad[1]
        error_list.append(e)
        theta_list.append((theta[0],theta[1]))
    return theta , error_list , theta_list

    


# In[11]:


theta,error_list,theta_list = gradientDescent(X,Y)
theta_list


# In[12]:


plt.plot(error_list)
plt.show()


# In[13]:


def prediction(X,theta):
    y_ = hypothesis(X,theta)
    return y_


# In[14]:


y_ = prediction(X,theta)


# In[15]:


plt.scatter(X,Y,c='red')
plt.plot(X,y_,label = "Predictions")
plt.xlabel('Number of Hours Studied')
plt.ylabel('Marks Obtained')
plt.legend()
plt.show()


# In[16]:


X_test = pd.read_csv('C:/Users/dines/OneDrive/Documents/Test Cases/Linear_X_Test.csv')


# In[17]:


X_test.values


# In[18]:


y_test = prediction(X_test,theta)
y_test.rename(columns={'x':'y'},inplace = True)


# In[19]:


y_test.to_csv('y_prediction.csv',index = False)


# In[20]:


def r2score(Y,Y_):
    num = np.sum((Y-Y_)**2)
    deno = np.sum((Y-Y.mean())**2)
    score = (1-num/deno)
    return score*100


# In[21]:


r2score(Y,y_)


# ## Visualizing the Loss function, Gradient Descent , Theta Updates.

# In[22]:


T0 = np.arange(-40,40,1)
T1 = np.arange(40,120,1)
T0,T1 = np.meshgrid(T0,T1)
m = X.shape[0]
J = np.zeros(T0.shape)
for i in range(J.shape[0]):
    for j in range(J.shape[1]):
        y_ = T1[i,j]*X+T0[i,j]
        J[i,j] = np.sum((Y-y_)**2)/m



# In[23]:


fig = plt.figure()
axes = fig.gca(projection = '3d')
axes.plot_surface(T0,T1,J,cmap = 'rainbow')
plt.show()


# In[24]:


theta_list = np.array(theta_list)
theta_list


# In[25]:


plt.plot(theta_list[:,0],label = 'Theta0')
plt.plot(theta_list[:,1],label = 'Theta1')
plt.legend()
plt.show()


# ### Trajectory traced by theta updates

# In[26]:


fig = plt.figure()
axes = fig.gca(projection = '3d')
axes.plot_surface(T0,T1,J,cmap = 'rainbow')
axes.scatter(theta_list[:,0],theta_list[:,1],error_list,color = 'red')
plt.show()


# In[27]:


fig = plt.figure()
axes = fig.gca(projection = '3d')
axes.contour(T0,T1,J,cmap = 'rainbow')
axes.scatter(theta_list[:,0],theta_list[:,1],error_list,color = 'red')
plt.show()


# ### 2-D contour

# In[28]:


plt.contour(T0,T1,J,cmap = 'rainbow')
plt.scatter(theta_list[:,0],theta_list[:,1],color = 'red')
plt.show()


# In[29]:


# save the data in file.
np.save('Thetafile.npy',theta_list)


# In[30]:


theta = np.load('Thetafile.npy')


# In[31]:


T0=theta[:,0]
T1 = theta[:,1]


# In[1]:


plt.ion
for i in range(0,50,3):
    y_ = T1[i]*X + T0
    #points
    plt.scatter(X,Y)
    plt.plot(X,y_,'red')
    plt.draw()
    plt.pause(1)
    plt.clf()
    
    

