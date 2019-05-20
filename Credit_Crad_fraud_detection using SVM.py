#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import *
from numpy import *
from sklearn.utils import shuffle

inp=read_csv("E:\ML\ML CLASSROOM\ML-ASSIGNMENT-2\creditcard.csv") 
inp.loc[inp['Class'] ==0, 'Class'] = -1 


# In[2]:


acc=0
for z in range(5):   
    A1=inp.loc[inp['Class'] == 1] 
    a1=A1.sample(n = 100) 
    #print(a1.shape)
    A0=inp.loc[inp['Class'] == -1] 
    a0=A0.sample(n = 100) 
    A=concat([a1, a0]) 
    data=shuffle(A) 
    
    Y=data['Class'] 
    trainy=int(round(len(Y)*0.8)) 
    Y_train=Y.iloc[:trainy]
    Y_test=Y.iloc[trainy:]
    #training data
    data.drop(["Class"], axis = 1, inplace = True) 
    trainx=int(round(len(data)*.8))
    x_train=data.iloc[:trainx]
    x_test=data.iloc[trainx:]
    x_test.shape
    
    from cvxopt import *   
    
    A=matrix(Y_train,tc='d')
    g=matrix(eye(160),tc='d')
    G=matrix((-1)*g,tc='d')
    h=matrix(zeros(160),tc='d')
    q=matrix(ones(160),tc='d')
    p=matrix((dot(dot(Y_train,Y_train.T),dot(x_train,x_train.T))),tc='d')
    b=matrix(zeros(1),tc='d')
    
    sol=solvers.qp(p,q,G,h,  kktsolver='ldl')
    
    alpha=matrix(sol['x']) 
    w_=multiply(alpha,A)
    #print(w_.shape)
    w=dot(x_train.T,w_)
    #print(w.shape)
    
    a1.drop(["Class"],axis =1,inplace=True)
    #print(a1.shape)
    y_1=dot(a1,w)
    #print(min(matrix(y_1)))
    a0.drop(["Class"],axis =1,inplace=True)
    y_0=dot(a0,w)
    B=(-.5)*(min(matrix(y_1))+max(matrix(y_0)))
    #print(B)                                
    #print(a.shape)
    a_=((dot(x_test,w) +B )) 
    #print(a_.shape)          
    #l=matrix(a_,tc='d')
    m_=(Y_test)
    m=matrix(m_,tc='d')
    #print(a_.shape)
    c=multiply(a_,m)               
    #print(len(c))
    count=0
    #print(c)
    for i in range(len(c)): 
        if(c[i]<0):
            count=count+1
    #print(count)
    #print(len(Y_test))
    accuracy=(1-(count/len(m_)))*100
    print("accuracy : {}".format(accuracy)) 
    acc=accuracy+acc
avg_accuracy=(acc/5)
print("avg_accuracy :{}".format(avg_accuracy))
    
    


# In[ ]:




