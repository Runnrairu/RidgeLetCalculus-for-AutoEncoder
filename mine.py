# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 03:54:38 2018

@author: 宮本来夏
"""

import numpy as np
import tensorflow as tf


def mnist_double(batch_y,task):#mnistを二値分類に帰着
    n = len(batch_y)
    
    y_new = [[0 for j in range(1)] for i in range(n)]
    if task == "OrS":
        for i in range(n):
            if batch_y[i][0] == 1:
                y_new[i][0]= 100000.0
            else:
                y_new[i][0]=-100000.0
                
                
    elif task =="label":
        
        for i in range(n):
            if batch_y[i][0] == 1:
                y_new[i][0] = 1.0
            else:
                y_new[i][0]= 0.0
        
        
    else:
        print ("error")
    return y_new



def double_class(y,n):
    
    y_new = [0]*(n)
    for i in range(n):
        y_new[i] = tf.cond(y[i][0]>0.5,lambda:1.0,lambda:0.0)
            
    return y_new
            
    


def oracle_sample(batch_x,batch_y,n,task):
    
    if task == "Classification":
        a,b = oracle_Classification(batch_x,batch_y,n)
    elif task == "Recursion":
        a,b = oracle_Recursion(batch_x,batch_y,n)
    else:
        print("task-error!")
    return a,b

def oracle_Classification(batch_x,batch_y,n):
    d = len(batch_x[1])
    m=len(batch_x)
    
    a_w = [[0.0 for j in range(d)]for i in range(n)] 
    b_w = [[0.0 for j in range(1)]for i in range(n)] 
    a_w = np.array(a_w)
    b_w = np.array(b_w)
    for i in range(n):

        
        c= [[None for j in range(d)]for i in range(2)]
        t = np.random.randint(m)
        s=t
        while(s==t):
            s= np.random.randint(m)
        
        
        c[0] = batch_x[t]
        c[1]= batch_x[s]
        
        zeta = np.random.beta(100,3)
        gamma = np.random.binomial(1,0.5,1)
        z = zeta*((-1)**gamma)
        abs_a = 1.0/np.linalg.norm(c[1]-c[0])
        
        a_w[i] = (abs_a /np.linalg.norm(c[0]))*c[0] 
        b_w[i] = np.inner(a_w[i],c[0]) - z
    
    return a_w,b_w
        


def oracle_Recursion(batch_x,batch_y,n):#一次元出力の回帰のオラクルサンプリング
    sum_y = np.sum(batch_y,dtype="float")
    d = len(batch_x[1])
    nor = 1.0/sum_y
    uni_y = nor*np.array(batch_y)
    print(d)
    
    
    N = len(batch_y)
    print(N)
    a_w = [[0.0 for j in range(d)]for i in range(n)] 
    b_w = [[0.0 for j in range(1)]for i in range(n)] 
    a_w = np.array(a_w)
    b_w = np.array(b_w)
    for j in range(n):
        u = np.random.uniform(0.0, 1.0)
        now = 0
        i = 0
        
        t=0
        while(now<u):
            t +=1
            print(t)
            now += uni_y[i]
            i += 1
        t = i
        s = -1
        while(t !=s):
            u = np.random.uniform(0.0, 1.0)
            now = 0
            i = 0
            while(now<u):
                now += uni_y[i]
                i += 1
            s = i
        c= [[None for j in range(d)]for i in range(2)]
        c[0] = batch_x[t]
        c[1]= batch_x[s]
        
        zeta = np.random.beta(100,3)
        gamma = np.random.binomial(1,0.5,1)
        z = zeta*((-1)**gamma)
        abs_a = 1.0/np.linalg.norm(c[1]-c[0])
        
        a_w[i] = (abs_a /np.linalg.norm(c[0]))*c[0] 
        b_w[i] = np.inner(a_w[i],c[0]) - z
    
    return a_w,b_w







