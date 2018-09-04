# -*- coding: utf-8 -*-


import numpy as np

def oracle_sample(batch_x,batch_y,n,task):
    if task == "Classification":
        a,b = oracle_Classification(batch_x,n)
    elif task == "Recursion":
        a,b = oracle_Recursion(batch_x,batch_y,n)
    else:
        print("task-error!")
    return a,b

def oracle_Classification(batch_x,n):
    d = len(batch_x[1])
    a_w = [[0 for j in range(d)]for i in range(n)] 
    b_w = [0]*(n)
    
    for i in range(n):
        
        
        
        
        
        c= [[None for j in range(d)]for i in range(2)]
        t = np.random.randint(d)
        s=t
        while(s==t):
            s= np.random.randint(d)
        
        c[0] = batch_x[t]
        c[1]= batch_x[s]
        
        zeta = np.random.beta(100,3)
        gamma = np.random.binomial(1,0.5,1)
        z = zeta*((-1)**gamma)
        abs_a = 1/np.linalg.norm(c[1]-c[0]) 
        a_w[i] = abs_a /np.linalg.norm(c[0])*c[0] 
        b_w[i] = np.inner(a_w[i],c[0]) - z
    
    return a_w,b_w
        
        
    
