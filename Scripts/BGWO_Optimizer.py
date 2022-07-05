import numpy as np
import random
import math

def sigmoid(x):
    sig = 1 / (1 + math.exp(-10*(x-0.5)))
    return sig

def GWO(objf,lb,ub,dim,searchAgents_no,maxIteration):
    
    alphaPos = np.zeros(dim)
    alphaScore = float("inf")
    
    betaPos = np.zeros(dim)
    betaScore = float("inf")
    
    deltaPos = np.zeros(dim)
    deltaScore = float("inf")
    
    if not isinstance(lb,list):
        lb = [lb] * dim
    if not isinstance(ub,list):
        ub = [ub] * dim
        
    positions = np.zeros((searchAgents_no,dim))
    
    for s in range(0,searchAgents_no):
        for d in range(dim):
            positions[s,d] = 1 if random.random() >= 0.5 else 0
            
    covergenceCurve = np.zeros(maxIteration)
    
    for i in range(0,maxIteration):
        print(i,'/',maxIteration)
        a = 2-i*((2)/maxIteration)
        
        for s in range(0,searchAgents_no):
            
            for d in range(dim):
                positions[s,d] = np.clip(positions[s,d],lb[d],ub[d]) #3
            
            fitness = objf(positions[s,:])
            
            if fitness<alphaScore:
                alphaPos = positions[s,:].copy()
                alphaScore = fitness
               
            
            if fitness>alphaScore and fitness<betaScore:
                deltaPos = betaPos
                deltaScore = deltaScore
                betaScore = fitness
                betaPos = positions[s,:].copy()
                
            if fitness>alphaScore and fitness>betaScore and fitness<deltaScore:
                deltaScore = fitness
                deltaPos = positions[s,:].copy()
    
        for s in range(0,searchAgents_no):
            
            for d in range(0,dim):
                
                # Calculate X1
                r1 = random.random()
                r2 = random.random()
                
                A1 = 2*a*r1-a
                C1 = 2*r2
                
                D_alpha = abs(C1*alphaPos[d] - positions[s,d])
                X1 = alphaPos[d] - A1*D_alpha
                
                # Calculate X2
                r1 = random.random()
                r2 = random.random()
                
                A2 = 2*a*r1-a
                C2 = 2*r2
                
                D_beta = abs(C2*betaPos[d] - positions[s,d])                    
                X2 = betaPos[d]- A2*D_beta
                
                # Calculate X3
                r1 = random.random()
                r2 = random.random()
                
                A3 = 2*a*r1-a
                C3 = 2*r2
                
                D_delta = abs(C3*deltaPos[d] - positions[s,d])
                X3 = deltaPos[d] - A3*D_delta
                
                Xn = (X1+X2+X3)/3

                if(sigmoid(Xn) >= random.random()):
                    positions[s,d] = 1
                else:
                    positions[s,d] = 0

                
        covergenceCurve[i] = alphaScore
        print("score:",round(alphaScore,3))
    
    return alphaPos
