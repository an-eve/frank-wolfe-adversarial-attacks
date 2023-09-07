import numpy as np
import tensorflow as tf
import itertools

import Utils as util

np.random.seed(2018)

def CG(g_0, u_0, eta, beta, Q):
    u = u_0.reshape(-1)
    g = g_0.reshape(-1)
    d = len(u)
    t = 0
    alpha_t=1

    while True:
        vidx = np.argmin(np.dot(g,Q))
        v=Q[:,vidx]
        dot=np.dot(g, u - v)
        #print(str(dot) + " " + str(beta))
        if (dot <= beta) or (t==1000000) or (alpha_t<0.00001):
            print('T number in CG = ' + str(t))
            print("final alpha_t: " + str(alpha_t))
            print(str(dot) + " " + str(beta))
            #print(u.reshape((28,28)))
            return u.reshape(u_0.shape)
        alpha_t = min(np.dot(g, u - v) / (eta * np.linalg.norm(u - v) ** 2), 1)
        #print("alpha_t: " + str(alpha_t))
        u = u + alpha_t * (v - u)
        g = g_0.reshape(-1) + eta * (u - u_0.reshape(-1))
        t+=1

def ZOSCGS(x0, N, M, M_2, epsilon, D, gamma, MGR, objfunc):
    best_Loss = 1e10
    best_delImgAT = x0

    shp = x0.shape
    d=shp[0]*shp[1]

    x = x0.copy()
    y = x0.copy()

    #q = 10  # 1/p + 1/q = 1 - from the article
    #p = 1 / (1 - 1/q)
    q=10000000
    p=1
    dzeta = 3 / (3 + np.arange(N)) # Stepsize
    eta = (8 * np.sqrt(d) * M * M_2) / (epsilon * (np.arange(N) + 3))  # Learning rate
    beta = (2 * np.sqrt(d) * M * M_2 * D ** 2) / (epsilon * (np.arange(N) + 1) * (np.arange(N) + 2)) # Accuracies
    B = np.minimum(q, np.log(d)) * d**(1 - 2/p) * (np.arange(N) + 3)**3 * epsilon**2 / (M * D)**2    # Batch size
    B = np.ceil(B).astype(int)
    print(B)
    print(beta)
    print(eta)
    
    num = 2000
    
    #h = 6
    #Qpos=np.zeros((d,(shp[0]-h)*shp[1]+shp[0]*(shp[1]-h)))
    #for a in range(shp[0]-h):
    #    for b in range(shp[1]):
    #        for c in range(h+1):
    #            Qpos[b*(shp[0])+a+c,b*(shp[0]-h)+a]+=num/(h+1)
    #for a in range(shp[0]):
    #    for b in range(shp[1]-h):
    #        for c in range(h+1):
    #            Qpos[(b+c)*(shp[0])+a,((shp[0]-h)*shp[1])+(b*(shp[0]-h)+a)]+=num/(h+1)
    #Q=Qpos
    #Q = np.concatenate((Qpos,-Qpos),axis=1)
    #Q = np.concatenate((np.eye(d),-np.eye(d)), axis=1)
    Q = np.eye(d)*num
    #Q = np.transpose(Q)
    Qrm = np.full(Q.shape, num/d)
    Q = Q - Qrm
    Q= np.concatenate((Q,-Q), axis=1)
    #print(Q)

    for k in range(N):
        B[k]=50 #used for fixed B
        randBatchIdx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), B[k], replace=True)
        print(randBatchIdx)
        #z = (1 - dzeta[k]) * x + dzeta[k] * y
        z = x
        # Sampling of e:
        E = np.random.uniform(-1.0, 1.0, size=(shp[0],shp[1],B[k]))
        norms = np.linalg.norm(E,axis=(0,1),keepdims=True)
        e = E/norms

        #print(e[:,:,0])
        g = np.zeros(z.shape)
        for idx in range(B[k]):
            #print("index: " +str(idx))
            #print("x0 " +str(objfunc.evaluate(x0,randBatchIdx[idx:idx+1])))
            #print("z " +str(objfunc.evaluate(z,randBatchIdx[idx:idx+1])))
            #print(gamma * e[:,:,idx])
            g += (1 / B[k]) * (d / (2 * gamma)) * (objfunc.evaluate(z + gamma * e[:,:,idx:idx+1],randBatchIdx[idx:idx+1]) - objfunc.evaluate(z - gamma * e[:,:,idx:idx+1],randBatchIdx[idx:idx+1])) * e[:,:,idx:idx+1]
            #g += (1 / B[k]) * (d / (2 * gamma)) * (objfunc.evaluate(z + gamma * e[:,:,idx:idx+1], np.array([])) - objfunc.evaluate(z - gamma * e[:,:,idx:idx+1], np.array([]))) * e[:,:,idx:idx+1]
        #print(g[:,:,0])
        y = CG(g, y, eta[k], beta[k], Q)
        #print(y[:,:,0])
        x = (1 - dzeta[k]) * x + (dzeta[k] * y)
        print(np.sum(np.abs(x-x0)))
        print(np.sum(x-x0))

        objfunc.evaluate(x,np.array([]),False)

        if(k%1 == 0):
            print('Iteration Index: ', k)
            objfunc.print_current_loss()

        if(objfunc.Loss_Overall < best_Loss):
            best_Loss = objfunc.Loss_Overall
            best_delImgAT = x
            #print('Updating best delta image record')

        #util.save_img(np.tanh(x/4)/2.0, "{}/Delta_{}.png".format(MGR.parSet['save_path'],k))

        MGR.logHandler.write('Iteration Index: ' + str(k))
        MGR.logHandler.write(' Query_Count: ' + str(objfunc.query_count))
        MGR.logHandler.write(' Loss_Overall: ' + str(objfunc.Loss_Overall))
        MGR.logHandler.write(' Loss_Distortion: ' + str(objfunc.Loss_L2))
        MGR.logHandler.write(' Loss_Attack: ' + str(objfunc.Loss_Attack))
        MGR.logHandler.write(' Current_Best_Distortion: ' + str(best_Loss))
        MGR.logHandler.write('\n')

    print(x[:,:,0])

    return best_delImgAT