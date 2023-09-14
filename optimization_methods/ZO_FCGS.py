import numpy as np
import tensorflow as tf
import itertools
import time

import Utils as util

np.random.seed(2018)

def CG(g_0, u_0, eta, beta, Q):
    u = u_0.reshape(-1)
    g = g_0.reshape(-1)
    d = len(u)
    t = 0
    alpha_t=1

    while True:
        #vidx = np.argmin(np.dot(g,Q))
        #v=Q[:,vidx]

        v = -np.sign(g)*4

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

def CGfcgs(g_0, u_0, gamma, eta, Q):
    u = u_0.reshape(-1)
    g = g_0.reshape(-1)
    d = len(u)
    t = 0
    alpha=1

    while True:
        uexp = np.repeat(np.expand_dims(u,axis=1),Q.shape[1],axis=1)
        vidx = np.argmax(np.dot(g+(1/gamma)*(u-u_0.reshape(-1)),-Q+uexp))
        v = Q[:,vidx]
        dot = np.dot(g+(1/gamma)*(u-u_0.reshape(-1)),-Q[:,vidx]+u)
        #print("dot product = " + str(dot) + " ; eta = " + str(eta))
        if (dot <= eta) or (t == 1000000) or (alpha<0.00001):
            print('T number in CG = ' + str(t))
            print("final alpha: " + str(alpha))
            print(str(dot) + " " + str(eta))
            return u.reshape(u_0.shape)

        alpha = min( gamma * np.dot((1/gamma)*(u - u_0.reshape(-1)) - g , v-u ) / (np.linalg.norm(v - u) ** 2), 1)
        #print("alpha = " + str(alpha))
        u = (1 - alpha)*u + alpha*v
        t+=1

def ZOFCGS(x0, N, q, kappa, L, MGR, objfunc):
    start_time = time.time()
    best_Loss = 1e10
    best_delImgAT = x0

    n=q**2
    shp = x0.shape
    d=shp[0]*shp[1]
    #mu=1/np.sqrt(d*kappa)
    mu=0.01
    gamma=1/(3*L)
    #eta=1/kappa
    eta = 0.1
    
    x = x0.copy()
    e = np.eye(d)
    
    num = 2000
    Q = np.eye(d)*num
    Qrm = np.full(Q.shape, num/d)
    Q = Q - Qrm
    Q= np.concatenate((Q,-Q), axis=1)

    B=50

    for k in range(N):
        if (k%q == 0):
            randBatchIdx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), n, replace=False)
            E = np.random.uniform(-1.0, 1.0, size=(shp[0],shp[1],B))
            norms = np.linalg.norm(E,axis=(0,1),keepdims=True)
            e = E/norms
            v = np.zeros(x.shape)
            for idx in range(B):
                v += (1/B)*(1/(2*mu))*(objfunc.evaluate(x + mu * e[:,:,idx:idx+1],randBatchIdx) - objfunc.evaluate(x - mu * e[:,:,idx:idx+1],randBatchIdx)) * e[:,:,idx:idx+1]
            #for idx in range(d):
            #    (1/(2*mu))*(objfunc.evaluate(x + mu * e[idx,:].reshape(shp),randBatchIdx) - objfunc.evaluate(x - mu * e[idx,:].reshape(shp),randBatchIdx)) * e[idx,:].reshape(shp)
        else:
            randBatchIdx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), q, replace=True)
            v = np.zeros(x.shape)
            for qidx in range(q):
                print("qidx = " + str(qidx))
                for idx in range(B):
                    v += (objfunc.evaluate(x + mu * e[:,:,idx:idx+1],randBatchIdx[qidx:qidx+1]) - objfunc.evaluate(x - mu * e[:,:,idx:idx+1],randBatchIdx[qidx:qidx+1])) * e[:,:,idx:idx+1] - ((objfunc.evaluate(xprec + mu * e[:,:,idx:idx+1],randBatchIdx[qidx:qidx+1]) - objfunc.evaluate(xprec - mu * e[:,:,idx:idx+1],randBatchIdx[qidx:qidx+1])) * e[:,:,idx:idx+1])
                #for idx in range(d):
                #    v += ((objfunc.evaluate(x + mu * e[idx,:].reshape(shp),randBatchIdx[qidx:qidx+1]) - objfunc.evaluate(x - mu * e[idx,:].reshape(shp),randBatchIdx[qidx:qidx+1])) * e[idx,:].reshape(shp)) - ((objfunc.evaluate(xprec + mu * e[idx,:].reshape(shp),randBatchIdx[qidx:qidx+1]) - objfunc.evaluate(xprec - mu * e[idx,:].reshape(shp),randBatchIdx[qidx:qidx+1])) * e[idx,:].reshape(shp))
            v = (1/B)*(1/(2*mu))*(v/q)+vprec
        #print(v[:,:,0])
        xprec = x.copy()
        vprec = v.copy()


        x = CG(v, x, gamma,eta,Q)

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
        MGR.logHandler.write(' Time: ' + str(time.time()-start_time))
        MGR.logHandler.write(' Loss_Overall: ' + str(objfunc.Loss_Overall))
        MGR.logHandler.write(' Loss_Distortion: ' + str(objfunc.Loss_L2))
        MGR.logHandler.write(' Loss_Attack: ' + str(objfunc.Loss_Attack))
        MGR.logHandler.write(' Current_Best_Distortion: ' + str(best_Loss))
        MGR.logHandler.write('\n')

    print(x[:,:,0])

    return best_delImgAT