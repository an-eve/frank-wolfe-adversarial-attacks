import numpy as np
import tensorflow as tf
import itertools

np.random.seed(2018)

def CG(g_0, u_0, gamma, eta, Q):
    u = u_0.reshape(-1)
    g = g_0.reshape(-1)
    d = len(u)

    while True:
        V = np.argmax(np.dot(-Q+u,g+(1/gamma)*(u-u_0.reshape(-1))))
        v = np.dot(u-Q[V],g+(1/gamma)*(u-u_0.reshape(-1)))
        if v <= beta:
            return u.reshape(u_0.shape)

        alpha = min( gamma * np.dot((1/gamma)*(u - u_0.reshape(-1)) - g , v-u ) / (np.linalg.norm(v - u) ** 2), 1)

        u = (1 - alpha)*u + alpha*v

def ZOFCGS(x0, N, q, kappa, L, MGR, objfunc):
    best_Loss = 1e10
    best_delImgAT = x0

    n=q**2
    shp = x0.shape
    d=shp[0]*shp[1]
    mu=1/np.sqrt(d*kappa)
    gamma=1/(3*L)
    eta=1/kappa

    x = x0.copy()
    e = np.eye(d)

    Q = np.random.choice([-1,1],size=(10000,d), p=(0.75,0.25))
    #Q = [num for num in itertools.product([0,1], repeat=d)]
    #Q = np.array(Q)

    for k in range(N):
        if (k%q == 0):
            randBatchIdx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), n, replace=False)
            v = np.zeros(x.shape)
            for idx in range(d):
                v += (1/(2*mu))*(objfunc.evaluate(x + mu * e[idx,:].reshape(shp),randBatchIdx) - objfunc.evaluate(x - mu * e[idx,:].reshape(shp),randBatchIdx)) * e[idx,:].reshape(shp)
        else:
            randBatchIdx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), q, replace=True)
            v = np.zeros(x.shape)
            for idx in range(q):
                v += (1/q)*(objfunc.evaluate(x,randBatchIdx[idx:idx+1]) - objfunc.evaluate(xprec,randBatchIdx[idx:idx+1]) + vprec)
        xprec = x.copy()
        vprec = v.copy()
        x = CG(v, x, gamma,eta,Q)

        if(k%1 == 0):
            print('Iteration Index: ', k)
            objfunc.print_current_loss()

        if(objfunc.Loss_Overall < best_Loss):
            best_Loss = objfunc.Loss_Overall
            best_delImgAT = x
            #print('Updating best delta image record')

        MGR.logHandler.write('Iteration Index: ' + str(k))
        MGR.logHandler.write(' Query_Count: ' + str(objfunc.query_count))
        MGR.logHandler.write(' Loss_Overall: ' + str(objfunc.Loss_Overall))
        MGR.logHandler.write(' Loss_Distortion: ' + str(objfunc.Loss_L2))
        MGR.logHandler.write(' Loss_Attack: ' + str(objfunc.Loss_Attack))
        MGR.logHandler.write(' Current_Best_Distortion: ' + str(best_Loss))
        MGR.logHandler.write('\n')

    return x