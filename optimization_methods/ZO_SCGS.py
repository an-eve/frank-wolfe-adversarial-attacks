import numpy as np
import tensorflow as tf
import itertools

np.random.seed(2018)

def CG(g_0, u_0, eta, beta, Q):
    u = u_0.reshape(-1)
    g = g_0.reshape(-1)
    d = len(u)
    t = 0

    while True:
        vidx = np.argmin(np.dot(g,Q))
        v=Q[:,vidx]
        print(np.dot(g, u - v))
        print(beta)
        if (np.dot(g, u - v) <= beta) | (t==1000):
            print(t)
            return u.reshape(u_0.shape)
        alpha_t = min(np.dot(g, u - v) / (eta * np.linalg.norm(u - v) ** 2), 1)
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

    q = 10  # 1/p + 1/q = 1 - from the article
    p = 1 / (1 - 1/q)
    dzeta = 3 / (3 + np.arange(N)) # Stepsize
    eta = (8 * np.sqrt(d) * M * M_2) / (epsilon * (np.arange(N) + 3))  # Learning rate
    beta = (2 * np.sqrt(d) * M * M_2 * D ** 2) / (epsilon * (np.arange(N) + 1) * (np.arange(N) + 2)) # Accuracies
    B = np.minimum(q, np.log(d)) * d**(1 - 2/p) * (np.arange(N) + 3)**3 * epsilon**2 / (M * D)**2    # Batch size
    B = np.ceil(B).astype(int)
    print(B)

    #Q = np.random.choice([-1,1],size=(d,10000), p=(0.975,0.025))
    Q = np.concatenate((np.eye(d),-np.eye(d)), axis=1)
    #Q = [num for num in itertools.product([0,1], repeat=d)]
    #Q = np.array(Q)

    for k in range(N):
        randBatchIdx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), B[k], replace=True)
        print(randBatchIdx)
        z = (1 - dzeta[k]) * x + dzeta[k] * y
        # Sampling of e and ksi:
        E = np.random.randn(shp[0],shp[1],B[k])
        norms = np.linalg.norm(E,axis=2,keepdims=True)
        e = E/norms
        ksi = np.random.randn(B[k])

        g = np.zeros(z.shape)
        for idx in range(B[k]):
            g += (1 / B[k]) * (d / (2 * gamma)) * (objfunc.evaluate(z + gamma * e[:,:,idx:idx+1],randBatchIdx[idx:idx+1]) - objfunc.evaluate(z - gamma * e[:,:,idx:idx+1],randBatchIdx[idx:idx+1])) * e[:,:,idx:idx+1]
            #g += (1 / B[k]) * (d / (2 * gamma)) * (objfunc.evaluate(z + gamma * e[:,:,idx:idx+1], np.array([])) - objfunc.evaluate(z - gamma * e[:,:,idx:idx+1], np.array([]))) * e[:,:,idx:idx+1]
        print(g[:,:,0])
        y = CG(g, y, eta[k], beta[k], Q)
        print(y[:,:,0])
        x = (1 - dzeta[k]) * x + (dzeta[k] * y)

        objfunc.evaluate(x,np.array([]),False)

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

    print(x[:,:,0])

    return x