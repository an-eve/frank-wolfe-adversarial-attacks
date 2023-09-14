import numpy as np
import tensorflow as tf
import itertools
import time

import Utils as util

np.random.seed(2018)


def ZOFWSA(x0, N, m, SA, MGR, objfunc):
    start_time = time.time()
    best_Loss = 1e10
    best_delImgAT = x0

    shp = x0.shape
    print(shp)
    d=shp[0]*shp[1]

    x=x0.copy()

    num = 400
    Q = np.eye(d)*num
    Qrm = np.full(Q.shape, num/d)
    Q = Q - Qrm
    Q = np.concatenate((Q,-Q), axis=1)

#    h = 6
#    Qpos=np.zeros((d,(shp[0]-h)*shp[1]+shp[0]*(shp[1]-h)))
#    for a in range(shp[0]-h):
#        for b in range(shp[1]):
#            for c in range(h+1):
#                Qpos[b*(shp[0])+a+c,b*(shp[0]-h)+a]+=num/(h+1)
#    for a in range(shp[0]):
#        for b in range(shp[1]-h):
#            for c in range(h+1):
#                Qpos[(b+c)*(shp[0])+a,((shp[0]-h)*shp[1])+(b*(shp[0]-h)+a)]+=num/(h+1)
#    Qrm = np.full(Qpos.shape, num/d)
#    Qpos = Qpos - Qrm
#    Q = np.concatenate((Qpos,-Qpos), axis=1)


    randBatchIdx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), N, replace=True)
    
    dt = np.zeros(d)
    for t in range(N):
        gamma = 2/(t+8)
        g = np.zeros(x.shape)
        grad_x = objfunc.evaluate(x, randBatchIdx[t:t+1])
        if SA == 'KWSA':
            mult = 8
            c = 2/((d**(1/2))*((t+8)**(1/3)))
            rho = 4/((t+8)**(2/3))
            e = np.eye(d)
            for idx in range(d):
                g += (objfunc.evaluate(x + c * e[idx,:].reshape(shp),randBatchIdx[t:t+1]) - grad_x) / c * e[idx,:].reshape(shp)
        elif SA=='RDSA':
            mult = 4
            rho = 4/((d**(1/3))*(t+8)**(2/3))
            c = 2/((d**(3/2))*((t+8)**(1/3)))
            e = np.random.normal(size=(shp[0],shp[1],1))
            g = (objfunc.evaluate(x + c * e[:,:,:],randBatchIdx[t:t+1]) - grad_x) / c * e[:,:,:]
        else:
            mult = 4
            rho = 4/(((1+d/m)**(1/3))*(t+8)**(2/3))
            c = 2*np.sqrt(m)/((d**(3/2))*((t+8)**(1/3)))
            e = np.random.normal(size=(shp[0],shp[1],m))
            for idx in range(m):
                g += (objfunc.evaluate(x + c * e[:,:,idx:idx+1],randBatchIdx[t:t+1]) - grad_x) / c * e[:,:,idx:idx+1]
            g = g/m
        gf = g.reshape(-1)
        dt = (1 - rho) * dt + rho * gf

        #vidx = np.argmin(np.dot(dt,Q))
        #v = Q[:,vidx]

        v = -np.sign(dt)*mult

        v = v.reshape(shp)
        
        #print(x.shape)
        #print(v.shape)
        x = (1 - gamma) * x + gamma * v


        objfunc.evaluate(x,np.array([]),False)

        if((t%1 == 0 and SA!="RDSA") or (t%100 == 0)):
            print('Iteration Index: ', t)
            objfunc.print_current_loss()

        if(objfunc.Loss_Overall < best_Loss):
            best_Loss = objfunc.Loss_Overall
            best_delImgAT = x
            #print('Updating best delta image record')

        #util.save_img(np.tanh(x/4)/2.0, "{}/Delta_{}.png".format(MGR.parSet['save_path'],t))

        MGR.logHandler.write('Iteration Index: ' + str(t))
        MGR.logHandler.write(' Query_Count: ' + str(objfunc.query_count))
        MGR.logHandler.write(' Time: ' + str(time.time()-start_time))
        MGR.logHandler.write(' Loss_Overall: ' + str(objfunc.Loss_Overall))
        MGR.logHandler.write(' Loss_Distortion: ' + str(objfunc.Loss_L2))
        MGR.logHandler.write(' Loss_Attack: ' + str(objfunc.Loss_Attack))
        MGR.logHandler.write(' Current_Best_Distortion: ' + str(best_Loss))
        MGR.logHandler.write('\n')

    print(best_delImgAT[:,:,0])

    return best_delImgAT