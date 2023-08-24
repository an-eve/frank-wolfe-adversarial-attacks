import numpy as np

np.random.seed(2018)

def CG(g_0, u_0, eta, beta):
    T = 10 #len(beta)
    u = u_0.copy()
    g = g_0.copy()
    d = len(u)

    Q = np.eye(d)  # Define the feasible set Q as an identity matrix (for simplex constraint)

    for t in range(T+1): # probably T+1
        v_t = np.argmin(np.dot(g, Q))
        #v_t = Q[np.argmin(np.dot(g, Q))]

        if np.dot(g, u - v_t) <= beta:
            return u

        alpha_t = min(np.dot(g, u - v_t) / (eta * np.linalg.norm(u - v_t) ** 2), 1)
        u = u + alpha_t * (v_t - u)
        g = g_0 + eta * (u - u_0)

    return u

def ZOSCGS(x0, N, M, M_2, epsilon, D, gamma, objfunc):
    best_Loss = 1e10
    best_delImgAT = x0

    d = len(x0)
    x = x0.copy()
    y = x0.copy()

    q = 10  # 1/p + 1/q = 1 - from the article
    p = 1 / (1 - 1/q)
    dzeta = 3 / (3 + np.arange(N)) # Stepsize
    eta = (8 * np.sqrt(d) * M * M_2) / (epsilon * (np.arange(N) + 3))  # Learning rate
    beta = (2 * np.sqrt(d) * M * M_2 * D ** 2) / (epsilon * (np.arange(N) + 1) * (np.arange(N) + 2)) # Accuracies
    B = np.minimum(q, np.log(d)) * d**(1 - 2/p) * (np.arange(N) + 3)**3 * epsilon**2 / (M * D)**2    # Batch size
    B = np.ceil(B).astype(int)

    for k in range(N):
        z = (1 - dzeta[k]) * x + dzeta[k] * y
        # Sampling of e and ksi:
        e = np.random.randn(B[k], d)
        ksi = np.random.randn(B[k])

        g = (1 / B[k]) * np.sum((d / (2 * gamma)) * (objfunc.evaluate(z + gamma * e,[]) - objfunc.evaluate(z - gamma * e,[])) * e, axis=0) #f is missing ksi
        y = CG(g, y, eta[k], beta[k])
        x = (1 - dzeta[k]) * x + dzeta[k] * y

        if(k%10 == 0):
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