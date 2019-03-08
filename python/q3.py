import numpy as np
import scipy.io
# Q3.1
def essentialMatrix(F,K1,K2):
    E = None
    E = (K2.T) @ F @ K1
    
    return E

# Q3.2

def triangulate(P1, pts1, P2, pts2):
    P, err = None, None
    N = np.shape(pts1)[0]
    P = np.zeros((N,3))
    A = np.zeros((4,4))
    #generate AP = 0
    for i in range(N):
        A[0,:] = pts1[i,1]* P1[2,:] - P1[1,:] #y*p3-p2
        A[1,:] = P1[0,:]- pts1[i,0] * P1[2,:] #p1-x*p3
        A[2,:] = pts2[i,1]* P2[2,:] - P2[1,:] #y_p*p3_p-p2_p
        A[3,:] = P2[0,:] - pts2[i,0]* P2[2,:] #p1_p-x_p*p3_p
    #SVD to compute P
        u, sigma, vh = np.linalg.svd(A, full_matrices=True)  
        v = vh.T
        h = v[:,-1] 
        h = h.reshape((4,1))
        h = h/h[3,0]
        P[i,:] = [h[0,0],h[1,0],h[2,0]]
    
    extra_ones = np.array(np.ones((N,1)))
    P_homo = (np.concatenate((P, extra_ones),axis=1)).T
    
    #project to 2D plane points
    p1_proj = P1 @ P_homo
    p2_proj = P2 @ P_homo
    
    #convert to homogenous coor
    p1_hat = np.zeros((N,2))
    p2_hat = np.zeros((N,2))
    
    for i in range(N):
        p1_hat[i,:] = (p1_proj[:2,i]/p1_proj[2,i]).T
        p2_hat[i,:] = (p2_proj[:2,i]/p2_proj[2,i]).T
    
    #compute the error(mean squre error)
    mse_1 = np.sum(np.square(pts1 - p1_hat))
    mse_2 = np.sum(np.square(pts2 - p2_hat))
    err = mse_1 + mse_2
  
    return P, err

