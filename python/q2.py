import numpy as np
from scipy.optimize import fsolve
import scipy.io
# Q 2.1
def eightpoint(pts1, pts2, M):
    #F = None
    N = np.shape(pts1)[0]
    A = np.zeros((N,9))
    width = 640
    height = 480
    #normalize pts1 and pts2 with thw matrix T
    Tmatrix = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]])
#    Tmatrix = np.array([[1/width,0,0],[0,1/height,0],[0,0,1]])
    extra_ones = np.array(np.ones((N,1)))
    dummy1 = (np.concatenate((pts1, extra_ones),axis=1)).T
    dummy2 = (np.concatenate((pts2, extra_ones),axis=1)).T
    norm_pts1 = (Tmatrix @ dummy1).T
    norm_pts2 = (Tmatrix @ dummy2).T
    for i in range(N):
        x,y   = norm_pts1[i,0],norm_pts1[i,1]
        xp,yp = norm_pts2[i,0],norm_pts2[i,1]
        A[i,:] = [ x*xp, x*yp, x, y*xp, y*yp,  y,  xp,   yp,   1]  
    #compute F    
    u, sigma, vh = np.linalg.svd(A, full_matrices=True)  
    v = vh.T
    h = v[:,-1] 
    F = np.reshape(h,(3,3))
    F = F.T
    #project onto essential space (enforce singularity condition)
    U,S,Vh = np.linalg.svd(F, full_matrices=True)
    S = [[S[0],0,0],[0,S[1],0],[0,0,0]]
    F = U @ S @ Vh
    #denormalized F
    F = Tmatrix.T @ F @ Tmatrix
    F = F/F[2,2]
    
    return F

# Q 2.2
# you'll probably want fsolve

def determinant(s, *data):
     F_8, F_9 = data
     return np.linalg.det(s*F_8 +(1-s)*F_9)

def sevenpoint(pts1, pts2, M=1):
    #F = None
    #assert(pts1.shape[0] == 7)
    #assert(pts2.shape[0] == 7)
    N = pts1.shape[0]
    A = np.zeros((N,9))
    width = 640
    height = 480
    #normalize pts1 and pts2 with thw matrix T
    Tmatrix = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]])
#    Tmatrix = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]])
    extra_ones = np.array(np.ones((N,1)))
    dummy1 = (np.concatenate((pts1, extra_ones),axis=1)).T
    dummy2 = (np.concatenate((pts2, extra_ones),axis=1)).T
    norm_pts1 = (Tmatrix @ dummy1).T
    norm_pts2 = (Tmatrix @ dummy2).T
    for i in range(N):
        x,y   = norm_pts1[i,0],norm_pts1[i,1]
        xp,yp = norm_pts2[i,0],norm_pts2[i,1]
        A[i,:] = [ x*xp, x*yp, x, y*xp, y*yp,  y,  xp,   yp,   1]  
    #compute F    
    u, sigma, vh = np.linalg.svd(A, full_matrices=True)  
    v = vh.T
    f_9 = v[:,-1] 
    f_8 = v[:,-2] 
    F_9 = np.reshape(f_9,(3,3))
    F_8 = np.reshape(f_8,(3,3))
    
    data = (F_8,F_9)
    
    s0 = 1
    s = fsolve(determinant, s0, args=data)
    
    F = s*F_8 + (1-s)*F_9
    F = Tmatrix.T @ F @ Tmatrix
    F = F/F[2,2]
    return F
