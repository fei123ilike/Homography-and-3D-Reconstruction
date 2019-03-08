import numpy as np
from q2 import eightpoint, sevenpoint
from q3 import triangulate
from q4 import epipolarCorrespondence, visualize, visualizeDense
from skimage.color import rgb2gray
import scipy
import skimage.io
from util import plot_epipolar_lines
# Q 5.1
# we're going to also return inliers
def ransacF(pts1, pts2, M):
    F = None
    inliers = None
    iteration=5000
    num=np.shape(pts1)[0]
    bestInliers=0
    inliers_max=np.zeros((1,num))
    threshold = 0.8

    ones = np.zeros((1,num))
    x1 = np.vstack((pts1.T,ones))
    x2 = np.vstack((pts2.T,ones))
    
    for i in range(iteration):
        randomIdx = np.random.randint(0,num,8) 
        random_1 = np.array(pts1[randomIdx,:])
        random_2 = np.array(pts2[randomIdx,:])
        F8 = eightpoint(random_1,random_2,M)
        inliers=np.zeros((1,num))
#        epiline = F8 @ x1
        for j in range(num):
            
#            predict = x2[:,j].T @ F8 @ x1[:,j]
#            if abs(predict) < threshold:
#                inliers[0,j]=1
#              
           l1=np.array([[pts1[j][0]],[pts1[j][1]],[1]])
           #calculate the predict points
           epiline = F8 @ l1
           #epiline = epiline / np.linalg.norm(epiline)
           a = epiline[0]
           b = epiline[1]
           c = epiline[2]
           numer = np.abs(a*pts2[j][0] +  b*pts2[j][1] + c)
           denom= np.sqrt(a**2 + b**2)
           distance = numer/denom
           if distance <= threshold:
                inliers[0,j]=1
        numInliers=np.sum(inliers)
        #update number of inliers, store the biggest one
        if numInliers>bestInliers:
            bestInliers=numInliers
            inliers_max=inliers
            

    inliers= inliers_max
    idx = []
    for i in range (num):
        if inliers[:,i]==1:
            idx.append(i)
    idx = np.array(idx)
    
    F = eightpoint(pts1[idx,:],pts2[idx,:],M)
    
    return F, idx

# Q 5.2
# r is a unit vector scaled by a rotation around that axis
# 3x3 rotatation matrix is R
# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# http://www.cs.rpi.edu/~trink/Courses/RobotManipulation/lectures/lecture6.pdf
def rodrigues(r):
    R = None
    I = np.eye((3))
    theta = np.linalg.norm(r)
    if theta > 1e-30:
        n = r/theta
        K = np.array([[0,-n[2],n[1]],
                      [n[2],0,-n[0]],
                      [-n[1],n[0],0]])
        R = I + (np.sin(theta)) * K + (1-np.cos(theta))*np.dot(K,K)
    else:
         K2 = np.array([[0,-r[2],r[1]],
                      [r[2],0,-r[0]],
                      [-r[1],r[0],0]])
         theta2 = theta**2
         R = I + (1-theta2/6.)*K2 + (.5-theta2/24.)*dot(K2,K2)
    return R

#r = np.array([[3],[3],[5]])
#r = r/np.linalg.norm(r)
#R = rodrigues(r)
    
# Q 5.2
# rotations about x,y,z is r
# 3x3 rotatation matrix is R
# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
def invRodrigues(R):
    r = None
    theta = np.arccos((np.trace(R)-1)/2)
    #coef = 1/(np.sin(theta))
    coef = 1/(2*np.sin(theta))
    r = np.array([[R[2,1]-R[1,2]],
                  [R[0,2]-R[2,0]],
                  [R[1,0]-R[0,1]]])
    r = coef * r * theta
    r = r.reshape((1,3))
    return r
#r = invRodrigues(R)
    
# Q5.3
# we're using numerical gradients here
# but one of the great things about the above formulation
# is it has a nice form of analytical gradient
def extract(x):
    p = np.reshape(x,((int)(len(x)/3),3))
    r = p[-2,:]
    t = p[-1,:]
    t = np.reshape(t,(3,1))
    R =rodrigues(r)
    return p,R,t

def rodriguesResidual(K1, M1, p1, K2, p2, x):
    residuals = None
    p,R,t = extract(x)
#    print(shape(R))
#    print(shape(t))
    M2 = np.concatenate((R,t),axis=1)
    
    C1 = K1 @ M1
    C2 = K2 @ M2
    
    N = (np.shape(p1)[0])
    extra_ones = np.array(np.ones((N,1)))
    P_homo = (np.concatenate((p[:-2][:], extra_ones),axis=1)).T
    
    p1_proj = C1 @ P_homo
    p2_proj = C2 @ P_homo
    
    #convert to homogenous coor
    p1_hat = np.zeros((N,2))
    p2_hat = np.zeros((N,2))
    
    for i in range(N):
        p1_hat[i,:] = (p1_proj[:2,i]/p1_proj[2,i]).T
        p2_hat[i,:] = (p2_proj[:2,i]/p2_proj[2,i]).T
    
    #compute the error(mean squre error)
    mse_1 = np.sum(np.square(p1 - p1_hat))
    mse_2 = np.sum(np.square(p2 - p2_hat))
    residuals = mse_1 + mse_2
    return residuals

# we should use scipy.optimize.minimize
# L-BFGS-B is good, feel free to use others and report results
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    
def minimize(x,K1, M1, p1, K2, p2):
    return rodriguesResidual(K1, M1, p1, K2, p2, x)

def bundleAdjustment(K1, M1, p1, K2, M2init, p2,Pinit):
    M2, P = None, None
    R2 = M2init[:,:3]
    t2 = M2init[:,3]
    t2 = np.reshape(t2,(1,3))
    r2 = invRodrigues(R2)
    p = np.concatenate((Pinit[:,:3],r2,t2),axis=0)
    x = np.reshape(p,(len(p)*3,1))
    result = scipy.optimize.minimize(minimize,x,args=(K1, M1, p1, K2, p2),method='L-BFGS-B')
    result_p,result_R,result_t=extract(result.x)
    M2 = np.concatenate((result_R,result_t),axis=1)
    P = result_p[:-2][:]
    return M2,P 