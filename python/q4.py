import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import scipy.stats as st
from q2 import eightpoint
from q3 import essentialMatrix, triangulate
from util import camera2
from skimage.color import rgb2gray
import skimage.io
import scipy.io
# Q 4.1
# np.pad may be helpful
def guassian_filter(nlen, nsig=4):
    
    interval = (2*nsig+1.)/(nlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., nlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def epipolarCorrespondence(im1, im2, F, x1, y1):
    x2, y2 = 0, 0
    im1 = rgb2gray(im1)
    im2 = rgb2gray(im2)
    im1 = np.pad(im1, [(3, 3), (3, 3)], mode='constant')
    im2 = np.pad(im2, [(3, 3), (3, 3)], mode='constant')
    m,n = np.shape(im2)
    #compute the scaled epipolar line
    pts1 = [x1,y1,1]
    epiline = F @ pts1
    epiline = epiline / np.linalg.norm(epiline)
    a = epiline[0]
    b = epiline[1]
    c = epiline[2]
    #initialize parameters
    stride = 4
    min_dis = 200
    sigma = 4
    filter_window = guassian_filter(9)
    x1 = np.round(x1)
    y1 = np.round(y1)
    patch1 = im1[int(y1 - stride):int(y1 + stride+1),int(x1 - stride):int(x1 + stride+1)]
    #iterating along the epline to find matched x2, y2
    for i in range (int(y1-20),int(y1+20)):
    #for i in range (m):
        x2_curr = np.round((-b * i - c)/ a)
        if x2_curr - stride > 0 and x2_curr + stride+1 <= np.size(im2,1) and i - stride>0 and i + stride +1<= np.size(im2,0):
            # small patch in im2 centered at x2, y2
            patch2 = im2[int(i-stride):int(i+stride+1),int(x2_curr-stride):int(x2_curr+stride+1)]
            dis = patch1 - patch2
            weighted_dis = filter_window * dis
            #Euclidean disdance
            err = np.sqrt(np.sum(np.square(weighted_dis[:])))
            if err < min_dis:
                min_dis = err
                x2 = x2_curr
                y2 = i
    
    return x2, y2

# Q 4.2
# this is the "all in one" function that combines everything
def visualize(IM1_PATH,IM2_PATH,TEMPLE_CORRS,F,K1,K2):
    
    fig = plt.figure()
#    ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim/set_ylim/set_zlim/
    # ax.set_aspect('equal')
    # may be useful
    # you'll want a roughly cubic meter
    # around the center of mass
    im1 = skimage.io.imread(IM1_PATH)
    im2 = skimage.io.imread(IM2_PATH)
    im1 = rgb2gray(im1)
    im2 = rgb2gray(im2)
    
    coor = scipy.io.loadmat(TEMPLE_CORRS)
    x1 = coor['x1']
    y1 = coor['y1']
    pts1 = np.hstack((x1,y1))
    
    pts2 = []
    x2 = np.zeros((np.shape(x1)[0]))
    y2 = np.zeros((np.shape(y1)[0]))
    for i in range(np.shape(x1)[0]):
        p2e = epipolarCorrespondence(im1, im2, F, pts1[i,0], pts1[i,1])
        pts2.append(p2e)
    pts2 = np.array(pts2)
   
    
    # M1 = [I|0];
    M1 = np.zeros((3,4))
    M1[0,0] = 1
    M1[1,1] = 1
    M1[2,2] = 1
    C1 = K1 @ M1
    # M2 = [R|t];
    #Go through 4 M2 and store the outputs coor
    E = essentialMatrix(F,K1,K2)
    M2e = camera2(E)
    P = np.zeros((np.size(x1),3))
    err = np.zeros((4,1))
    z_position = np.zeros((4,1))
    #find thr posotive Z coor
    for i in range (4):
        C2 = K2 @ M2e[i,:,:]
        p, err = triangulate( C1, pts1, C2, pts2 )
        if (p[:,2]>0).all():
            P = p
            M2 = M2e[i,:,:]
#        P_total[:,:,i] = P
#        idx = np.where(P[:,2]>0)
#        z_position[i,:] = length(idx)
#    correct = np.where(z_position == np.size(P,0))
    
    PX = P[:,0]
    PY = P[:,1]
    PZ = P[:,2]
        
    ax.scatter(PX, PY, PZ, c='blue', marker = 'o',s=3)
#    ax.set_aspect('equal') 
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    scipy.io.savemat('q4_2.mat', {'M1':M1,'M2':M2,'C1':C1,'C2':C2,})
# Extra credit
def visualizeDense(IM1_PATH,IM2_PATH,TEMPLE_CORRS,F,K1,K2):
    fig = plt.figure()
#    ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim/set_ylim/set_zlim/
    # ax.set_aspect('equal')
    # may be useful
    # you'll want a roughly cubic meter
    # around the center of mass
    im1 = skimage.io.imread(IM1_PATH)
    im2 = skimage.io.imread(IM2_PATH)
    im1 = rgb2gray(im1)
    im2 = rgb2gray(im2)
    pts1 = []
    pts2 = []

#    for i in range(np.shape(im1)[0]):
#        for j in range(np.shape(im1)[1]):
#            a = [i,j]
#            pts1.append(a)
    npoint = 150000
    randomIdx_x = np.random.randint(5,635,npoint) 
    randomIdx_y = np.random.randint(5,475,npoint) 
    pts1 = []
    for i in range(npoint):
        a = np.array([randomIdx_x[i],randomIdx_y[i]])
        pts1.append(a)
    pts1 = np.array(pts1)
   
    for i in range(np.shape(pts1)[0]):
        p2e = epipolarCorrespondence(im1, im2, F, pts1[i,0], pts1[i,1])
        pts2.append(p2e)
    pts2 = np.array(pts2)
   
    
    # M1 = [I|0];
    M1 = np.zeros((3,4))
    M1[0,0] = 1
    M1[1,1] = 1
    M1[2,2] = 1
    C1 = K1 @ M1
    # M2 = [R|t];
    #Go through 4 M2 and store the outputs coor
    E = essentialMatrix(F,K1,K2)
    M2e = camera2(E)
    P = np.zeros((np.shape(pts1)[0],3))
    err = np.zeros((4,1))
    z_position = np.zeros((4,1))
    #find thr posotive Z coor
    for i in range (4):
        C2 = K2 @ M2e[i,:,:]
        p, err = triangulate( C1, pts1, C2, pts2 )
        if (p[:,2]>0).all():
            P = p
            M2 = M2e[i,:,:]
#        P_total[:,:,i] = P
#        idx = np.where(P[:,2]>0)
#        z_position[i,:] = length(idx)
#    correct = np.where(z_position == np.size(P,0))
    
    PX = P[:,0]
    PY = P[:,1]
    PZ = P[:,2]
        
    ax.scatter(PX, PY, PZ, c='blue', marker = 'o', s = 0.01)
#    ax.set_aspect('equal') 
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_zlim(3.70,4.00)
    plt.show()