import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from sklearn.cluster import KMeans
import random
import math
 
def main():

    n_classes = 3
    orig_dim = 100
    radius = 4
    ball_capacity = np.matrix(20)
    max_req_balls = 0
    n_balls = 0
    balls_limit = 1000
    threshold_ratio = 0.1

    hit_count = 0

    ball_classes = []
    ball_capacity_classes = []

    ball_centres = np.matrix([])

    a = [100]*orig_dim
    b = [1]*orig_dim

    for i in range(n_classes):
        ball_classes.append(np.matrix([a]))
        ball_capacity_classes.append(np.matrix(0))

    #print(ball_capacity_classes[0])
    #print(ball_capacity_classes[1])
    #print(ball_classes[1])
    # #print(ball_classes[2])

    
    tr_data = load_svmlight_file("train60000.txt")
    X = np.matrix(tr_data[0].toarray()); # Converts sparse matrices to dense
    Y = np.matrix(tr_data[1]);

    Y = Y.T
    newX = np.matrix(np.zeros(X.shape[1]))
    newY = np.zeros(1)
    classes = np.matrix([])
    for j in range(X.shape[0]):
        ##print(j)
        x = X[j]
        y = Y[j]
        ##print((ball_classes[int(y)-1]).)
        if((ball_classes[int(y)-1]).shape[0]==1):
            ball_classes[int(y)-1] = np.append(ball_classes[int(y)-1],x,axis = 0)
            ball_capacity_classes[int(y)-1] = np.append(ball_capacity_classes[int(y)-1],ball_capacity,axis = 0)
            n_balls = n_balls + 1

        else:
            dists = ball_classes[int(y)-1] - x
            dists = np.square(dists)
            dists = np.sum(dists,axis = 1)
            ##print(min(dists))            
            


            if(min(dists) < radius**2 and ball_capacity_classes[int(y)-1][np.argmin(dists)] > 0):
                ##print(np.argmin(dists))
                hit_count = hit_count+1
                ball_capacity_classes[int(y)-1][np.argmin(dists)] = ball_capacity_classes[int(y)-1][np.argmin(dists)]-1

                # if(ball_capacity_classes[int(y)-1][np.argmin(dists)] == 0):
                #     n_balls = n_balls-1
                ##print(hit_count)
            else:

                ball_classes[int(y)-1] = np.append(ball_classes[int(y)-1],x,axis = 0)
                ball_capacity_classes[int(y)-1] = np.append(ball_capacity_classes[int(y)-1],ball_capacity,axis = 0)
                n_balls = n_balls+1
        # print(n_balls, balls_limit)
        if(n_balls == balls_limit):
            print("CAPACITY FULL... DUMPING")
            for i in range(n_classes):
                ball_classes[i] = ball_classes[i][1:]
                #print(i)
            if(classes.size == 0):
                balls_pack = ball_classes[0]
                classes = np.repeat(1,ball_classes[0].shape[0],axis = 0)
            else : 
                classes = np.append(classes,np.repeat(1,ball_classes[0].shape[0],axis = 0))
                balls_pack = np.append(balls_pack,ball_classes[0],axis = 0)
            
            for i in range(1,n_classes):
                balls_pack = np.append(balls_pack,ball_classes[i],axis = 0)
                classes = np.append(classes,np.repeat(i+1,ball_classes[i].shape[0],axis = 0),axis = 0)
            
            ball_classes = []
            ball_capacity_classes = []
            for i in range(n_classes):
                ball_classes.append(np.matrix([a]))
                ball_capacity_classes.append(np.matrix(0))

            n_balls = 0

            print("BALLS DUMPED!!")
        #print(n_balls)
        ##print(x,y)
    #print("This class has "+str(ball_capacity_classes[0].shape) + " balls")
    #print("This class has "+str(ball_capacity_classes[1].shape) + " balls")
    #print("This class has "+str(ball_capacity_classes[2].shape) + " balls")
    # #print (ball_classes)

    #******************************************************************************
    for i in range(n_classes):
        ball_classes[i] = ball_classes[i][1:]
        #print(i)
    if(classes.size == 0):
        balls_pack = ball_classes[0]
        classes = np.repeat(1,ball_classes[0].shape[0],axis = 0)
    else : 
    	classes = np.append(classes,np.repeat(1,ball_classes[0].shape[0],axis = 0))
    	balls_pack = np.append(balls_pack,ball_classes[0], axis=0)
    
    for i in range(1,n_classes):
        balls_pack = np.append(balls_pack,ball_classes[i],axis = 0)
        classes = np.append(classes,np.repeat(i+1,ball_classes[i].shape[0],axis = 0),axis = 0)

    ball_classes = []
    ball_capacity_classes = []
    for i in range(n_classes):
        ball_classes.append(np.matrix([a]))
        ball_capacity_classes.append(np.matrix(0))

    #******************************************************************************
    dump_svmlight_file(balls_pack, classes, "zeroBalls", zero_based=False)


    #print(n_balls)
    quit()

    #print(newX.shape)
    #print(newY.shape)
    dump_svmlight_file(newX[1:], newY[1:], "newReduced", zero_based=False)

if __name__ == '__main__':
    main()