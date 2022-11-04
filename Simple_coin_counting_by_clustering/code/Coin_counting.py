import numpy as np
import cv2
import glob
import os 
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN

# import and load data and also rename all of samples (images) to normal name (img_0) to (img_n)
data_path = "../data"
dirc_images = glob.glob(os.path.join(data_path,"*"))
for number, filename in enumerate(dirc_images):
    try:
        os.rename(filename,os.path.join(data_path,"img_{0}.jpg".format(number)))
    except OSError as e:
        print("Something happened:", e)
dirc_images = glob.glob(os.path.join(data_path,"*"))

# make figure for display
fig = plt.figure(figsize =(25,35))
fig.subplots_adjust(hspace = 0.5,wspace=0.1)
fig.suptitle('Coin Counter by Clustering(DBSCAN)')

# main of code
for i in range(9):
    path = dirc_images[i]
    img = cv2.imread(path)
    # show orginal image before resize it 
    # ax = fig.add_subplot(3, 3, i)
    # ax.imshow(img)
    # ax.set_title("Input")

    # resize 
    img = cv2.resize(img, (0,0), fx=0.03, fy=0.03)
    # change bgr to gray and make contour of gray version(contour make background to 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
    image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # show contour 
    # ax = fig.add_subplot(3, 3, i)
    # ax.imshow(image)
    # ax.set_title("Convert to Contoure")
    
    #creat a data of pixel locations and pixel values >> shape (n_pixls * 3)>>(rowindex   colindex   pixelvalue) 
    indices = np.dstack(np.indices(image.shape[:2]))
    rows, cols, chs = indices.shape
    indices = np.reshape(indices,(rows*cols,chs))
    image = np.reshape(image,(rows*cols,1))
    feature_image = np.concatenate((indices,image), axis=1)
    #define the dbscan model and get labels the set-1 (-1 means background)
    db = DBSCAN(eps=2, min_samples=10, metric = 'euclidean',algorithm ='auto')
    db.fit(feature_image)
    labels = db.labels_
    N_Cluster = len(set(labels))-1
    print(path," has ", N_Cluster ," coin")
    #  3d plot fo showdbscan cluster
    ax = fig.add_subplot(3, 3, i+1, projection='3d')
    ax.scatter3D(feature_image[:,0],feature_image[:,1],feature_image[:,2] ,c=labels)
    ax.set_title("DBSCAN Output: "+str(N_Cluster)+" coin in image")

plt.show()