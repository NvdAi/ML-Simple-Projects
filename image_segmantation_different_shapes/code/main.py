import numpy as np
import cv2
import glob
import os 
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from wand.image import Image
from skimage.util import random_noise


# import and load data and also rename all of samples (images) to normal name (img_0) to (img_n)
data_path = "../new_data"
dirc_images = glob.glob(os.path.join(data_path,"*"))
# for number, filename in enumerate(dirc_images):
    # try:
        # os.rename(filename,os.path.join(data_path,"img_{0}.jpg".format(number)))
    # except OSError as e:
        # print("Something happened:", e)
# dirc_images = glob.glob(os.path.join(data_path,"*"))

# make figure for display
fig = plt.figure(figsize =(25,35))
fig.subplots_adjust(hspace = 0.5,wspace=0.1)
fig.suptitle('Coin Counter by Clustering(DBSCAN)')
dirc_images = [ "../new_data/img_0.jpg" ,"../new_data/img_1.jpg"]

# with Image(filename ="../new_data/img_2.jpg") as img:
#     # Generate noise image using spread() function
#     img.noise("gaussian", attenuate = 0.1)
#     img.save(filename ="../new_data/img_noise.jpg")

# main of code
for i in range(1):
    noise_rng = [(-0.1,0.1,0.01),(-0.3,0.3,0.02),(-0.5,0.5,0.03)]
    for j,item in enumerate(noise_rng):
        path = dirc_images[i]
        img = cv2.imread(path)
        lower,upeer,amount = item
            
        # noise_img = random_noise(img, mode='s&p',amount=amount)
        # noise_img = np.array(255*noise_img, dtype = 'uint8')

        gauss = np.random.normal(lower,upeer,img.size)
        gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
        noise_img = cv2.add(img,gauss)
        
        ax = fig.add_subplot(2, 3, 1+j)
        ax.imshow(noise_img)
        ax.set_title("guassian noise range "+str(lower)+str(upeer))

        # resize 
        img = cv2.resize(noise_img, (0,0), fx=0.05, fy=0.05)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #creat a data of pixel locations and pixel values >> shape (n_pixls * 3)>>(rowindex   colindex   pixelvalue) 
        indices = np.dstack(np.indices(img.shape[:2]))
        rows, cols, chs = indices.shape
        indices = np.reshape(indices,(rows*cols,chs))
        image = np.reshape(img,(rows*cols,3))
        image = np.concatenate((indices,image), axis=1)
        #define the dbscan model and get labels the set-1 (-1 means background)
        db = DBSCAN(eps=2, min_samples=10, metric = 'euclidean',algorithm ='auto')
        db.fit(image)
        labels = db.labels_
        N_Cluster = len(set(labels))-2
        print(path," has ", N_Cluster ," coin",set(labels))
        #  3d plot fo showdbscan cluster
        # pca = PCA(n_components=2)
        # pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
        # Xt = pipe.fit_transform(image)

        tsne = TSNE(n_components=2, random_state=123)
        z = tsne.fit_transform(image)
        Xt=z
        ax = fig.add_subplot(2, 3, 4+j)
        # sns.scatterplot(x=Xt[:,0], y=Xt[:,1])
        ax.scatter(Xt[:,0], Xt[:,1], c=labels)
        ax.set_title("DBSCAN Output: "+str(N_Cluster)+" cluster in image")

plt.show()