import scipy
import matplotlib.image as img
from PIL import Image, ImageStat
import numpy as np
import copy
from tempfile import TemporaryFile
from scipy.spatial.distance import cdist


im = Image.open("hw3_img.jpg")
img_width, img_height = im.size
px = im.load()

img_arr = []
for x in range(0, img_width):
    for y in range(0, img_height):
        pi = px[x, y]
        img_arr.append(list(pi))
img_arr = np.array(img_arr)

img_arr2 = img_arr/255

class Kmeans:
    def __init__(self, K, width, height, img_arr_value, iteration):
        self.K = K
        self.I = np.eye(K)
        self.img_width = width
        self.img_height = height
        self.centroids = []
        self.old_centroids = []
        self.px = img_arr_value
        self.size, self.dim = img_arr_value.shape
        self.iteration = iteration
        self.cov = np.zeros((K, self.dim , self.dim))
        self.mixcoef = [0 for i in range(K)]
        self.group_cluster = dict()

        for i in range(self.K):
            self.group_cluster.update({i:list()})

        
    
    def runs(self):
        

        self.centroids = self.px[np.random.choice(self.size, self.K, replace=False)]
        #print(self.centroids )
        
        for i in range(self.iteration):
            
            #print("Iteration :" + str(i))
            
            self.old_centroids = self.centroids  
            
            Distance = cdist(self.px, self.centroids)
            cluster_id = np.argmin(Distance, axis=1)
            cluster_id =  self.I[cluster_id]
            
            self.centroids = np.sum(self.px[:, None, :] * cluster_id[:, :, None], axis=0) / np.sum(cluster_id, axis=0)[:, None]
            if np.allclose(self.old_centroids, self.centroids) == True :
                break


        Distance = cdist(self.px, self.centroids)
        cluster_id = np.argmin(Distance, axis=1)
        
        for idex in range(len(cluster_id)):
            self.group_cluster[cluster_id[idex]].append(self.px[idex])
       
        # compute covariance

        for idex in self.group_cluster.keys():
            mean = self.centroids[idex]
            values = np.array([self.group_cluster[idex]]).reshape( (len(self.group_cluster[idex]),  self.dim))
            values = values - mean
            self.cov[idex] = np.dot(values.transpose(), values)/len(values)

        self.mixCoefficent()


    def kmeanDraw(self):
        cp_imgarr = copy.deepcopy(self.px)
        img = Image.new('RGB', (self.img_width, self.img_height), "white")
        p = img.load()
        
        Distance = cdist(self.px, self.centroids)
        cluster_id = np.argmin(Distance, axis=1) 
        
        for i in range(len(cluster_id)):  
            RGB_value = self.centroids[cluster_id[i]]*255
            cp_imgarr[i] = RGB_value
        
        i = 0
        for x in range(self.img_width):
            for y in range(self.img_height):
                p[x, y] = tuple(cp_imgarr[i].astype(int))
                i+=1 
        img.show()
        img.save('Kmean'+str(self.K)+'_poblem3.jpg')

    def mixCoefficent(self):

        Distance = cdist(self.px, self.centroids)
        cluster_id = np.argmin(Distance, axis=1)
        for i in cluster_id:
            self.mixcoef[i] +=1
        for i in range(self.K):
            self.mixcoef[i] = self.mixcoef[i]/len(self.px)

'''kmodel = Kmeans(20, img_width, img_height, img_arr2, 20)
kmodel.runs()
kmodel.kmeanDraw()'''