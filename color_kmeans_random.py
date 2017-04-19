# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)

	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
 
	# return the histogram
	return hist

def plot_colors(data):
    
    hist=data[0]
    centroids=data[1]
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster	            
        print "{0}-{1}".format(percent,color)
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX
    
    # show the color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
    

def exec_color_kmeans(image,k):
    # reshape the image to be a list of pixels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=image[image!=[0,0,0]]
    image=image.reshape((len(image)/3,3))
    
    clt = KMeans(n_clusters = k)
    clt.fit(image)    
    
    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(clt)
    
    #compute the color predominant
    hist, centroids = (list(t) for t in zip(*sorted(zip(hist, clt.cluster_centers_), reverse=True)))
    
    return [hist,centroids]
    

image = cv2.imread("../images/401.png")
data=exec_color_kmeans(image,2)
plot_colors(data)