# necessary imports
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()

def centroid_histogram(clt,numLabels):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	(hist, _) = np.histogram(clt, bins = numLabels)

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
	
    # show the bar chart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
    
def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)
    
    
def exec_color_kmeans(image):
    # reshape the image to be a list of pixels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=image[image!=[0,0,0]]
    image=image.reshape((len(image)/3,3))
    
    # initialize centroids
    centroids=np.array([[254,250,242],[198,158,97],[246,223,105],[122,59,62],[243, 123, 173],[143,40,59],[71,54,65],[48,46,62]])
        
    labels=closest_centroid(image, centroids)
    unique_labels=np.unique(labels)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(labels,unique_labels)
    
    #select centroid found on clustering
    centroids_on=centroids[unique_labels]
    
    #compute the color predominant
    hist, centroids = (list(t) for t in zip(*sorted(zip(hist, centroids_on), reverse=True)))
    
    return [hist,centroids]
    
    
image = cv2.imread("../images/401.png")
data=exec_color_kmeans(image)

plot_colors(data)