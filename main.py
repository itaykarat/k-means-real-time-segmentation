import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


class k_means():

    def __init__(self, k):
        self.cap = cv2.VideoCapture(0)  # default cap is webcam stream
        self.k = k  # define number of clusters

    def k_means_filtering(self,image):
        # read the image
        # image=cv2.imread(image)
        # convert to RGB
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('rgb',image)
        # cv2.waitKey()


        # reshape the image to a 2D array of pixels and 3 color values (RGB)
        pixel_values = image.reshape((-1, 3))
        # convert to float
        pixel_values = np.float32(pixel_values)

        # define stopping criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        compactness, labels, (centers) = cv2.kmeans(pixel_values, self.k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # convert back to 8 bit values
        centers = np.uint8(centers)

        # flatten the labels array
        labels = labels.flatten()

        # convert all pixels to the color of the centroids
        segmented_image = centers[labels]

        # reshape back to the original image dimension
        segmented_image = segmented_image.reshape(image.shape)


        # show the image
        # plt.imshow(segmented_image)
        # plt.show()

        # disable only the cluster number 2 (turn the pixel into black)
        masked_image = np.copy(image)
        # convert to the shape of a vector of pixel values
        masked_image = masked_image.reshape((-1, 3))
        # color (i.e cluster) to disable
        for cluster in range(2):
            masked_image[labels == cluster] = [0, 0, 0]

        # cluster = 1
        # masked_image[labels == 1] = [0, 0, 0]
        # masked_image[labels == 2] = [0, 0, 0]
        # masked_image[labels == 3] = [0, 0, 0]
        # masked_image[labels == 4] = [0, 0, 0]


        # convert back to original shape
        masked_image = masked_image.reshape(image.shape)
        # show the image

        # plt.imshow(masked_image)
        #
        # plt.show()
        return segmented_image,masked_image

image=cv2.imread('assests/IMG_0209.JPG')
image = cv2.resize(image, (600, 600))
# cv2.imshow('try',image)
# cv2.waitKey()
test= k_means(3)
test.k_means_filtering(image)