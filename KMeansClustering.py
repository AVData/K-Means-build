import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np


class Kmeans:
    def __init__(self, k=2, tollerance=0.01, max_iter=300):
        self.k = k
        self.tollerance = tollerance
        self.max_iter = max_iter

    def fit(self, data):

        # creating a dictionary to contain the initial kmeans using the
        # first two data points.  The following for loop does the assigning of
        # the kmeans.
        # kmean will always contain a 0 and 1 key, but the value will change
        # based on the clusters when they are itterated through.
        self.kmeans = {}
        for i in range(self.k):
            self.kmeans[i] = data[i]

        # clusters contains the kmeans (keys) and the featureset
        # which are the values; cluster changes everytime kmean value
        # chenges.
        for i in range(3):
            self.clusters = {}

            for i in range(self.k):
                self.clusters[i] = []

            # following for loop creates a list of distances between data
            # points and kmeans given the number of centriods.  The distance
            # is calculate dusing the linalg.norm function
            for featureset in data:
                distance = [np.linalg.norm(featureset - self.kmeans
                            [kmean]) for kmean in self.kmeans]
                # cluster takes the index values and replaces
                cluster = distance.index(min(distance))
                self.clusters[cluster].append(featureset)

            # using this to compare the change in kmean change, to apply
            # tollerance value arguement in function.
            prev_kmeans = dict(self.kmeans)

            # following for loop takes classfications array and determines the
            # mean of all of the values to pick a kmean.
            # to see it in action alternate the commenting and look at the
            # resulting graph.
            for cluster in self.clusters:
                self.kmeans[cluster] = \
                    np.average(self.clusters[cluster], axis=0)
            # innocent until proven guilty
            optimized = True

            # for loop that lookas at every kmean and compares it to it's
            # previous kmean to see the change using the rate of change
            # formula; and if that rate of change remains greater than the
            # tollerancelerance argument (0.001), then the optimize will equal
            # False, and the previous loop will continue.  Incorporating a
            # break if statement to break the for loop in case it movement
            # never reaches tollerancelerance threashold
            for c in self.kmeans:
                original_kmean = prev_kmeans[c]
                current_kmean = self.kmeans[c]
                if np.sum((current_kmean - original_kmean) /
                          original_kmean*100.0) > self.tollerance:
                    print(np.sum((current_kmean - original_kmean) /
                                 original_kmean*100.0))
                    optimized = False

                if optimized:
                    break

    def predict(self, data):
        '''
        Similar to cluster previous for loop used for featureset, the
        only difference is that instead of using featureset we are using the
        actuall data being imported, and return the cluster.
        '''
        distances = [np.linalg.norm(data - self.kmeans[kmean]) for
                     kmean in self.kmeans]
        cluster = distances.index(min(distances))
        return cluster
