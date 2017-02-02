import numpy as np

class KNNClassifier:
    """K Nearest Neighbours classifier implementation from scratch.
       KNN is a lazy algorithm which isn't required to compute the learning
       parameters unlike most of the machine learning algorithms. It predicts
       the instance in the test set based on the closest neighbours in the
       training set measured by the given distance metric.

       KNN performs well in the cases that no linear decision boundaries exist
       in the feature space. However, it is computationally expensive because
       it needs to calculate the distance to all the training instances for
       each prediction.
    """

    def __init__(self, k=5, metric='euclidean'):
        """
        Parameters:
        -----------
            k: specify the number of closest neighbours (default = 5)
            metric: distance metric (default = 'euclidean')
                The following metrics are available:
                    euclidean: Euclidean distance
                    manhattan: Manhattan distance
                    consine:   Cosine distance
        """
        self.k = k
        if not metric in ['euclidean', 'manhattan', 'cosine']:
            raise ValueError('ill defined metric')
        else:
            self.metric = metric

    def train(self, X, y):
        """Train the k Nearest Neighbours classifier

        Parameters:
        -----------
            X: feature array for training samples
            y: labels for training samples
        """
        self.X = X
        self.y = y

    def predict(self, X_test):
        """Make the prediction for the test set

        Parameters:
        -----------
            X_test: feature array for test set
        Return:
        -------
            An array of the predicted classes
        """
        num_instances = X_test.shape[0]
        res = np.empty(num_instances, dtype=self.y.dtype)
        for i in range(num_instances):
            res[i] = self._predict_one_instance(X_test[i, :])
        return res


    def _predict_one_instance(self, x_test):
        if self.metric == 'euclidean':
            # compute the index of the k closest neighbours
            k_indexes = np.argsort(self._dis_euclidean(self.X, x_test))[:self.k]
        elif self.metric == 'manhattan':
            # compute the index of the k closest neighbours
            k_indexes = np.argsort(self._dis_euclidean(self.X, x_test))[:self.k]

            # get the majority vote of kNN
            return np.unique(self.y[k_indexes], return_counts=True)[0][0]

    @staticmethod
    def _dis_euclidean(data_points, target):
        """Calculate the Euclidean distance between the target
           and the training instances
        """
        # axis=1 is row-wise operation
        return np.sqrt(np.sum((data_points - target) ** 2, axis=1))

    @staticmethod
    def _dis_manhattan(data_points, target):
        """Calculate the Manhattan distance between the target
           and the training instances
        """
        # axis=1 is row-wise operation
        return np.sum((data_points - target), axis=1)

    @staticmethod
    def _dis_cosine(data_points, target):
        """Calculate the Euclidean distance between the target
           and the training instances
        """
        # axis=1 is row-wise operation
        return np.sum((data_points - target), axis=1)
