import numpy as np
# import math

class MultinomialNBClassifier:
    """Multinomial Naive Bayes classifier implementation from scratch.
       Naive Bayes is a generative model which is calcuating the probabilities
       for each class of the target and the probabilities of each feature
       given a specific class. It is based on the Bayes Therom and the
       assumption that each feature is conditionally independent.

       Even though the feature independency rarely holds true, Naive Bayes model
       performs very well in document classification, sentiment analysis, spam
       filter, etc. Another nice thing of Naive Bayes is it only needs to
       calculate the required probabilities which makes it very efficient, unlike
       most of the algorithms with iterative computational process

       Ref:
       ----
       http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes
    """

    def __init__(self, alpha=1):
        """Initialisation
        Parameters:
        ----------
            alpha: smoothing parameter (default = 1)
        """
        self.alpha = alpha
#         self.X = None
#         self.y = None
#         self.classes = None
#         self.posteriori = None
#         self.feat_prob = None

    def train(self, X, y):
        """Train the Multinomial Naive Bayes classifier

        Parameters:
        -----------
            X: feature array for training samples
            y: labels for training samples
        """
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self._cal_posteriori()
        self._cal_feat_prob()


    def _cal_posteriori(self):
        """compute posterior"""
        self.posteriori = np.array([self.y[self.y==k].size/float(self.y.size) for k in self.classes])
#         return posterior

    def _cal_feat_prob(self):
        """compute the probabilities for each feature given a class.
        It is assumed that the features are with multinomial distribution
        """
        num_feats = self.X.shape[1]
        self.feat_prob = np.zeros((self.classes.shape[0], num_feats))
#         alpha = 1

        for i in range(self.classes.shape[0]):
            for feat_ix in range(num_feats):
                numerator = self.X[self.y==self.classes[i], feat_ix].sum() + self.alpha
                denominator =  self.X[self.y==self.classes[i], :].sum() + self.alpha * num_feats
                self.feat_prob[i, feat_ix] = numerator / float(denominator)


    def predict(self, X_test):
        """Make the prediction with the trained model and the test set
        The probability is calculated as P(y_hat) = argmax[P(y) * P(X_i | y)]

        Parameters:
        -----------
            X_test: feature array for test set
        Return:
        -------
            An array of the predicted classes
        """
        return self.classes[np.argmax(np.dot(X_test, self.feat_prob.T) * self.posteriori, axis=1)]


    def predict_prob(self, X_test):
        """Make the prediction with the trained model and the test set,
        return the probabily for each class
        The probability is calculated as P(y_hat) = argmax[P(y) * P(X_i | y)]

        Parameters:
        -----------
            X_test: feature array for test set (m samples x n features)
        Return:
        -------
            An matrix of the predicted probabilities (m samples x k classes)
        """
        res = np.dot(X_test, self.feat_prob.T) * self.posteriori
        return res / res.sum(axis=1)[:, None]


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
