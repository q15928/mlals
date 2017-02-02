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
