class LogisticRegression:
    """Logistic regression model implementation from scratch.
    Logistic regression model is a simple classification model
    which the output variable is discrete and categorical. 
    By using "sigmoid" function, Logistic regression "squashes"
    the output variables between 0 and 1 to represent the 
    probability of predicting possitive value.
    
    Gradient descent is the optimisation algorithm to minimise
    the cost function which is the log loss function.
    """
    
    def __init__(self, learning_rate=0.01, max_iter=100, positive_label=None, threshold=0.5):
        """Initialisation with model parameters
        
        Parameters:
        -----------
            learning_rate: learning rate for gradient descent 
                algorithm (default = 0.01)
            max_iter: maximum iteration to run gradient
                descent algorithm (default = 100)
            positive_label: the positive label for output
                variable (default = None)
            threshold: threshold to define the probability of 
                positive or negative predictions (default = 0.5)            
        """
        
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.positive_label = positive_label
        self.threshold = threshold

        self.logloss_list = []

    def train(self, X, y):
        """Train the Logistic Regression Model with Gradient
        descent algorithm to learn the model weights
        
        Parameters:
        -----------
            X: feature array for training samples
            y: labels for training samples
        """
        self.X = X
        self.y = y
        self.labels = np.sort(np.unique(y))
        # set the positive label
        if not self.positive_label:
            self.positive_label = self.labels[0]
        self.neg_label = self.labels[self.labels != self.positive_label][0]

        # initialise the weights with all zeros
        # self.W = np.zeros(X.shape[1]+1)
        # initialise the weights with small numbers 
        self.W = np.random.randn(X.shape[1]+1) / np.sqrt(X.shape[1]+1)
        
        # transform output variable to [0, 1] if required
        self.transformed_y = self._convert_labels()
        
        for _ in range(self.max_iter):
            # store the rss for each iteration
#             self.rss_list.append(self._cal_rss(self.X, self.y))
            
            # calculate the error for a prediction
            delta = self.predict_prob(self.X) - self.transformed_y
            
            # calculate the partial derivates
            ones = np.ones(self.X.shape[0])[:, None]
            X_add_ones = np.hstack((ones, self.X))
            derivatives = np.dot(X_add_ones.T, delta)
            
            # update the weights
            self.W = self.W - self.learning_rate * derivatives.reshape(self.W.shape)
        

    def predict_prob(self, X_test):
        """Predict the probability for being positive of output
        variable with the weights learned by the model
        
        Parameters:
        -----------
        X_test: input variables (feature matrix) as a numpy array
        
        Return:
        -------
        Predicted probability of output variable
        """
        # add constant 1 as the first column of input variable
        ones = np.ones(X_test.shape[0])[:, None]
        X_new = np.hstack((ones, X_test))
        
        h_X = np.dot(X_new, self.W)
        
        # get the result from sigmoid function
        y_predict_prob = self._sigmoid(h_X)
        
        return y_predict_prob.reshape(y_predict_prob.shape[0])

    def _predict(self, X_test):
        """Predict the output variable in [0, 1] with the weights
        learned by the model
        
        Parameters:
        -----------
        X_test: input variables (feature matrix) as a numpy array
        
        Return:
        -------
        Predicted output variable in [0, 1]
        """
        y_predict_prob = self.predict_prob(X_test)
        ones = np.ones(y_predict_prob.shape)
        zeros = np.zeros(y_predict_prob.shape)
        
        return np.where(y_predict_prob > self.threshold, ones, zeros)
        
    def predict(self, X_test):
        """Predict the output variable with the weights learned
        by the model
        
        Parameters:
        -----------
        X_test: input variables (feature matrix) as a numpy array
        
        Return:
        -------
        Predicted output variable
        """
        y_predict_prob = self.predict_prob(X_test)
        positives = np.repeat(self.positive_label, y_predict_prob.shape)
        negatives = np.repeat(self.neg_label, y_predict_prob.shape)
        
        return np.where(y_predict_prob > self.threshold, positives, negatives)
        
    @staticmethod
    def _sigmoid(x):
        """Sigmoid function
        """
        result = 1 / (1 + np.exp(-x))
        result = result.reshape(x.shape[0])
#         print "@sigmoid", result.shape
        return result

    def _convert_labels(self):
        """Convert the output variables to [0, 1] if not
        """
        
        if set(np.unique(self.y)) != set([0, 1]):
            ones = np.ones(self.y.shape)
            zeros = np.zeros(self.y.shape)
            return np.where(self.y==self.positive_label, ones, zeros)
        else:
            return self.y
