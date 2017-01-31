class LinearRegression:
    """Linear regression model implementation from scratch.
    Linear regression model is a very common regression model
    which the output variable is numeric and continuous.
    Linear regression model represents the relationship between
    output variable and input variables where the relationship 
    is linear.
    
    Gradient descent is the optimisation algorithm to minimise
    the cost function which is the residual sum of square (RSS).
    """
    
    def __init__(self, learning_rate=0.01, max_iter=100):
        """Initialisation with model parameters"""
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.rss_list = []
        
    
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
        # add constant 1 as the first column of input variable
        ones = np.ones(X_test.shape[0])[:, None]
        X_new = np.hstack((ones, X_test))
        
        return np.dot(X_new, self.W)[:, None]
    
    def train(self, X, y):
        """
        """
        # initialise the weights with all zero
        self.X = X
        self.y = y
        self.W = np.zeros(X.shape[1]+1)
        
        for _ in range(self.max_iter):
            # store the rss
            self.rss_list.append(self._cal_rss(self.X, self.y))
            
            # calculate the error for a prediction
            delta = self.predict(self.X) - self.y
            delta = self.y - self.predict(self.X)
#             print delta
            
            # calculate the partial derivates
            ones = np.ones(self.X.shape[0])[:, None]
            X_add_ones = np.hstack((ones, self.X))
            derivatives = np.dot(X_add_ones.T, delta)
            
            # update the weights
            self.W = self.W - self.learning_rate * derivatives.reshape(self.W.shape)
#             print self.W
    
        
    def _cal_rss(self, X, y):
        """Calculate the RSS with the weights learned
        by the model
        """
        y_predicted = self.predict(X)
        return np.sum((y - y_predicted) ** 2)
