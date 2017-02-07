# Machine Learning Algorithms from Scratch
The best way to understand an algorithm is to implement it from scratch. Here I use NumPy and SciPy to implement some of the most common machine learning algorithms since all the algorithms involve linear algebra (vector, matrix) computation.

## Linear Regression
Linear regression model is a very common regression model which the output variable is numeric and continuous. Linear regression model represents the relationship between output variable and input variables where the relationship is linear.

Gradient descent is the most common optimisation algorithm to learn the model parameters (weights) with the goal to minimise the cost function which is the residual sum of square (RSS).

## Logistic Regression
Logistic regression model is a simple classification model which the output variable is discrete and categorical. By using "sigmoid" function, Logistic regression "squashes" the output variables between 0 and 1 to represent the probability of predicting possitive value.     
Gradient descent is the optimisation algorithm to minimise the cost function which is the negtive log loss function.

## Navie Bayes
Naive Bayes is a generative model which is calculating the probabilities for each class of the target and the probabilities of each feature given a specific class. It is based on the Bayes Therom and the assumption that each feature is conditionally independent (that is why it is Naive).

Even though the feature independency rarely holds true, Naive Bayes model performs very well in document classification, sentiment analysis, spam filter, etc. Another nice thing of Naive Bayes is it only needs to calculate the required probabilities which makes it very efficient, unlike most of the algorithms with iterative computational process

Based on the distribution of features, there are a few variance of Naive Bayes:
* Multinomial Naive Bayes
* Gaussian Naive Bayes
* Poisson Navie Bayes

## K Nearest Neighbours
KNN is a non-parametric lazy algorithm which isn't required to compute the learning parameters unlike most of the machine learning algorithms. It predicts the instance in the test set based on the closest neighbours in the training set measured by the given distance metric. It is kind of instance based or memory based model.

KNN performs well in the cases that no linear decision boundaries exist in the feature space. However, it is computationally expensive because it needs to calculate the distance to all the training instances for each prediction especially for large dataset.
