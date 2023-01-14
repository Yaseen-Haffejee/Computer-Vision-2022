import numpy as np
from scipy.stats import multivariate_normal
class GMM:
    def __init__(self, n_components,max_iterations,tolerance):
        self.components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def initialize_parameters(self, X):
        np.random.seed(42)
        # returns the (r,c) value of the numpy array of X
        self.shape = X.shape 
        # n has the number of rows while m has the number of columns of dataset X
        self.n, self.m = self.shape 
        

        # initial weights given to each cluster are stored in phi or P(Ci=j)
        self.lambda_k = np.full(shape=self.components, fill_value=1/self.components) 

        # probability of each point belonging to each component
        self.responsibilities = np.random.uniform(size=(self.n, self.components))
        self.responsibilities /= self.responsibilities.sum(axis=1)[:, np.newaxis]


        # initial value of mean of k Gaussians
        chosen = np.random.choice(self.n, self.components, replace = False)
        self.means = X[chosen]

        # Covariance matrix for each component
        self.covariance = []
        for i in range(self.components):
            A =  np.random.random((self.m,self.m))
            A*= A.T
            A += self.m*np.eye(self.m)*456432313
            self.covariance.append(A)

    # predicts probability of each data point wrt each cluster
    def predict_probability(self, X,predict= False):
        # create matrix of size rows x components
        # probability of each point belonging to either component
 
        if(predict):
            likelihood = np.zeros( (X.shape[0], self.components) )
        else:
            likelihood = np.zeros( (self.n, self.components) )

        for i in range(self.components):
            likelihood[:,i] = multivariate_normal(mean=self.means[i],cov=self.covariance[i],seed=42).pdf(X)

        numerator = likelihood * self.lambda_k
        if(predict):
            return numerator
        else:
            denominator = numerator.sum(axis=1,keepdims=1)
            weights = numerator / denominator
        return weights

    def _compute_log_likelihood(self):
        log_likelihood = np.sum(np.log(np.sum(self.responsibilities, axis = 1)))
        return log_likelihood

    # E-Step: update responsibilities holding means and covariance constant
    def E(self, X):
        self.responsibilities = self.predict_probability(X)
        

    # M-Step: update means and covariance holding responsibilities constant
    def M(self, X):
        # Update params of each component
        responsibility_weights = self.responsibilities.sum(axis = 0)
        # Update the probability of each Component
        self.lambda_k = responsibility_weights / X.shape[0]
        weighted_sum = np.dot(self.responsibilities.T, X)
        # Update the means of each component
        self.means = weighted_sum / responsibility_weights.reshape(-1, 1)
        for k in range(self.components):
            diff = (X - self.means[k]).T
            weighted_sum = np.dot(self.responsibilities[:, k] * diff, diff.T)
            # Update the covariance matrix
            self.covariance[k] = weighted_sum / responsibility_weights[k]
            np.fill_diagonal(self.covariance[k],np.diag(self.covariance[k]) + 1e-6)

    # Do expectation maximisation
    def fit(self, X):

        self.initialize_parameters(X)
        log_likelihood = self._compute_log_likelihood()
        self.converged = False
        self.log_likelihood_trace = [] 
        for i in range(self.max_iterations):
            self.E(X)
            self.M(X)
            log_likelihood_new = self._compute_log_likelihood()
            if abs(log_likelihood_new - log_likelihood) <= self.tolerance:
                self.converged = True
                break
  
            log_likelihood = log_likelihood_new
            self.log_likelihood_trace.append(log_likelihood)

    # Predicts which component each point lies in 
    def predict(self, X):
        weights = self.predict_probability(X,True)
        return np.argmax(weights, axis=1),weights