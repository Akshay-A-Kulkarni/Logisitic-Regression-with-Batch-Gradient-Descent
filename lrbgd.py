class LogisticRegressionWithGD:

    def __init__(self, lr=0.01, num_iters=10000, fit_intercept=True, printLoss=False):
        self.lr = lr
        self.num_iters = num_iters
        self.fit_intercept = fit_intercept
        self.printLoss = printLoss
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))    
    
    def __cross_entropy_loss(self, h, y):
        h = h + 1e-9
        return (-((y * np.log(h)) - ((1 - y) * np.log(1 - h)))).mean()
    
    def fit(self, X, y):
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)  
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iters):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y))/y.shape
            self.theta -= self.lr * gradient
                
        if(self.printLoss == True):
            z = np.matmul(X, self.theta)
            h = self.__sigmoid(z)
            print(f'Cross Entropy Loss for 𝛼 = {self.lr}, iters = {self.num_iters} : {self.__cross_entropy_loss(h, y)} \t')
    
    def predict_prob(self, X):
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return [1 if i >= threshold else 0 for i in self.predict_prob(X)]
