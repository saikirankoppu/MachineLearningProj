import numpy as np
from matplotlib import pyplot as plt

class Perceptron :

    """An implementation of the perceptron algorithm.
    Note that this implementation does not include a bias term"""

    def __init__(self, max_iterations=100, learning_rate=0.2) :

        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y) :

        self.w = np.zeros(len(X[0]))
        converged = False
        iterations = 0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)-1500) :
                if y[i] * self.discriminant(X[i]) <= 0 :#ywx<=0
                    self.w = self.w + y[i] * self.learning_rate * X[i]
                    converged = False
            iterations += 1
            #plot_data(X, y, self.w)
            
        self.converged = converged
        if converged :
            print 'converged in %d iterations ' % iterations
        else:
            print 'not converged'

    def discriminant(self, x) :
        return np.dot(self.w, x)
            
    def predict(self, X) :

        scores = np.dot(self.w, X)
        return np.sign(scores)


def accuracy(X,y,w):
        accuracy=0
	for i in range(len(X)-1500,len(X)):
		if(np.dot(w, X[i])>0 and y[i]==1):
			accuracy=accuracy+1
			a.append(X[i])
		if(np.dot(w, X[i])<0 and y[i]==-1):
			accuracy=accuracy+1
			a.append(X[i])
	print 'accuracy (Aout) ' +str(accuracy/1500.0)
	print 'Error (Eout) '+str(1-(accuracy/1500.0))		

def accuracy_in(X,y,w):
        accuracy=0
	for i in range(len(X)-1500):
		if(np.dot(w, X[i])>0 and y[i]==1):
			accuracy=accuracy+1
		if(np.dot(w, X[i])<0 and y[i]==-1):
			accuracy=accuracy+1
	print 'accuracy (Ain) ' +str(accuracy/((len(X)-1500)*1.0))
	print 'Error (Ein) '+str(1-accuracy/((len(X)-1500)*1.0))			






if __name__=='__main__' :
    a=[]
    y=np.genfromtxt("/home/sai/Downloads/ML1/gisette/gisette_train.labels",delimiter=" ", comments="#")
    print y,y.shape
    X_old=X=np.genfromtxt("/home/sai/Downloads/ML1/gisette/gisette_train.data",delimiter=" ", comments="#")
    print X_old,X_old.shape
    z = np.ones((X_old.shape[0],1),dtype=X_old.dtype)
    X=np.append(z,X_old,axis=1)
    
    #X.append(np.ones(len(X[:,1])))
    print X,X.shape 
	   
    #print 'accuracy '+accuracy	
    
    #X,y,w = generate_separable_data(20)
    p = Perceptron()
    p.fit(X,y)
    w=p.w
    accuracy_in(X,y,p.w)
    accuracy(X,y,p.w)
    #print 'accuracy '+accuracy	