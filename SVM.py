from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from matplotlib import pyplot as plt 
fh = open("/home/sai/ML3/scop_motif.data", "r")

k=""
for i in range (0,500):
    s=fh.readline()
    g=s.split(',')[1]
    f=g.replace("rest", "-1").replace("a.1.1.2", "1")
    k+=f
fl=open("/home/sai/ML3/motif.data", "w")
fl.write(k)    
#end of pre processing
X, y = load_svmlight_file("/home/sai/ML3/motif.data")
d_degree=[1,2,3,4,5,6,7]
c_C=[0.001,0.01,0.1,1,10,100,1000]
acc=np.zeros(shape=(7,7))
for d in range(len(d_degree)):
    for c in range(len(c_C)):
        classifier = svm.SVC(kernel='poly',degree=d_degree[d],coef0=1,gamma=1, C=c_C[c])
        cv = cross_validation.StratifiedKFold(y, 5, shuffle=True,random_state=0)
        acc[d][c]=np.mean(cross_validation.cross_val_score(classifier, X, y, cv=5, scoring='roc_auc'))
n=np.asarray(acc)

#differing plots for different degree
f=plt.figure()
for i in range(len(d_degree)):
    plt.xscale('log')
    #plt.ylim(0.98,1.00001)
    plt.plot(c_C,n[i,:],'-o')
    plt.xlabel('different values of C -->')
    plt.ylabel('accuracy -->')
    plt.legend(d_degree)
plt.show()
        #C is soft
h=plt.figure()                        
#differing C
for i in range(len(c_C)):
    #plt.xscale('log')
    #plt.ylim(0.45,1.05)
    plt.plot(d_degree,n[:,i],'-o')
    plt.xlabel('different values of degree  -->')
    plt.ylabel('accuracy -->')
    plt.legend(c_C)
plt.show()
