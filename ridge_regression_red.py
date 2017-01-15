import random
import numpy as np
from matplotlib import pyplot as plt

class Reg :
    #def __init__(self): #lamda=0.001) :
        #self.lamda=lamda
        #self.w=0
    
        
    def regularize(self,X_train,y_train,lamda):
       self.lamda=lamda 
       z= np.dot(X_train.T,X_train).shape[0]
      # print 'Z '+str(z)

       m=np.dot(X_train.T,X_train)+self.lamda*np.identity(z, dtype=None)
       self.w=np.dot((np.linalg.inv(m)),np.dot(X_train.T,y_train))   
      # print self.w,self.w.shape
    
    def rms(self,X,y):
        self.h=y-np.dot(X,self.w)
        #for i in range(len(y)):
            
        
        self.h=np.dot(self.h,self.h)
        #self.s=0
        #for i in range(len(self.h)):
           # self.s=self.s+self.h[i]
        N=len(y)
        self.rmse=np.sqrt(self.h/N)
        return self.rmse
        
    def mad(self,X,y):
         self.e=abs(y-np.dot(X,self.w))
         self.esum=0
         for i in range(len(self.e)):
            self.esum+=self.e[i]
         N=len(y)
         return self.esum/N  
         
          
    def rec_curve(self,X,y):
         e=abs(y-np.dot(X,self.w))
         #print 'Errors '+str(e)
         e.sort()
         #print 'sort '+str(e)
         E_prev=0
         correct=0
         f3=plt.figure()
         mx=[]
         my=[]
         for i in e:
                
                if (i>E_prev):
                    #print 't'
                    z=(float)(correct)/(len(e))
                    mx.append(E_prev)
                    my.append(z)
                    #plt.plot(E_prev,z,'-r-')
                    E_prev=i
                correct=correct+1    
         z=(float)(correct)/(len(e))
         mx.append(e[-1])
         my.append(z)
         plt.plot(mx,my)
         plt.grid()
         plt.ylim(0,1.02)
         #plt.plot(e[-1],z,'-r')
         plt.xlabel('tolerance -->')
         plt.ylabel('accuracy -->')
         plt.show()
        
            
    
if __name__=='__main__' :
    
    data1=np.genfromtxt("J:/courses/MachineLearning/ML2/winequality-red.csv",delimiter=";", comments="\"")
    #print data,data.shape
    #standardization
    for i in range(data1.shape[1]-1): 
        f=data1[:,i].mean()
        std_dev=data1[:,i].std()
        data1[:,i]=(data1[:,i]-f)/std_dev
    #print data,data.shape
    z = np.ones((data1.shape[0],1),dtype=data1.dtype)  
    data=np.append(z,data1,axis=1)
#    np.shuffle(data)#shuffke
    Xt=[]
    yt=[]
    X_est=[]
    y_est=[]
    for i in range(len(data)):
        if(i%3==1):#test
            X_est.append(data[i,0:12])
            y_est.append(data[i,12])
        else:
            Xt.append(data[i,0:12])
            yt.append(data[i,12])     

    X_train=np.asarray(Xt)
    y_train=np.asarray(yt)
    X_test=np.asarray(X_est)
    y_test=np.asarray(y_est)
    
    #X_test = np.ones((X_traino.shape[0],1),dtype=X_traino.dtype) 
    print 'X_train'+str(X_train.shape)
    print 'y_train'+str(y_train.shape)
    print 'data '+str(data.shape)   
    print 'X_test'+str(X_test.shape)
    print 'y_test'+str(y_test.shape)
    p=Reg()
    L=[0.001,0.01, 0.1, 1, 10, 100, 1000, 10000,100000,1000000,10000000]
    #rmse_train=[]
    rmse_test=[]
    mad_test=[]
    for lamda in L:
        p.regularize(X_train,y_train,lamda)
        rmse_test.append(p.rms(X_test,y_test))
        mad_test.append(p.mad(X_test,y_test))
        
    f1=plt.figure()
    plt.plot(L,rmse_test,'-ro', label=u'RMSE')
    #plt.ylim(0,2)
    plt.xscale('log')
    plt.xlabel('lamda-->')

    plt.plot(L,mad_test, '-bo', label=u'MAD')
    #plt.ylim(0.4,1.2)
    plt.xscale('log')
    #plt.xscale('log')
    plt.xlabel('lamda-->')
    plt.ylabel('RSME/MAD  -->')
    plt.legend(loc='lower right')
    #plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)

    #plt.legend()
    plt.show()
    p.regularize(X_train,y_train,10)
    p.rec_curve(X_test,y_test)
 
    pearson_cof=[] # from 1 to 11 as 0 is bias
    w=[]
    y_train_std=y_train.std()
    f2=plt.figure()
    for i in range(1,X_train.shape[1]):
        a=np.corrcoef(X_train[:,i],y_train)[0,1]
        pearson_cof.append(a)
        #plt.plot(p.w[i],a,'-bo')
    #f2=plt.figure()
    plt.scatter(p.w[1:],pearson_cof)    
    plt.xlabel('weight-->')
    plt.ylabel('pearson coefficient  -->')
    names='nan;fixed acidity;volatile acidity;citric acid;residual sugar;chlorides;free sulfur dioxide;total sulfurdioxide;density;pH;sulphates;alcohol'
    fearures_names=names.split(";")
    decen=[]
    w_valu=[]
    plt.show()
    n=11
    no_features=[]
    rmse_f=[]
    mad_f=[]
    #part 2 
    #X_train=np.delete(X_train,np.s_[0],axis=1)#remove bias
    p.regularize(X_train,y_train,10)# lamda 10 has least RMSE
    print p.w
    s=p.w.copy()
    d=X_train.copy()
    no_features.append(n)
    rmse_f.append(p.rms(X_train,y_train))
    mad_f.append(p.mad(X_test,y_test))
    #absI=np.argmin(np.absolute(p.w))
    #print X_train.shape
    for i in range(10):
        print X_train.shape
        absI=np.argmin(np.absolute(p.w[1:]))+1
        decen.append(fearures_names.pop(absI))
        
        print absI
        n-=1
        w_valu.append(np.absolute(p.w[absI]))
        X_train=np.delete(X_train,np.s_[absI],axis=1).copy()
        p.regularize(X_train,y_train,10)
        no_features.append(n)
        rmse_f.append(p.rms(X_train,y_train))
        mad_f.append(p.mad(X_train,y_train))
        #absI=np.argmin(np.absolute(p.w))            
    w_valu.append(np.absolute(p.w[1]))
    decen.append(fearures_names.pop(absI))
    f5=plt.figure()
    plt.plot(no_features,rmse_f,'-ro', label=u'RMSE')    
    plt.plot(no_features,mad_f,'-bo', label=u'MAD')
    plt.ylabel('RMSE/MAD -->')
    plt.xlabel('number of features -->')
    plt.legend(loc='upper right')
   # plt.ylim(0,0.75)
    plt.show()            
    