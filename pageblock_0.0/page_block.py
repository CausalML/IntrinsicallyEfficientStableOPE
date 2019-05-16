import numpy as np
from scipy import io
import sys
import scipy as sp
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize, rosen, rosen_der
from sklearn import svm
import tensorflow as tf

data = np.genfromtxt("page-blocks.data", delimiter='',dtype="str")
                     

data2 = data.astype(float)
data_X_train =  data2[0:1500,0:10] 
data_Y_train =  data2[0:1500,10]-1
data_X_test =  data2[1500:,0:10] 
data_Y_test =  data2[1500:,10]-1

sample_number = data_X_test.shape[0]
class_number = 5

clf = LogisticRegression(random_state=0,solver='lbfgs',multi_class="multinomial").fit(data_X_train,data_Y_train)

#### Get PI_e
alpha_ = 0.9
args = sys.argv
print args
beta_ = np.float(args[2])
ite_number = np.int(args[3])
option = 2

def clf2(X,alpha=alpha_,beta=beta_):

    uni2 = np.random.uniform()
    ans = np.int(clf.predict(X)[0])
    prob2 = np.ones(class_number)*1.0/class_number
    prob3 = np.zeros(class_number)
    prob3[ans] = 1.0
    prob = alpha*prob3+(1.0-alpha)*prob2
    return np.random.choice(np.arange(class_number), p=prob)
    
def clf2_dens(Y,X,alpha=alpha_,beta=beta_):
    uni2 = np.random.uniform()
    ans = np.int(clf.predict(X)[0])
    prob2 = np.ones(class_number)*1.0/class_number
    prob3 = np.zeros(class_number)
    prob3[ans] = 1.0
    prob = alpha*prob3+(1-alpha)*prob2
    return prob[Y]

def clf4(X,alpha=alpha_,beta=beta_):

    uni2 = np.random.uniform()
    ans = np.int(clf.predict(X)[0])
    ###prob2 = np.array([i+1.0 for i in range(class_number)])
    ###prob2 = prob2/np.sum(prob2)
    prob2 = np.ones(class_number)*1.0/class_number
    prob3 = np.zeros(class_number)
    prob3[ans] = 1.0
    prob = beta*prob3+(1-beta)*prob2
    
    return np.random.choice(np.arange(class_number), p=prob)
    
def clf4_dens(Y,X,alpha=alpha_,beta=beta_):
    uni2 = np.random.uniform()
    ans = np.int(clf.predict(X)[0])
    ####prob2 = np.array([i+1.0 for i in range(class_number)])
    ###prob2 = prob2/np.sum(prob2)
    prob2 = np.ones(class_number)*1.0/class_number
    prob3 = np.zeros(class_number)
    prob3[ans] = 1.0
    prob = beta*prob3+(1.0-beta)*prob2
    return prob[Y]
    


class_func = [clf2, clf2, clf4]
class_dens = [clf2_dens, clf2_dens, clf4_dens]

ground_truth = []
for j in range(200):
    pred = []
    for i in range(sample_number):
        pred.append(clf2(data_X_test[i:i+1,:],0.9,0.0))
    ground_truth.append(accuracy_score(pred, data_Y_test))

    
print np.mean(ground_truth) 
truth = np.mean(ground_truth)

truth = np.mean(ground_truth)

from scipy.optimize import Bounds
sample_dim = 10

def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))




ipw_list = []
bipw_list = []
dm_list = []
dm2_list = []
dr_list  =[]
bdr_list = []
drs_list = []
mdr_list = []
mdr2_list = []
mdr3_list = []
bdrs_list = []
drss_list = []

for kkk in range(ite_number):
    check = 0
    ### Make a new data 
    print kkk
    data_behaiv_Y = []
    for i in range(sample_number):
        data_behaiv_Y.append(np.int(class_func[option](data_X_test[i:i+1,:])))
        
    #### Make a behavior policy 
    data_behaiv_Y = np.array(data_behaiv_Y)
    reward = np.ones(sample_number)
    ratio = np.ones(sample_number)
    for i in range(sample_number):
        ratio[i] = clf2_dens(data_behaiv_Y[i],data_X_test[i:i+1,:],alpha_,0.0)/class_dens[option](data_behaiv_Y[i],data_X_test[i:i+1,:])
        reward[i] = (data_behaiv_Y[i]==data_Y_test[i])
    
    for i in range(class_number):
        pesdo_X = data_X_test[data_behaiv_Y==i,:]
        pesdo_Y = reward[data_behaiv_Y==i]
        if np.sum(pesdo_Y)==0:
            check = 1
            print 1
    if check !=1: 
        #### Make a Q-function
        reg = [i for i in range(class_number)]
        reg_w = [i for i in range(class_number)]
        for i in range(class_number):
            pesdo_X = data_X_test[data_behaiv_Y==i,:]
            pesdo_Y = reward[data_behaiv_Y==i]
            reg[i] = LogisticRegression(random_state=0,solver='lbfgs').fit(pesdo_X,pesdo_Y)
            reg_w[i] = LogisticRegression(random_state=0,penalty= "l1").fit(pesdo_X,pesdo_Y)
    
    
    ###### Optimization ############       
        score = np.ones(sample_number)
        score_w = np.ones(sample_number)
        score2 = np.ones(sample_number)
        score2_w = np.ones(sample_number)
        for i in range(sample_number):
            ### DM part 
            sss = np.sum([reg[j].predict_proba(data_X_test[i:i+1,:])[0][1]*clf2_dens(j,data_X_test[i:i+1,:],alpha_,0.0) for j in range(class_number)])
            sss2 = np.sum([reg_w[j].predict_proba(data_X_test[i:i+1,:])[0][1]*clf2_dens(j,data_X_test[i:i+1,:],alpha_,0.0) for j in range(class_number)])
            ### IPW part
            ttt = reg[data_behaiv_Y[i]].predict_proba(data_X_test[i:i+1,:])[0][1]
            ttt2 = reg_w[data_behaiv_Y[i]].predict_proba(data_X_test[i:i+1,:])[0][1]
            score[i] = sss
            score_w[i] =sss2
            score2[i] = ttt
            score2_w[i] = ttt2


        ### IPW 
        ipw_list.append(np.mean(ratio*reward))

        ### BIPW 

        bipw_list.append(np.mean(ratio*reward)/np.mean(ratio))

        ###dm
        dm_list.append(np.mean(score))
        dm2_list.append(np.mean(score_w))

        ### dr 

        score3 = score+ratio*(reward-score2)
        score3 = 0.5*score+0.5*score_w+ratio*(reward-0.5*score2-0.5*score2_w)
        dr_list.append(np.mean(score3))

        ###bdr 

        score3 = score+ratio*(reward-score2)/np.mean(ratio)
        score3 = 0.5*score+0.5*score_w+ratio*(reward-0.5*score2-0.5*score2_w)/np.mean(ratio)
        bdr_list.append(np.mean(score3))

        ### drs 

        def loss(beta):
            score3 = beta[0]+score*beta[1]+score_w*beta[2]+ratio*(reward-beta[0]-score2*beta[1]-score2_w*beta[2])
            return np.var(score3)

        minmin = minimize(loss,np.array([0.0,1.0,1.0]),method='BFGS')
        beta = minmin.x
        print(beta)
        score3 = beta[0]+score*beta[1]+score_w*beta[2]+ratio*(reward-beta[0]-score2*beta[1]-score2_w*beta[2])
        drs_list.append(np.mean(score3))

        #### bdrss 
        def loss(beta):
            score3 = beta[0]+score*beta[1]+score_w*beta[2]+ratio*(reward-beta[0]-score2*beta[1]-score2_w*beta[2])
            score4 = ratio*(reward-beta[0]-score2*beta[1]-score2_w*beta[2])
            po1 = np.cov(-score3,-ratio)[0,0]
            po2 = np.cov(-score3,-ratio)[0,1]
            po3 = np.cov(-score3,-ratio)[1,1]
            return po1+2.0*np.mean(score4)*po2+np.mean(score4)**2*po3
    

    
        minmin = minimize(loss,np.array([0.0,1.0,1.0]),method='BFGS')
        beta = minmin.x
        print(beta)
        score3 = beta[0]+score*beta[1]+score_w*beta[2]+ratio*(reward-beta[0]-score2*beta[1]-score2_w*beta[2])/np.mean(ratio)
        bdrs_list.append(np.mean(score3))



        #### drss
        def loss(beta):
            score3_ = 1.0+beta[0]*(ratio-1)+(score-ratio*score2)*beta[1]+(score_w-ratio*score2_w)*beta[2]
            if np.min(score3_)<0.0:
                return(10)
            else:
                score3 = np.log(score3_)
            return -np.mean(score3)
        minmin = minimize(loss,method="L-BFGS-B",x0=np.array([0.0,0.0,0.0]))


        beta = minmin.x
        print(beta)
        score3 = 1.0/(1.0+beta[0]*(ratio-1)+(score-ratio*score2)*beta[1]+(score_w-ratio*score2_w)*beta[2])
        score4 = ratio*reward*score3

        drss_list.append(np.mean(score4)/(np.mean(score3)))
        
        ### mdr3
        """
        def loss(beta):
            beta = beta.reshape([class_number, sample_dim])
            score = np.array([ np.sum([sigmoid(np.dot(beta[j,],data_X_test[i:i+1,:][0,]))*clf2_dens(j,data_X_test[i:i+1,:],alpha_,0.0)  for j in range(class_number)] ) for i in range(sample_number)])
            score2 = np.array([np.dot(beta[np.int(data_Y_test[i:i+1]),],data_X_test[i:i+1,:][0,]) for i in range(sample_number)])
            score2 = 1.0/(1.0+np.exp(-score2))
            score3 = score+ratio*(reward-score2)
            return np.mean(score3*score3)

        minmin = minimize(loss,np.zeros(sample_dim*class_number),method='BFGS',options={'maxiter':20})
        beta = minmin.x
        beta = beta.reshape([class_number, sample_dim])
        score = np.array([ np.sum([sigmoid(np.dot(beta[j,],data_X_test[i:i+1,:][0,]))*clf2_dens(j,data_X_test[i:i+1,:],alpha_,0.0)  for j in range(class_number)] ) for i in range(sample_number)])
        score2 = np.array([np.dot(beta[np.int(data_Y_test[i:i+1]),],data_X_test[i:i+1,:][0,]) for i in range(sample_number)])
        score2 = 1.0/(1.0+np.exp(-score2))
        score3 = score+ratio*(reward-score2) 
        mdr3_list.append(np.mean(score3))
        """
        

	print(mdr3_list)                    
	np.savez("ipw_list"+str(args[1]), x=ipw_list)
	np.savez("bipw_list"+str(args[1]), x=bipw_list)
	np.savez("dm_list"+str(args[1]), x=dm_list)
	np.savez("dm2_list"+str(args[1]), x=dm2_list)
	np.savez("dr_list"+str(args[1]), x=dr_list)
	np.savez("bdr_list"+str(args[1]), x=bdr_list)
	np.savez("drs_list"+str(args[1]), x=drs_list)
	np.savez("bdrs_list"+str(args[1]), x=bdrs_list)
	np.savez("drss_list"+str(args[1]), x=drss_list)
	np.savez("mdr_list"+str(args[1]), x=mdr3_list)

