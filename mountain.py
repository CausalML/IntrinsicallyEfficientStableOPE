####%matplotlib inline

import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
import matplotlib.pyplot as plt
%matplotlib inline

if "../" not in sys.path:
  sys.path.append("../") 

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from scipy.optimize import minimize, rosen, rosen_der
from scipy.optimize import Bounds
from collections import defaultdict


args = sys.argv
ver_ = str(args[1])
sample_size = np.int(args[2])

####sample_size = 1000

matplotlib.style.use('ggplot')

env = gym.envs.make("MountainCar-v0")

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
###observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
###np.savez("observa.npz", xxx=observation_examples)
observation_examples = np.load("observa.npz")["xxx"]

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100,random_state=1)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100,random_state=1)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100,random_state=1)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100,random_state=1))
        ])
featurizer.fit(scaler.transform(observation_examples))


state = env.reset()

class Estimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]
    
    def predict(self, s, a=None):
        """
        Makes value function predictions.
        
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
            
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
            
        """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]
    
    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])
        
    def get_parameter(self):
        coef_list = []
        for model in self.models:
            coef_list.append(model.coef_)
        return coef_list
    
    def set_parameter(self,parameter_list):
        
        for i in range(env.action_space.n):
            self.models[i].coef_ = parameter_list[i,:]

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env,estimator,estimator2, num_episodes, discount_factor=1.0, epsilon=0.1,epsilon2=0.15, epsilon_decay=1.0):
   

    policy = make_epsilon_greedy_policy(estimator, epsilon, env.action_space.n)
    policy2 = make_epsilon_greedy_policy(estimator, epsilon2, env.action_space.n)

    
    for i_episode in range(num_episodes):
        
        if i_episode%100 ==0:
            print i_episode
            
        state = env.reset()
        
        next_action = None
        action_probs = policy2(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        # One step in the environment
        for t in itertools.count():
                        
            next_state, reward, done, _ = env.step(action)
    
        
            q_values_next = estimator2.predict(next_state)
                
            next_action_probs = policy2(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)             
            td_target = reward + q_values_next[next_action]*policy(next_state)[next_action]/policy2(next_state)[next_action]
            
            # Update the function approximator using our target
            estimator2.update(state, action, td_target)
            
                
            if done:
                break
                
            state = next_state
            action = next_action
    
    return estimator2.get_parameter()



def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))
###bounds = Bounds([-0.1,-0.1,-0.1],[0.1,0.1])

depth = 1

def all_(env,estimator,estimator2, num_episodes, discount_factor=1.0, epsilon=0.1,epsilon2=0.15,epsilon_decay=1.0):
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    returns_count2 = defaultdict(float)
 
    predic_list = []
    predic_list2 = []
    predic_list3 = []
    predic_list4 = []
    predic_list5 = np.ones(num_episodes)
    auxiauxi = [] ### FOR ipw 2
    epiepi = []
    weight_list = np.zeros([num_episodes,1000]) ### For bounded IPW
    weight_list2 = np.zeros([num_episodes,1002]) ### For bounded IPW
    auxi_list = np.zeros([num_episodes,1000])
    auxi_list2 = np.zeros([num_episodes,1000])
    reward_list = np.zeros([num_episodes,1000])
    ###state_list = np.zeros([num_episodes,1000])
    state_list = []
    
    action_list = np.zeros([num_episodes,1000])
    
    count_list = np.zeros(1000) ### For bounded IPW
    episolode_longe_list = []
    
    predic_list = []
    # Keeps track of useful statistics
        
    
    policy = make_epsilon_greedy_policy(estimator, epsilon, env.action_space.n)
    policy2 = make_epsilon_greedy_policy(estimator, epsilon2, env.action_space.n)
        
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 200 == 0:
            print(i_episode, num_episodes)
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
    
        state = env.reset()
        for t in itertools.count():
            action_probs = policy2(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            epiepi.append((state, action, reward))
            if done:
                break
            state = next_state
            
        state_list.append(episode)
    

        W = 1.0
        W_list = []
        episolode_longe_list.append(len(episode))
        
        weight_list2[i_episode,0] = 1.0
        for t in range(len(episode)):
            state, action, reward = episode[t]
            reward_list[i_episode,t] = reward
            ####state_list[i_episode,t] = state
            ####action_list[i_episode,t] = action
            
            W = W*policy(state)[action]/policy2(state)[action]*discount_factor
            probprob = policy(state)
            W_list.append(W)
            weight_list[i_episode,t] = W_list[t]
            weight_list2[i_episode,t+1] = W_list[t]
            
            count_list[t] += 1.0
            
            if t==0:
                auxi_list[i_episode,t] = W_list[t]*estimator2.predict(state)[action]-np.sum(probprob*estimator2.predict(state))
            else:
                auxi_list[i_episode,t] = W_list[t]*estimator2.predict(state)[action]-W_list[t-1]*np.sum(probprob*estimator2.predict(state))
          
            if t==0:
                auxi_list2[i_episode,t] = W_list[t]-1.0
            else:
                auxi_list2[i_episode,t] = W_list[t]-W_list[t-1]


    print np.max(np.array(episolode_longe_list))
    
        
    weight_list_mean = np.mean(weight_list,1)
    reward_list_mean = np.mean(reward_list,1)
    auxi_list_mean = np.mean(auxi_list,1)
    auxi_list2_mean = np.mean(auxi_list2,1)
    
    val = []    
 
    ##### IPW
    for i in range(num_episodes):
        predic_list.append(np.sum(weight_list[i,:]*reward_list[i,:]))   
    
    val.append(np.mean(predic_list))
    
    ### B-IPW 

    for i in range(num_episodes):
        ddd1 = weight_list[i,:]*reward_list[i,:]
        ddd1 = ddd1[0:episolode_longe_list[i]]
        ddd2 = np.sum(weight_list,0)/(count_list+0.0001)
        ddd2 = ddd2[0:episolode_longe_list[i]]
        predic_list2.append(np.sum(ddd1/ddd2))
        
    val.append(np.mean(predic_list2))
    
    #### DR
    val.append(np.mean(predic_list)-np.mean(np.sum(auxi_list,0)))
    
    
    #### B-DR 
    val.append(np.mean(predic_list)-np.mean(np.sum(auxi_list,0)))
    """
    for i in range(num_episodes):
        ddd1 = weight_list[i,:]
        ddd1 = ddd1[0:episolode_longe_list[i]]
        ddd2 = np.sum(weight_list,0)/(count_list+0.0001)
        ddd2 = ddd2[0:episolode_longe_list[i]]
        ddd3 = ddd1/ddd2
        sumsum = 0.0
        for t in range(episolode_longe_list[i]):
            temp1 = np.int(state_list[i,t])
            probprob = 0.9*Q_space[temp1,:] + 0.1*prob1
            temp2 = np.int(action_list[i,t])
            if t==0:
                sumsum += ddd3[t]*Q_[temp1,temp2]-np.sum(probprob*Q_[temp1,:])
            else:
                sumsum += ddd3[t]*Q_[temp1,temp2]-ddd3[t-1]*np.sum(probprob*Q_[temp1,:])
        predic_list3.append(sumsum)
     
    
    val.append(np.mean(predic_list2)-np.mean(predic_list3))
    """
    #### DRS1
    
    def variance(para):
        return np.var(predic_list-para[0]*np.sum(auxi_list,1)-para[1]*np.sum(auxi_list2,1))
    
    res = minimize(variance, [0,1], method='L-BFGS-B')
    print res.x
    val.append(np.mean(predic_list-res.x[0]*np.sum(auxi_list,1)-res.x[1]*np.sum(auxi_list2,1)))
    
    #### DRS 2
    
    depth =3 
    def variance2(para):
        para2 = np.ones(1000)*para[1]
        para2[0:depth-2] = para[2:depth]     
        return np.var(predic_list-para[0]*np.sum(auxi_list,1)-np.sum(auxi_list2*para2,1))
    
    res = minimize(variance2, np.zeros(depth), method='L-BFGS-B')
    print res.x
    para_ = np.ones(1000)*res.x[1]
    para_[0:depth-2] = res.x[2:depth]
    val.append(np.mean(predic_list-res.x[0]*np.sum(auxi_list,1)-np.sum(auxi_list2*para_,1)))
    
    ### EMP 
    
    def emp(para):
        if np.min(1.0+para[0]*np.sum(auxi_list,1)+para[1]*np.sum(auxi_list2,1))<0.0:
            return(10)
        else:
            aaa = np.log(1.0+para[0]*np.sum(auxi_list,1)+para[1]*np.sum(auxi_list2,1))
        return -np.mean(aaa) 
    
    res2 =  minimize(emp,method="L-BFGS-B",x0=np.array([0.0,0.0]))
    print res2.x
    
    www = 1.0/(1.0+res2.x[0]*np.sum(auxi_list,1)+res2.x[1]*np.sum(auxi_list2,1))
    val.append(np.mean(predic_list*www)/np.mean(www))
    
    ### MDR 
    
    
    sample_dim = 2
    sample_size = num_episodes
    def loss(beta):
        score = np.array([ np.sum([np.dot(beta,state_list[i][t][0])*weight_list2[i,t+1] for t in range(episolode_longe_list[i])] )  for i in range(sample_size)])
        score2 = np.array([ np.sum([np.dot(beta,state_list[i][t][0])*weight_list2[i,t] for t in range(episolode_longe_list[i])] )  for i in range(sample_size)])
        score3 = predic_list-score+score2
        return np.mean(score3*score3)

    minmin = minimize(loss,np.zeros(sample_dim),method='BFGS')
    beta = minmin.x
    ###print(beta)
    score = np.array([ np.sum([np.dot(beta,state_list[i][t][0])*weight_list2[i,t+1] for t in range(episolode_longe_list[i])] )  for i in range(sample_size)])
    score2 = np.array([ np.sum([np.dot(beta,state_list[i][t][0])*weight_list2[i,t] for t in range(episolode_longe_list[i])] )  for i in range(sample_size)])
    score3 = predic_list-score+score2
    
    val.append(np.mean(score3))
    
    bbb = []
    e_policy = make_epsilon_greedy_policy(estimator, epsilon2, env.action_space.n)
    
    #### DM 
    
    for i in range(1000):
        state = env.reset()
        q_values = estimator2.predict(state)
        next_action_probs = e_policy(state)
        bbb.append(np.sum(next_action_probs*q_values))

    val.append(np.mean(bbb))
    
    return val

is_list = []
wis_list = []
wis2_list = []
dm_list = []
dr_list = []
bdr_list = []
drs_list = []
drs2_list = []
drss_list = []
mdr_list = []
#####mdr_list2 = []

sample_size = 3000
bbb_list = []
for i in range(20):
    estimator = Estimator()
    weight = np.load("weight.npz")["xxx"]
    estimator.set_parameter(np.copy(weight))
    estimator2 = Estimator()
    estimator2.set_parameter(np.copy(weight))
    paraparapara = q_learning(env,estimator,estimator2, sample_size,)
    estimator2.set_parameter(np.copy(paraparapara))
    print i
    V_10k =  all_(env,estimator,estimator2, sample_size)   
    is_list.append(np.mean(V_10k[0]))
    np.save("./ope_data/mountain_data"+ver_+"is_list_"+str(sample_size)+".npy", is_list)
    wis_list.append(np.mean(V_10k[1]))
    np.save("./ope_data/mountain_data"+ver_+"wis_list_"+str(sample_size)+".npy", wis_list)
    dr_list.append(np.mean(V_10k[2]))
    np.save("./ope_data/mountain_data"+ver_+ "dr_list_"+str(sample_size)+".npy", dr_list)
    bdr_list.append(np.mean(V_10k[3]))  
    np.save("./ope_data/mountain_data"+ver_+"bdr_list_"+str(sample_size)+".npy", bdr_list)
    drs_list.append(np.mean(V_10k[4]))
    np.save("./ope_data/mountain_data"+ver_+ "drs_list_"+str(sample_size)+".npy", drs_list)
    drs2_list.append(np.mean(V_10k[5]))
    np.save("./ope_data/mountain_data"+ver_+"drs2_list_"+str(sample_size)+".npy", drs2_list)
    drss_list.append(np.mean(V_10k[6]))
    np.save("./ope_data/mountain_data"+ver_+"drss_list_"+str(sample_size)+".npy", drss_list)
    mdr_list.append(np.mean(V_10k[7]))
    np.save("./ope_data/mountain_data"+ver_+"mdr_list_"+str(sample_size)+".npy", mdr_list)
    dm_list.append(np.mean(V_10k[8]))
    np.save("./ope_data/mountain_data"+ver_+"dm_list_"+str(sample_size)+".npy", dm_list)
    np.save("./ope_data/mountain_data"+ver_+"ite_"+str(sample_size)+".npy", i)

