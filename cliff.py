import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys


if "../" not in sys.path:
  sys.path.append("../") 

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.envs.windy_gridworld import WindyGridworldEnv
####from lib import plotting
from scipy.optimize import minimize, rosen, rosen_der
from scipy.optimize import Bounds
###import matplotlib.pyplot as plt
#####%matplotlib inline

bounds = Bounds([-0.1,-0.1],[0.1,0.1])
sample_size = 250
ite_number = 400


#####matplotlib.style.use('ggplot')

env = CliffWalkingEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.0):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

       
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            i_episode
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            
            
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state
    
    return Q

Q = q_learning(env, 1000)

Q_space = np.load("Q-table-cliff.npz")["xxx"]
Q_space2 = np.load("Q-table-real-cliff.npz")["xxx"]

prob1 = [1.0 for i in range((env.nA))]
prob1 = prob1/np.sum(prob1)


def sample_policy(observation,alpha=0.9):
    prob2 = alpha*Q_space[observation,:] +(1-alpha)*prob1
    return np.random.choice(env.nA,1,p=prob2)[0]
    
        
def behavior_policy(observation,beta=0.8):
    prob2 = beta*Q_space[observation,:]+ (1-beta)*prob1
    return np.random.choice(env.nA,1,p=prob2)[0]
    
    
def target_dense(observation,alpha=0.9):
    prob2 = alpha*Q_space[observation,:]+ (1-alpha)*prob1
    return prob2
def behav_dense(observation,beta=0.8):
    prob2 = beta*Q_space[observation,:] + (1-beta)*prob1
    return prob2

def sarsa2(env,policy, policy2,num_episodes, discount_factor=1.0,Q_space2=Q_space2, alpha=0.1, epsilon=0.03):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = Q_space2*0.0##defaultdict(lambda: np.zeros(env.action_space.n))
    episode_whole = []
    episolode_longe_list = []
    


    # The policy we're following
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 200 == 0:
            ###print i_episode + 1, num_episodes
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        episode  =[]
        action = policy2(state)
        # One step in the environment
        
        for t in itertools.count():
            # Take a step
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            
            
            # Pick the next action
            next_action= policy2(next_state)
            
     
            
            # TD Update
            td_target = reward + discount_factor * Q[next_state,next_action]
            td_delta = td_target - Q[state,action]
            Q[state,action] += alpha * td_delta * target_dense(state)[action]/behav_dense(state)[action]
    
            if done:
                break
                
            action = next_action
            state = next_state  
        episolode_longe_list.append(len(episode))
        episode_whole.append(episode)
    
    return Q, episode_whole, episolode_longe_list



bounds = Bounds([-0.2,-0.2],[0.2,0.2])
def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))
###bounds = Bounds([-0.1,-0.1,-0.1],[0.1,0.1])

depth = 1
def mc_prediction(env, policy,policy2, episode_whole, episolode_longe_list, Q_=1.0,num_episodes=100, discount_factor=1.0):
   

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
    state_list = np.zeros([num_episodes,1000])
    action_list = np.zeros([num_episodes,1000])
    
    count_list = np.zeros(1000) ### For bounded IPW
    ###episolode_longe_list = []
    

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 200 == 0:
            ####print(i_episode, num_episodes)
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = episode_whole[i_episode]
        
        
        ####print episode
        """
        episode = []
        
        print(episode)
        state = env.reset()
        for t in itertools.count():
            action = policy2(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            epiepi.append((state, action, reward))
            if done:
                break
            state = next_state
        
        print(episode)
        """
        # Find all states the we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        '''
        Original coding
        for i,state in enumerate(episode):
            G = sum([x[2]*(discount_factor**j)  for j,x in enumerate(episode[i:])])
            ratio = np.sum([np.log(target_dense(x[0],x[1])/behav_dense(x[0],x[1])) for j,x in enumerate(episode[i:])])
            state_tuple = tuple(state[0])
            returns_sum[state_tuple] += G*np.exp(ratio)
            returns_count[state_tuple] += 1.0 #### np.exp(ratio)
            V[state_tuple] = returns_sum[state_tuple] / returns_count[state_tuple]
        '''  

        W = 1.0
        W_list = []
        ####episolode_longe_list.append(len(episode))
        
        
        weight_list2[i_episode,0] = 1.0
        for t in range(len(episode)):
            state, action, reward = episode[t]
            reward_list[i_episode,t] = reward
            state_list[i_episode,t] = state
            action_list[i_episode,t] = action
            
            W = W*target_dense(state)[action]/behav_dense(state)[action]*discount_factor
            probprob = 0.9*Q_space[state,:] + 0.1*prob1
            W_list.append(W)
            weight_list[i_episode,t] = W_list[t]
            weight_list2[i_episode,t+1] = W_list[t]
            
            count_list[t] += 1.0
            
            if t==0:
                auxi_list[i_episode,t] = W_list[t]*Q_[state,action]-np.sum(probprob*Q_[state,:])
            else:
                auxi_list[i_episode,t] = W_list[t]*Q_[state,action]-W_list[t-1]*np.sum(probprob*Q_[state,:])
          
            if t==0:
                auxi_list2[i_episode,t] = W_list[t]-1.0
            else:
                auxi_list2[i_episode,t] = W_list[t]-W_list[t-1]

    ###print np.max(np.array(episolode_longe_list))
    
        
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
        
        
    ###print ddd1
    #####print ddd2
    val.append(np.mean(predic_list2))
    
    #### DR
    val.append(np.mean(predic_list)-np.mean(np.sum(auxi_list,0)))
    
    
    #### B-DR 

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
    
    #### DRS1
    
    def variance(para):
        return np.var(predic_list-para[0]*np.sum(auxi_list,1)-para[1]*np.sum(auxi_list2,1))
    
    res = minimize(variance, [0,1], method='L-BFGS-B')
    ######print res.x
    val.append(np.mean(predic_list-res.x[0]*np.sum(auxi_list,1)-res.x[1]*np.sum(auxi_list2,1)))
    
    #### DRS 2
    
    depth =3 
    def variance2(para):
        para2 = np.ones(1000)*para[1]
        para2[0:depth-2] = para[2:depth]     
        return np.var(predic_list-para[0]*np.sum(auxi_list,1)-np.sum(auxi_list2*para2,1))
    
    res = minimize(variance2, np.zeros(depth), method='L-BFGS-B')
    ####print res.x
    para_ = np.ones(1000)*res.x[1]
    para_[0:depth-2] = res.x[2:depth]
    val.append(np.mean(predic_list-res.x[0]*np.sum(auxi_list,1)-np.sum(auxi_list2*para_,1)))
    
    
    
    #### MDR 
    
    
    sample_dim = 1
    def loss(beta):
        score = np.array([ np.sum([np.dot(beta,state_list[i,t])*weight_list2[i,t+1] for t in range(episolode_longe_list[i])] )  for i in range(sample_size)])
        score2 = np.array([ np.sum([np.dot(beta,state_list[i,t])*weight_list2[i,t] for t in range(episolode_longe_list[i])] )  for i in range(sample_size)])
        score3 = predic_list-score+score2
        return np.mean(score3*score3)

    minmin = minimize(loss,np.zeros(sample_dim),method='BFGS')
    beta = minmin.x
    ####print(beta)
    score = np.array([ np.sum([np.dot(beta,state_list[i,t])*weight_list2[i,t+1] for t in range(episolode_longe_list[i])] )  for i in range(sample_size)])
    score2 = np.array([ np.sum([np.dot(beta,state_list[i,t])*weight_list2[i,t] for t in range(episolode_longe_list[i])] )  for i in range(sample_size)])
    score3 = predic_list-score+score2
    
    val.append(np.mean(score3))
   
    #### MDR2
    
    """
    sample_dim = 4
    def loss(beta):
        score = np.array([ np.sum([np.dot(beta[np.int(action_list[i,t])],state_list[i,t])*weight_list2[i,t+1] for t in range(episolode_longe_list[i])] )  for i in range(sample_size)])
        score2 = np.array([ np.sum([ np.sum([np.dot(beta[j],state_list[i,t])*target_dense(np.int(state_list[i,t]))[j] for j in range(4) ])*weight_list2[i,t] for t in range(episolode_longe_list[i])] )  for i in range(sample_size)])
        score3 = predic_list-score+score2
        return np.mean(score3*score3)
    
    minmin = minimize(loss,np.zeros(sample_dim),method='BFGS')
    beta = minmin.x
    print(beta)
    score = np.array([ np.sum([np.dot(beta[np.int(action_list[i,t])],state_list[i,t])*weight_list2[i,t+1] for t in range(episolode_longe_list[i])] )  for i in range(sample_size)])
    score2 = np.array([ np.sum([ np.sum([np.dot(beta[j],state_list[i,t])*target_dense(np.int(state_list[i,t]))[j] for j in range(4) ])*weight_list2[i,t] for t in range(episolode_longe_list[i])] )  for i in range(sample_size)])
    score3 = predic_list-score+score2
    
    val.append(np.mean(score3))
    """
    return val
    
    
                                                  
    


##### Policy evalution using SARSA



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
mdr_list2 = []

sample_size = 500
for kkk in range(500):
    print kkk
    predicted_Q,episode_whole, episolode_longe_list = sarsa2(env,sample_policy,behavior_policy, num_episodes=sample_size)
    V_10k = mc_prediction(env,sample_policy,behavior_policy,episode_whole, episolode_longe_list, predicted_Q,num_episodes=sample_size)
    is_list.append(np.mean(V_10k[0]))
    wis_list.append(np.mean(V_10k[1]))
    dr_list.append(np.mean(V_10k[2]))
    bdr_list.append(np.mean(V_10k[3]))               
    drs_list.append(np.mean(V_10k[4]))
    drs2_list.append(np.mean(V_10k[5]))
    mdr_list.append(np.mean(V_10k[6]))
    ####mdr_list2.append(np.mean(V_10k[7]))


###print np.sqrt(np.mean(np.square(np.array(is_list)+1.96)))

true = -42.49
def mse(aaa):
    aaa = np.array(aaa)
    return np.sqrt(np.mean((aaa+17.159)*(aaa+17.159)))

print np.mean(is_list)
print mse(is_list)
print "wis"
print np.mean(wis_list)
print mse(wis_list)
print "wis2"
print np.mean(wis2_list)
print mse(wis2_list)
print "dm"
print np.mean(dm_list)
print mse(dm_list)
print "dr"
print np.mean(dr_list)
print mse(dr_list)
print "bdr"
print np.mean(bdr_list)
print mse(bdr_list)
print "drs"
print np.mean(drs_list)
print mse(drs_list)
print "drs2"
print np.mean(drs2_list)
print mse(drs2_list)
print "drss"
print np.mean(drss_list)
print mse(drss_list)
print "mdr"
print np.mean(mdr_list)
print mse(mdr_list)
print "mdr"
print np.mean(mdr_list2)
print mse(mdr_list2)
