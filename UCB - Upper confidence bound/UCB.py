# Implementation of UCB Algorithm
# coding: utf-8

# In[1]:


import numpy as np
from math import ceil,log,log10,sqrt
import matplotlib.pyplot as plt
import pandas as pd
import dill


# In[14]:


# Saving & Loading Variables
filename = 'globalsave.pkl'
# dill.load_session(filename)


# In[ ]:


# for reproducibility
np.random.seed(1234)


# In[3]:


# UCB Implementation given horizon (time steps), #replications, True arm means & Type of UCB algorithm
# For plotting % Commulative Optimal arm pulls & Commulative regret
def UCB(horizon,replications,arms_prob,ucbtype=1):

    optimal_arm = 0

    optimal_arm_pulls_per_round = np.zeros([horizon,replications]) # Stores % optimal arm pulls for every time steps
    regret_per_round = np.zeros([horizon,replications]) # Stores regret for every time steps
    
    for r in range(replications):
        arm_means = [0]*len(arms_prob) # Initializing arm means & pulls to 0
        arm_pulls = [0]*len(arms_prob)
        t = 0
        #initially playing each arm once
        for i in range(len(arms_prob)):
            arm_pulls[i]+=1
            temp = np.random.binomial(1,arms_prob[i])
            arm_means[i] += (temp - arm_means[i])/arm_pulls[i] # Updating arm means estimate
            if i == optimal_arm: # Incrementing % optimal arm pulls if current arm pulled is optimal arm
                optimal_arm_pulls_per_round[t][r] += 1
            regret_per_round[t][r] = (arms_prob[optimal_arm] - arms_prob[i]) # Storing regret for every time step
            t+=1
            
        while t < horizon:
            #Picking arm according to UCB algorithm or UCB' algorithm
            if ucbtype == 1:
                UCBEstimate = arm_means + np.sqrt(2*np.log(t)/arm_pulls)
                arm_selected = np.argmax(UCBEstimate)
            else:
                UCBEstimate = arm_means + np.sqrt(2*np.log(horizon)/arm_pulls)
                arm_selected = np.argmax(UCBEstimate)

            arm_pulls[arm_selected] += 1
            temp = np.random.binomial(1, arms_prob[arm_selected]) # Updating arm means estimate
            arm_means[arm_selected] += (temp - arm_means[arm_selected]) /arm_pulls[arm_selected]
            if arm_selected == optimal_arm: # Incrementing % optimal arm pulls if current arm pulled is optimal arm
                optimal_arm_pulls_per_round[t][r] += 1
            regret_per_round[t][r] = (arms_prob[optimal_arm] - arms_prob[arm_selected]) # Storing regret for every time step
            t+=1


    # Calculating Mean and Standard Error for % optimal arm pulls
    optimal_arm_means_stderr = np.zeros([horizon,2]) # Store % optimal arm means & stderr in (horizion X 2) array
    optimal_arm_means_stderr[:,0] = np.mean(optimal_arm_pulls_per_round,axis=1)
    optimal_arm_means_stderr[:,1] = (np.std(optimal_arm_pulls_per_round, axis=1)/sqrt(replications))
    optimal_arm_percentage = sum(optimal_arm_means_stderr[:,0])/horizon*100
    optimal_arm_pulls_sum = np.cumsum(optimal_arm_means_stderr[:,0])/horizon*100
    print("\nTotal Optimal arm pulls :",sum(optimal_arm_means_stderr[:,0]),'and percentage is :',optimal_arm_percentage)


    # Calculating Mean and Standard Error for commulative regret
    regret_means_stderr = np.zeros([horizon,2]) # Store regret means & stderr in (horizion X 2) array
    regret_means_stderr[:,0] = np.mean(regret_per_round,axis=1)
    regret_means_stderr[:,1] = (np.std(regret_per_round, axis=1)/sqrt(replications))
    total_regret = sum(regret_means_stderr[:,0])
    regret_per_round_sum = np.cumsum(regret_means_stderr[:,0])
    print("Total Regret :",total_regret)


    return regret_per_round_sum,regret_means_stderr, optimal_arm_pulls_sum,optimal_arm_means_stderr,optimal_arm_percentage,total_regret



# In[4]:


# UCB1 Implementation given horizon (time steps), #replications, True arm means & Type of UCB algorithm
# For plotting average % Optimal arm pulls
def UCB1(horizon,replications,arms_prob,ucbtype=1):

    optimal_arm = 0

    optimal_arm_pulls_per_round = np.zeros([horizon,replications]) # Stores % optimal arm pulls for every time steps
    regret_per_round = np.zeros([horizon,replications]) # Stores regret for every time steps
    
    for r in range(replications):
        arm_means = [0]*len(arms_prob) # Initializing arm means & pulls to 0
        arm_pulls = [0]*len(arms_prob)
        t = 0
        #initially playing each arm once
        for i in range(len(arms_prob)):
            arm_pulls[i]+=1
            temp = np.random.binomial(1,arms_prob[i])
            arm_means[i] += (temp - arm_means[i])/arm_pulls[i] # Updating arm means estimate
            optimal_arm_pulls_per_round[t][r] = arm_pulls[optimal_arm]*100.0/(t+1) # Storing % optimal arm pulls at every time step
            regret_per_round[t][r] = (arms_prob[optimal_arm] - arms_prob[i]) # Storing regret for every time step
            t+=1
            
        while t < horizon:
            #Picking arm according to UCB algorithm or UCB' algorithm
            if ucbtype == 1:
                UCBEstimate = arm_means + np.sqrt(2*np.log(t)/arm_pulls)
                arm_selected = np.argmax(UCBEstimate)
            else:
                UCBEstimate = arm_means + np.sqrt(2*np.log(horizon)/arm_pulls)
                arm_selected = np.argmax(UCBEstimate)

            arm_pulls[arm_selected] += 1
            temp = np.random.binomial(1, arms_prob[arm_selected]) # Updating arm means estimate
            arm_means[arm_selected] += (temp - arm_means[arm_selected]) /arm_pulls[arm_selected]
            optimal_arm_pulls_per_round[t][r] = arm_pulls[optimal_arm]*100.0/(t+1) # Storing % optimal arm pulls at every time step
            regret_per_round[t][r] = (arms_prob[optimal_arm] - arms_prob[arm_selected]) # Storing regret for every time step
            t+=1


    # Calculating Mean and Standard Error for % optimal arm pulls
    optimal_arm_means_stderr = np.zeros([horizon,2]) # Store % optimal arm means & stderr in (horizion X 2) array
    optimal_arm_means_stderr[:,0] = np.mean(optimal_arm_pulls_per_round,axis=1)
    optimal_arm_means_stderr[:,1] = (np.std(optimal_arm_pulls_per_round, axis=1)/sqrt(replications))
    optimal_arm_percentage = sum(optimal_arm_means_stderr[:,0])/horizon
    optimal_arm_pulls_sum = np.mean(optimal_arm_pulls_per_round,axis=1)
   

    # Calculating Mean and Standard Error for commulative regret
    regret_means_stderr = np.zeros([horizon,2]) # Store regret means & stderr in (horizion X 2) array
    regret_means_stderr[:,0] = np.mean(regret_per_round,axis=1)
    regret_means_stderr[:,1] = (np.std(regret_per_round, axis=1)/sqrt(replications))
    total_regret = sum(regret_means_stderr[:,0])
    regret_per_round_sum = np.cumsum(regret_means_stderr[:,0])
    print("Total Regret :",total_regret)


    return regret_per_round_sum,regret_means_stderr, optimal_arm_pulls_sum,optimal_arm_means_stderr,optimal_arm_percentage,total_regret



# In[5]:


# Plotting % Cummulative Optimal Arm Pulls Vs Time steps with error bars
def plotOptimalArmPulls_old(horizon,optimal_arm_means_stderr,optimal_arm_pulls_sum,problem,step):
    x = np.arange(horizon)
    ind = [i for i in range(0,horizon,step)]

    for i in range(m_len):
        plt.errorbar(x[ind],optimal_arm_pulls_sum[i,ind], optimal_arm_means_stderr[i,ind,1],
                    linestyle='-', marker='x',capsize=4,capthick=1.5,elinewidth=1.5)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Arm Pulls')
    plt.legend(['UCB',"UCB'"],loc=0,frameon=False)
    plt.title(arms_prob[problem])
    plt.xlim((0,10000))
    plt.ylim((0,100))
    plt.savefig('OptimalArmPulls_old_'+str(problem)+'.png',dpi=300)
    plt.show()

    print("optimal_arm_stderr")
    print(optimal_arm_means_stderr[:,[500,2000,5000,8000,9500],1])


# In[7]:


# Plotting % Average Optimal Arm Pulls Vs Time steps with error bars
def plotOptimalArmPulls(horizon,optimal_arm_means_stderr,optimal_arm_pulls_sum,problem,step):
    x = np.arange(horizon)
    ind = [i for i in range(0,horizon,step)]

    for i in range(m_len):
        plt.errorbar(x[ind],optimal_arm_pulls_sum[i,ind], optimal_arm_means_stderr[i,ind,1],
                    linestyle='-', marker='x',capsize=4,capthick=1.5,elinewidth=1.5)
    plt.xlabel('Steps')
    plt.ylabel('% Average Optimal Arm Pulls')
    plt.legend(['UCB',"UCB'"],loc=0,frameon=False)
    plt.title(arms_prob[problem])
    plt.xlim((0,10000))
    plt.ylim((0,100))
    plt.savefig('OptimalArmPulls_'+str(problem)+'.png',dpi=300)
    plt.show()

    print("optimal_arm_means_stderr")
    print(optimal_arm_means_stderr[:,[500,2000,5000,8000,9500],1])


# In[8]:


# Plotting Cumulative Regret Vs Time steps with error bars
def plotCummRegret(horizon,regret_means_stderr,regret_per_round_sum,problem,step):
    x = np.arange(horizon)
    ind = [i for i in range(0,horizon,step)]

    for i in range(m_len):
        plt.errorbar(x[ind],regret_per_round_sum[i,ind], regret_means_stderr[i,ind,1],
                linestyle='-', marker='x',capsize=4,capthick=1.5,elinewidth=1.5)

    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.legend(['UCB',"UCB'"],loc=0,frameon=False)
    plt.title(arms_prob[problem])
    plt.xlim((0,10000))
    plt.savefig('CumulativeRegret_'+str(problem)+'.png',dpi=300)
    plt.show()

    print("regret_means_stderr")
    print(regret_means_stderr[:,[500,2000,5000,8000,9500],1])


# In[11]:

# For Printing table
from IPython.display import HTML, display

def tableIt(data):
    print(pd.DataFrame(data))


# In[10]:


horizon = 10000
replications = 100
arms_prob = [[0.9, 0.6], [0.9, 0.8], [0.55, 0.45]]
UCBtype = [1,2]
m_len = len(UCBtype)

for problem in range(3): # Repeating for 3 problems
    optimal_arm_pulls_sum = np.zeros([m_len,horizon]) # Storing variables returned by UCB function
    regret_per_round_sum = np.zeros([m_len,horizon])
    optimal_arm_means_stderr = np.zeros([m_len,horizon,2])
    regret_means_stderr = np.zeros([m_len,horizon,2])
    optimal_arm_percentage = np.zeros([m_len])
    total_regret = np.zeros([m_len])


    for i in range(m_len): # Repeating for two algorithms
        regret_per_round_sum[i,:],regret_means_stderr[i,:,:], optimal_arm_pulls_sum[i,:],optimal_arm_means_stderr[i,:,:],optimal_arm_percentage[i],total_regret[i] = UCB1(horizon,replications,arms_prob[problem],ucbtype=UCBtype[i])


    step = 300
    print("\n")
    print("optimal_arm_percentage")
    tableIt(optimal_arm_percentage)
    print("\n")
    print("total_regret")
    tableIt(total_regret)

    # Calling function to plot % Average Optimal Arm Pulls & Commulative regret with error bars
    plotOptimalArmPulls(horizon,optimal_arm_means_stderr,optimal_arm_pulls_sum,problem,step)
    plotCummRegret(horizon,regret_means_stderr,regret_per_round_sum,problem,step)
    



# In[12]:



horizon = 10000
replications = 100
arms_prob = [[0.9, 0.6], [0.9, 0.8], [0.55, 0.45]]
UCBtype = [1,2]
m_len = len(UCBtype)

for problem in range(3): # Repeating for 3 problems
    optimal_arm_pulls_sum = np.zeros([m_len,horizon]) # Storing variables returned by UCB function
    regret_per_round_sum = np.zeros([m_len,horizon])
    optimal_arm_means_stderr = np.zeros([m_len,horizon,2])
    regret_means_stderr = np.zeros([m_len,horizon,2])
    optimal_arm_percentage = np.zeros([m_len])
    total_regret = np.zeros([m_len])


    for i in range(m_len): # Repeating for two algorithms
        regret_per_round_sum[i,:],regret_means_stderr[i,:,:], optimal_arm_pulls_sum[i,:],optimal_arm_means_stderr[i,:,:],optimal_arm_percentage[i],total_regret[i] = UCB(horizon,replications,arms_prob[problem],ucbtype=UCBtype[i])


    step = 300
    print("\n")
    print("optimal_arm_percentage")
    tableIt(optimal_arm_percentage)
    print("\n")
    print("total_regret")
    tableIt(total_regret)

    # Calling function to plot % Optimal Arm Pulls Vs Time steps with error bars
    plotOptimalArmPulls_old(horizon,optimal_arm_means_stderr,optimal_arm_pulls_sum,problem,step)


# In[12]:

# Saving variables
dill.dump_session(filename)


# Calculating Theoretical Regret
n = 10000
arms_prob = [[0.9, 0.6], [0.9, 0.8], [0.55, 0.45]]
gap_dependent_regret = []
gap_independent_regret = []
for i in range(3):
    gap = arms_prob[i][0] - arms_prob[i][1]
    dep_reg = (32*log(n)/gap) + 2*(1 + (2*np.pi**2/3))
    gap_dependent_regret.append(dep_reg)
    
    indep_reg = sqrt(2*n*(32*log(n) + 1 + (np.pi**2/3)))
    gap_independent_regret.append(indep_reg)
    
print('gap_dependent_regret')
print(gap_dependent_regret)

print('\ngap_independent_regret')
print(gap_independent_regret)
