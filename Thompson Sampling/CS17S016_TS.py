
# coding: utf-8

# In[1]:


import numpy as np
from math import ceil,log,log10,sqrt,exp
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd
import dill


# In[2]:


# Saving & Loading Variables
filename = 'globalsave.pkl'
# dill.load_session(filename)


# In[3]:


# for reproducibility
np.random.seed(1234)


# In[4]:


# UCB Implementation given horizon (time steps), #replications, True arm means & Type of UCB algorithm
# For plotting average % Optimal arm pulls
def UCB(horizon,replications,arms_prob,ucbtype,optimalpulls):

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
            if optimalpulls == 'Avg':
                optimal_arm_pulls_per_round[t][r] = arm_pulls[optimal_arm]*100.0/(t+1) # Storing % optimal arm pulls at every time step
            else:
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
            regret_per_round[t][r] = (arms_prob[optimal_arm] - arms_prob[arm_selected]) # Storing regret for every time step
            if optimalpulls == 'Avg':
                optimal_arm_pulls_per_round[t][r] = arm_pulls[optimal_arm]*100.0/(t+1) # Storing % optimal arm pulls at every time step
            else:
                if arm_selected == optimal_arm: # Incrementing % optimal arm pulls if current arm pulled is optimal arm
                    optimal_arm_pulls_per_round[t][r] += 1
            t+=1


    # Calculating Mean and Standard Error for % optimal arm pulls
    optimal_arm_means_stderr = np.zeros([horizon,2]) # Store % optimal arm means & stderr in (horizion X 2) array
    optimal_arm_means_stderr[:,0] = np.mean(optimal_arm_pulls_per_round,axis=1)
    optimal_arm_means_stderr[:,1] = (np.std(optimal_arm_pulls_per_round, axis=1)/sqrt(replications))
    if optimalpulls == 'Avg':
        optimal_arm_percentage = sum(optimal_arm_means_stderr[:,0])/horizon
        optimal_arm_pulls_sum = np.mean(optimal_arm_pulls_per_round,axis=1)
    else:
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



# In[5]:


# Implemention Thompson Sampling
def TS(horizon,replications,arms_prob,alpha,beta,optimalpulls):
    optimal_arm = 0
    optimal_arm_pulls_per_round = np.zeros([horizon,replications])
    regret_per_round = np.zeros([horizon,replications])
    savepoints = (0,1000,5000,9999)
    success_ret = np.zeros([len(savepoints),len(arms_prob)])
    faliure_ret = np.zeros([len(savepoints),len(arms_prob)])
    
    for r in range(replications):
        arm_pulls = [0]*len(arms_prob)
        success = np.array(alpha)
        failure = np.array(beta)
        t = 0
        s = 0
        while t < horizon:
            if t in savepoints and r == replications-1:
                success_ret[s] = success
                faliure_ret[s] = failure
                s+=1
                
            #Picking arm according to Posterior distribution
            sample_means = [0]*len(arms_prob)
            for i in range(len(arms_prob)):
                sample_means[i] = np.random.beta(success[i],failure[i])
            arm_selected = np.argmax(sample_means)
            

            arm_pulls[arm_selected] += 1
            temp = np.random.binomial(1, arms_prob[arm_selected])
            success[arm_selected] += temp
            failure[arm_selected] += 1 - temp
            
            if optimalpulls == 'Avg':
                optimal_arm_pulls_per_round[t][r] = arm_pulls[optimal_arm]*100.0/(t+1)
            else:
                if arm_selected == optimal_arm: # Incrementing % optimal arm pulls if current arm pulled is optimal arm
                    optimal_arm_pulls_per_round[t][r] += 1
                    
            regret_per_round[t][r] = (arms_prob[optimal_arm] - arms_prob[arm_selected])
            
            t+=1


    # Calculating Mean and Standard Error for % optimal arm pulls
    optimal_arm_means_stderr = np.zeros([horizon,2])
    optimal_arm_means_stderr[:,0] = np.mean(optimal_arm_pulls_per_round,axis=1)
    optimal_arm_means_stderr[:,1] = (np.std(optimal_arm_pulls_per_round, axis=1)/sqrt(replications))
    
    if optimalpulls == 'Avg':
        optimal_arm_percentage = sum(optimal_arm_means_stderr[:,0])/horizon
        optimal_arm_pulls_sum = np.mean(optimal_arm_pulls_per_round,axis=1)
    else:
        optimal_arm_percentage = sum(optimal_arm_means_stderr[:,0])/horizon*100
        optimal_arm_pulls_sum = np.cumsum(optimal_arm_means_stderr[:,0])/horizon*100

    print("\nTotal Optimal arm pulls :",sum(optimal_arm_means_stderr[:,0]),'and percentage is :',optimal_arm_percentage)


    # Calculating Mean and Standard Error for commulative regret
    regret_means_stderr = np.zeros([horizon,2])
    regret_means_stderr[:,0] = np.mean(regret_per_round,axis=1)
    regret_means_stderr[:,1] = (np.std(regret_per_round, axis=1)/sqrt(replications))
    total_regret = sum(regret_means_stderr[:,0])
    regret_per_round_sum = np.cumsum(regret_means_stderr[:,0])
    print("Total Regret :",total_regret)


    return success_ret,faliure_ret,regret_per_round_sum,regret_means_stderr, optimal_arm_pulls_sum,optimal_arm_means_stderr,optimal_arm_percentage,total_regret


# In[6]:


# Calculating Lower bound Regret
def calculate_lower_bound():
    arms_prob = [[0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48], [0.5, 0.2, 0.1]]

    n = 10000
    gap_dependent_regret = np.zeros([3,n])
    gap_independent_regret = np.zeros([3,n])
    for i in range(3):
        k = len(arms_prob[i])
        for j in range(1,n):
            dep_reg = 0
            for g in range(len(arms_prob[i])):
                gap = arms_prob[i][0] - arms_prob[i][g]
                if gap > 0 and dep_reg >= 0:
                    dep_reg += log(j*gap**2)/(8*gap) #n*gap * exp(-4*j*gap**2/(k-1))/4 
                    if dep_reg < 0:
                        dep_reg = 0
            gap_dependent_regret[i][j] = dep_reg

            indep_reg = sqrt((k-1)*j/8)*exp(-1/2)/4
            gap_independent_regret[i][j] = indep_reg
    return gap_dependent_regret,gap_independent_regret


gap_dependent_regret,gap_independent_regret = calculate_lower_bound()

print('gap_dependent_regret')
print(gap_dependent_regret)

print('\ngap_independent_regret')
print(gap_independent_regret)


# In[7]:


# Plotting % Cummulative Optimal Arm Pulls Vs Time steps with error bars
def plotCumOptimalArmPulls(horizon,optimal_arm_means_stderr,optimal_arm_pulls_sum,problem,step):
    x = np.arange(horizon)
    ind = [i for i in range(0,horizon,step)]

    for i in range(m_len):
        plt.errorbar(x[ind],optimal_arm_pulls_sum[i,ind], optimal_arm_means_stderr[i,ind,1],
                    linestyle='-', marker='x',capsize=4,capthick=1.5,elinewidth=1.5)
    plt.xlabel('Steps')
    plt.ylabel('% Cummulative Optimal Arm Pulls')
    plt.legend(['UCB',"TS with prior mean=0.5","TS with prior mean=0.2"],loc=0,frameon=False)
    plt.title(arms_prob[problem])
    plt.xlim((0,10000))
    plt.ylim((0,100))
    plt.savefig('CummOptimalArmPulls_'+str(problem)+optimalpulls+'.png',dpi=300)
    plt.show()

    print("optimal_arm_stderr")
    print(optimal_arm_means_stderr[:,[500,2000,5000,8000,9500],1])



# In[8]:


# Plotting % Average Optimal Arm Pulls Vs Time steps with error bars
def plotAvgOptimalArmPulls(horizon,optimal_arm_means_stderr,optimal_arm_pulls_sum,problem,step):
    x = np.arange(horizon)
    ind = [i for i in range(0,horizon,step)]

    for i in range(m_len):
        plt.errorbar(x[ind],optimal_arm_pulls_sum[i,ind], optimal_arm_means_stderr[i,ind,1],
                    linestyle='-', marker='x',capsize=4,capthick=1.5,elinewidth=1.5)
    plt.xlabel('Steps')
    plt.ylabel('% Average Optimal Arm Pulls')
    plt.legend(['UCB',"TS with prior mean=0.5","TS with prior mean=0.2"],loc=0,frameon=False)
    plt.title(arms_prob[problem])
    plt.xlim((0,10000))
    plt.ylim((0,100))
    plt.savefig('AvgOptimalArmPulls_'+str(problem)+optimalpulls+'.png',dpi=300)
    plt.show()

    print("optimal_arm_means_stderr")
    print(optimal_arm_means_stderr[:,[500,2000,5000,8000,9500],1])


# In[9]:


# Plotting Cumulative Regret Vs Time steps with error bars
def plotCummRegret(horizon,regret_means_stderr,regret_per_round_sum,problem,step):
    labels = ["Gap-dependent lower bound","Gap-independent lower bound",'UCB',"TS with prior mean=0.5","TS with prior mean=0.2"]
    x = np.arange(horizon)
    ind = [i for i in range(0,horizon,step)]

    plt.plot(x[ind],gap_dependent_regret[problem][ind],label = labels[0])
    plt.plot(x[ind],gap_independent_regret[problem][ind],label = labels[1])
    
    for i in range(m_len):
        plt.errorbar(x[ind],regret_per_round_sum[i,ind], regret_means_stderr[i,ind,1],label = labels[i+2],
                linestyle='-', marker='x',capsize=4,capthick=1.5,elinewidth=1.5)

    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.legend(loc=0,frameon=False)
    plt.title(arms_prob[problem])
    plt.xlim((0,10000))
#     plt.ylim((0,80))
    plt.savefig('CumulativeRegret_'+str(problem)+optimalpulls+'.png',dpi=300)
    plt.show()

    print("regret_means_stderr")
    print(regret_means_stderr[:,[500,2000,5000,8000,9500],1])
    print("regret_per_round_sum")
    print(regret_per_round_sum[:,[500,2000,5000,8000,9500]])
    print("gap_dependent_regret")
    print(gap_dependent_regret[problem,[500,2000,5000,8000,9500]])
    print("gap_independent_regret")
    print(gap_independent_regret[problem,[500,2000,5000,8000,9500]])


# In[10]:


def plot_arm_distribution(alpha_values,beta_values):
    
    savepoints = [1,1000,5000,10000]
    s = 0
    for p in range(len(alpha_values)):
        
        fig, ax = plt.subplots(figsize=(10, 6))
        m = alpha_values[p] / (alpha_values[p] + beta_values[p])
        print("Mean at time t = "+str(savepoints[s])+" is: ",m)

        x = np.linspace(0, 1, 1000)[1:-1]
        i = 1
        for a, b in zip(alpha_values[p], beta_values[p]):
            plt.plot(x, beta.pdf(x,a,b),
                     label=r'Arm %d : $\alpha=%d,\ \beta=%d$' % (i, a, b))
            i+=1

        plt.ylim(0, 20)
        plt.xticks(np.arange(0, 1.1,0.1))
        plt.xlabel(r'$ \theta $')
        plt.ylabel(r'$p(\theta|\alpha,\beta)$')
        plt.title('Problem '+str(problem+1)+' at t = '+str(savepoints[s]))
        s+=1
        plt.legend(loc=0)
        plt.savefig('Arm_dist_t_'+str(savepoints[s-1])+'_'+str(problem+1)+'_'+optimalpulls+'.png',dpi=300)
        plt.show()
    


# In[11]:


# For Printing table
from IPython.display import HTML, display

def tableIt(data):
    print(pd.DataFrame(data))


# In[12]:


horizon = 10000
replications = 100

arms_prob = [[0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48], [0.5, 0.2, 0.1]]
types = ['UCB','TS M0.5','TS M0.2']
m_len = len(types)
success = [1,1]
failure = [1,4]
optimalpulls = 'Avg'

for problem in range(3): # Repeating for 3 problems
    optimal_arm_pulls_sum = np.zeros([m_len,horizon]) # Storing variables returned by UCB function
    regret_per_round_sum = np.zeros([m_len,horizon])
    optimal_arm_means_stderr = np.zeros([m_len,horizon,2])
    regret_means_stderr = np.zeros([m_len,horizon,2])
    optimal_arm_percentage = np.zeros([m_len])
    total_regret = np.zeros([m_len])
    success_ret = np.zeros([2,4,len(arms_prob[problem])])
    failure_ret = np.zeros([2,4,len(arms_prob[problem])])


    regret_per_round_sum[0,:],regret_means_stderr[0,:,:], optimal_arm_pulls_sum[0,:],optimal_arm_means_stderr[0,:,:],optimal_arm_percentage[0],total_regret[0] = UCB(horizon,replications,arms_prob[problem],1,optimalpulls)
    success_ret[0,:,:],failure_ret[0,:,:],regret_per_round_sum[1,:],regret_means_stderr[1,:,:], optimal_arm_pulls_sum[1,:],optimal_arm_means_stderr[1,:,:],optimal_arm_percentage[1],total_regret[1] = TS(horizon,replications,arms_prob[problem],[success[0]]*len(arms_prob[problem]),[failure[0]]*len(arms_prob[problem]),optimalpulls)
    success_ret[1,:,:],failure_ret[1,:,:],regret_per_round_sum[2,:],regret_means_stderr[2,:,:], optimal_arm_pulls_sum[2,:],optimal_arm_means_stderr[2,:,:],optimal_arm_percentage[2],total_regret[2] = TS(horizon,replications,arms_prob[problem],[success[1]]*len(arms_prob[problem]),[failure[1]]*len(arms_prob[problem]),optimalpulls)
    


    step = 500
    print("\n")
    print("optimal_arm_percentage")
    tableIt(optimal_arm_percentage)
    print("\n")
    print("total_regret")
    tableIt(total_regret)

    # Calling function to plot % Average Optimal Arm Pulls & Commulative regret with error bars
    plotAvgOptimalArmPulls(horizon,optimal_arm_means_stderr,optimal_arm_pulls_sum,problem,step)
    plotCummRegret(horizon,regret_means_stderr,regret_per_round_sum,problem,step)
    plot_arm_distribution(success_ret[0],failure_ret[0])
    plot_arm_distribution(success_ret[1],failure_ret[1])
    


# In[13]:


horizon = 10000
replications = 100

arms_prob = [[0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], [0.5, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48], [0.5, 0.2, 0.1]]
types = ['UCB','TS M0.5','TS M0.2']
m_len = len(types)
success = [1,1]
failure = [1,4]
optimalpulls = 'Cum'

for problem in range(3): # Repeating for 3 problems
    optimal_arm_pulls_sum = np.zeros([m_len,horizon]) # Storing variables returned by UCB function
    regret_per_round_sum = np.zeros([m_len,horizon])
    optimal_arm_means_stderr = np.zeros([m_len,horizon,2])
    regret_means_stderr = np.zeros([m_len,horizon,2])
    optimal_arm_percentage = np.zeros([m_len])
    total_regret = np.zeros([m_len])
    success_ret = np.zeros([2,4,len(arms_prob[problem])])
    failure_ret = np.zeros([2,4,len(arms_prob[problem])])


    regret_per_round_sum[0,:],regret_means_stderr[0,:,:], optimal_arm_pulls_sum[0,:],optimal_arm_means_stderr[0,:,:],optimal_arm_percentage[0],total_regret[0] = UCB(horizon,replications,arms_prob[problem],1,optimalpulls)
    success_ret[0,:,:],failure_ret[0,:,:],regret_per_round_sum[1,:],regret_means_stderr[1,:,:], optimal_arm_pulls_sum[1,:],optimal_arm_means_stderr[1,:,:],optimal_arm_percentage[1],total_regret[1] = TS(horizon,replications,arms_prob[problem],[success[0]]*len(arms_prob[problem]),[failure[0]]*len(arms_prob[problem]),optimalpulls)
    success_ret[1,:,:],failure_ret[1,:,:],regret_per_round_sum[2,:],regret_means_stderr[2,:,:], optimal_arm_pulls_sum[2,:],optimal_arm_means_stderr[2,:,:],optimal_arm_percentage[2],total_regret[2] = TS(horizon,replications,arms_prob[problem],[success[1]]*len(arms_prob[problem]),[failure[1]]*len(arms_prob[problem]),optimalpulls)
    

    step = 500
    print("\n")
    print("optimal_arm_percentage")
    tableIt(optimal_arm_percentage)
    print("\n")
    print("total_regret")
    tableIt(total_regret)


    # Calling function to plot % Optimal Arm Pulls Vs Time steps with error bars
    plotCumOptimalArmPulls(horizon,optimal_arm_means_stderr,optimal_arm_pulls_sum,problem,step)
    plot_arm_distribution(success_ret[0],failure_ret[0])
    plot_arm_distribution(success_ret[1],failure_ret[1])

    


# In[14]:


# Saving variables
dill.dump_session(filename)


# In[15]:


def get_arm_playing_prob(success,failure,trials=10000):

    sample_means = np.zeros([trials,len(success)])
    arm_played_times = np.zeros(trials)
    arm_paying_prob = np.zeros(len(success))
    for i in range(len(success)):
        sample_means[:,i] = np.random.beta(success[i],failure[i],trials)
    
    for j in range(trials):
        arm_played_times[j] = np.argmax(sample_means[j,:])

    for i in range(len(success)):
        arm_paying_prob[i] = (arm_played_times == i).sum()/trials
    
    return arm_paying_prob


# In[31]:


# Plotting Arm Playing Probability Vs Time steps for TS
def plot_arms_playing_prob(horizon,arm_play_Prob,problem):
    x = np.arange(horizon)
    for i in range(len(arm_play_Prob[1,:])):
        plt.plot(x,arm_play_Prob[:,i])
    plt.xlabel('Steps')
    plt.ylabel('Arm Playing Probability per step')
    if len(arm_play_Prob[1,:]) == 3:
        plt.legend(["Arm 1","Arm 2","Arm 3"],loc="best",frameon=False)
    else:
        plt.legend(["Arm 1","Arm 2","Arm 3","Arm 4","Arm 5","Arm 6","Arm 7","Arm 8","Arm 9","Arm 10"],loc=0,frameon=False)
    if problem % 2 == 0:
        plt.title("Problem 3: TS with prior mean = 0.2")
    else:
        plt.title("Problem 3: TS with prior mean = 0.5")

    plt.xlim((0,10000))
    plt.ylim((0,1))
    plt.savefig('Arm_play_prob_'+str(problem)+'.png',dpi=300)
    plt.show()

 



# In[17]:


# Implemention Thompson Sampling for plotting arm playing probability at each time step
def TSForPlottingArmProb(horizon,replications,arms_prob,alpha,beta,optimalpulls,problem):
    optimal_arm = 0
#     optimal_arm_pulls_per_round = np.zeros([horizon,replications])
#     regret_per_round = np.zeros([horizon,replications])
    arm_playing_prob = np.zeros([horizon,len(arms_prob)])
    savepoints = (0,1000,5000,9999)
    success_ret = np.zeros([len(savepoints),len(arms_prob)])
    faliure_ret = np.zeros([len(savepoints),len(arms_prob)])
    
    for r in range(replications):
        arm_pulls = [0]*len(arms_prob)
        success = np.array(alpha)
        failure = np.array(beta)
        t = 0
        s = 0
        while t < horizon:
            if t in savepoints and r == replications-1:
                success_ret[s] = success
                faliure_ret[s] = failure
                s+=1
                
            #Picking arm according to Posterior distribution
            sample_means = [0]*len(arms_prob)
            for i in range(len(arms_prob)):
                sample_means[i] = np.random.beta(success[i],failure[i])
            arm_selected = np.argmax(sample_means)
            
            arm_playing_prob[t] = get_arm_playing_prob(success,failure)
            
            arm_pulls[arm_selected] += 1
            temp = np.random.binomial(1, arms_prob[arm_selected])
            success[arm_selected] += temp
            failure[arm_selected] += 1 - temp

            t+=1
            
#     plot_arms_playing_prob(horizon,arm_playing_prob,problem)
    return arm_playing_prob


# In[18]:


arm_playing_prob1 = TSForPlottingArmProb(10000,1,[0.5, 0.2, 0.1],[1,1,1],[1,1,1],'Avg',1)
plot_arms_playing_prob(10000,arm_playing_prob1,1)

arm_playing_prob2 = TSForPlottingArmProb(10000,1,[0.5, 0.2, 0.1],[1,1,1],[4,4,4],'Avg',2)
plot_arms_playing_prob(10000,arm_playing_prob2,2)



# In[19]:


arm_playing_prob3 = TSForPlottingArmProb(10000,1,[0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],[1]*10,[1]*10,'Avg',3)
plot_arms_playing_prob(10000,arm_playing_prob3,3)

arm_playing_prob4 = TSForPlottingArmProb(10000,1,[0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],[1]*10,[4]*10,'Avg',3)
plot_arms_playing_prob(10000,arm_playing_prob4,4)


# In[20]:


arm_playing_prob5 = TSForPlottingArmProb(10000,1,[0.5, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48],[1]*10,[1]*10,'Avg',3)
plot_arms_playing_prob(10000,arm_playing_prob5,5)

arm_playing_prob6 = TSForPlottingArmProb(10000,1,[0.5, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48],[1]*10,[4]*10,'Avg',3)
plot_arms_playing_prob(10000,arm_playing_prob6,6)


# In[21]:


dill.dump_session(filename)

