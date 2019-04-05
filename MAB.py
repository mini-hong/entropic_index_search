#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.optimize import minimize


# In[2]:


def np_max_single_q(q_logit, q=1.):
        q_logit = np.reshape(q_logit,[-1,])
        max_q_logit = np.max(q_logit)
        safe_q_logit = q_logit - max_q_logit
        if q==1.:
            maxq = np.log(np.sum(np.exp(safe_q_logit))) + max_q_logit
            pq = np.exp(safe_q_logit)
            pq = pq/np.sum(pq)
        else:
            obj = lambda x: -np.sum(safe_q_logit*x) - 1/(q-1.)*(1.-np.sum(x**q))
            const = ({'type':'eq', 'fun':lambda x:np.sum(x)-1.})
            bnds = [(0.,1.) for i in range(safe_q_logit.shape[0])]
            res = minimize(obj, x0=np.ones_like(safe_q_logit)/safe_q_logit.shape[0], constraints=const, bounds=bnds)
            maxq = -res.fun+max_q_logit
            pq = res.x
            pq = pq/np.sum(pq)
        return maxq, pq


# In[3]:


def np_max_q(q_logits,q=1):
    maxq_list = []
    pq_list = []
    for q_logit in q_logits:
        maxq, pq = np_max_single_q(q_logit,q=q)
        pq_list.append(pq)
        maxq_list.append(maxq)
    return np.array(maxq_list), np.array(pq_list)


# In[4]:

ps = []
for i in range(10):
    p = np.random.random_sample(50)
    #p.sort()
    #p = p[::-1]
    ps.append(p)


# In[ ]:


class MDP:
    def __init__(self, p, N):
        self.prob = p[:N]
    def step(self, a):
        rand = np.random.sample()
        return 0 if self.prob[a] > rand else -1


# In[ ]:


class Agent:
    def __init__(self, N, q, T):
        self.N = N
        self.logit = np.zeros(N)
        self.q = q
        if q != 1:
            self.eta = np.sqrt((1-q)*N**q*T/(1*(N**(1-q)-1)))
        else:
            self.eta = np.sqrt(N**q*T/np.log(N))
    '''
    def np_max_q(self, q_logits,q=1):
        maxq_list = []
        pq_list = []
        for q_logit in q_logits:
            maxq, pq = np_max_single_q(q_logit,q=q)
            pq_list.append(pq)
            maxq_list.append(maxq)
        return np.array(maxq_list), np.array(pq_list)
    '''
    def update(self, a, r, prob):
        self.logit[a] += r / prob
    def get_action(self):
        maxq, pq = np_max_single_q(self.logit/self.eta, self.q)
        a = np.random.choice(self.N, p=pq)
        return a, pq[a]


# In[ ]:


Ns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30,35,40,45,50]
qs = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
all_rets = []
best_q = []
T = 5000
X = []
Y = []
for N in Ns:
    rets = np.zeros(len(qs))
    for i in range(len(qs)):
        agent = Agent(N,qs[i],T)
        print("****************")
        for p_ in ps:
            mdp = MDP(p_, N)
            ret = 0
            for j in range(T):
                a, prob = agent.get_action()
                r = mdp.step(a)
                ret += r
                agent.update(a, r, prob)
            rets[i] += ret
            print('N : {}, q : {}, eta : {}, ret : {}'.format(N,qs[i],agent.eta,rets[i]))
        all_rets.append(rets[i])
        X.append(N)
        Y.append(qs[i])
    best_q.append(qs[np.argmax(rets)])
    print("==========================")
    print("N : {}, best q : {}".format(N, qs[np.argmax(rets)]))
    print("==========================")
    with open('results.txt','a') as f:
        f.write('{},{}\n'.format(N,qs[np.argmax(rets)]))
        


# In[ ]:


from matplotlib import pyplot as plt
plt.figure()
plt.scatter(X, Y, c=all_rets, s=20, cmap=plt.cm.rainbow,alpha=0.5)
plt.colorbar(label='color')
plt.figure()
plt.plot(Ns, best_q)
plt.show()


# In[ ]:





# In[ ]:




