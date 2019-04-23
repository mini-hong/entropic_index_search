import numpy as np
from scipy.optimize import minimize

def np_max_single_q(q_logit, old_x, q=1.):
    q_logit = np.reshape(q_logit, [-1, ])
    max_q_logit = np.max(q_logit)
    safe_q_logit = q_logit - max_q_logit
    if q == 1.:
        maxq = np.log(np.sum(np.exp(safe_q_logit))) + max_q_logit
        pq = np.exp(safe_q_logit)
        pq = pq / np.sum(pq)
    else:
        obj = lambda x: -np.sum(safe_q_logit * x) - 1 / (q - 1.) * (1. - np.sum(x ** q))
        const = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.})
        bnds = [(0., 1.) for i in range(safe_q_logit.shape[0])]
        res = minimize(obj,
                       x0=old_x,
                       constraints=const,
                       bounds=bnds)
        maxq = -res.fun + max_q_logit
        pq = res.x
        pq = pq / np.sum(pq)
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
for i in range(50):
    p = np.random.random_sample(50)
    #p.sort()
    #p = p[::-1]
    ps.append(p)


class MAB:
    def __init__(self, p, N):
        self.prob = p[:N]
    def step(self, a):
        rand = np.random.sample()
        return 1 if self.prob[a] > rand else 0

class Agent:
    def __init__(self, N, q, T):
        self.N = N
        self.logit = np.zeros(N)
        self.q = q
        self.old_p = np.ones(N)
        if q != 1:
            self.eta = np.sqrt((1-q)*N**q*T/(q*(N**(1-q)-1)))
        else:
            self.eta = np.sqrt(N*T/np.log(N))
    def update(self, a, r, prob):
        self.logit[a] += r / prob
    def get_action(self):
        maxq, pq = np_max_single_q(self.logit/self.eta, self.old_p, self.q)
        a = np.random.choice(self.N, p=pq)
        self.old_p = pq
        return a, pq[a]

class buffer:
    def __init__(self, window_size):
        self.returns = []
        self.size = 0
        self.window_size = window_size
    def put(self, ret):
        if self.size < self.window_size:
            self.returns.append(ret)
            self.size += 1
        else:
            self.returns[self.size % self.window_size] = ret
            self.size += 1
        return np.mean(self.returns)

def mov_avg(rews, window, target):
    avg_rews = np.zeros(len(rews))
    for i in range(len(rews)):
        if i < window:
            avg_rews[i] = np.mean(rews[:i+1])
        else:
            avg_rews[i] = np.mean(rews[i+1-window:i+1])
    return avg_rews / target

def mov_avg_all(rews_all, window, targets):
    avg_rews_all = []
    for i in range(len(rews_all)):
        avg_rews_all.append(mov_avg(rews_all[i], window, targets[i]))
    return np.mean(avg_rews_all, axis=0)


Ns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
#Ns = [2, 3, 4, 5, 6]
#Ns = [7, 8, 9, 10]
#Ns = [12, 15]
#Ns = [20, 30]
qs = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
T = 5000
best_qs = []
#p = np.random.random_sample(50)
for N in Ns:
    end_steps = []
    total_rews = []
    for qi in range(len(qs)):
        rews = np.zeros(shape=(len(ps), T))
        targs = np.zeros(len(ps))
        for pi in range(len(ps)):
            mab = MAB(ps[pi], N)
            agent = Agent(N, qs[qi], T)
            targs[pi] = np.max(ps[pi][:N])
            for t in range(T):
                a, prob = agent.get_action()
                r = mab.step(a)
                agent.update(a, r, prob)
                rews[pi][t] = r
        avg_rews = mov_avg_all(rews, 100, targs)
        avg_rews[avg_rews > 1.] = 1.
        total_rews.append(avg_rews)
        for i in range(50, len(avg_rews)):
            if avg_rews[i] > 0.95:
                print('N : {}, q : {}, ended {}'.format(N, qs[qi], i))
                end_steps.append(i)
                with open('mab/end_step.txt', 'a') as f:
                    f.write('{} {} {}\n'.format(N, qs[qi], i))
                break
            if i == len(avg_rews) - 1:
                print('N : {}, q : {}, ended {}'.format(N, qs[qi], len(avg_rews)))
                end_steps.append(T)
                with open('mab/end_step.txt', 'a') as f:
                    f.write('{} {} {}\n'.format(N, qs[qi], len(avg_rews)))
    with open('mab/mab_N_{}'.format(N), 'w') as f:
        for j in range(T):
            f.write('{} '.format(j))
            for i in range(len(total_rews)):
                f.write('{} '.format(total_rews[i][j]))
            f.write('\n')
    best_q = qs[np.argmin(end_steps)]
    best_qs.append(best_q)
    print('=====================')
    print('N : {}, best q : {}'.format(N, best_q))
    print('=====================')
    with open('mab/results.txt', 'a') as f:
        f.write('{} {} \n'.format(N, best_q))

from matplotlib import pyplot as plt
#plt.figure()
#plt.scatter(X, Y, c=all_rets, s=20, cmap=plt.cm.rainbow,alpha=0.5)
#plt.colorbar(label='color')
plt.figure()
plt.plot(Ns, best_qs)
plt.show()



