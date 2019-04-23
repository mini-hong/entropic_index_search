import numpy as np
from scipy.optimize import minimize

def np_max_single_q(q_logit, old_x, q=1.):
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
            res = minimize(obj,
                            x0=old_x,
                            constraints=const,
                            bounds=bnds)
            maxq = -res.fun+max_q_logit
            pq = res.x
            pq = pq/np.sum(pq)
        return maxq, pq

def np_max_q(q_logits,q=1):
    maxq_list = []
    pq_list = []
    for q_logit in q_logits:
        maxq, pq = np_max_single_q(q_logit,q=q)
        pq_list.append(pq)
        maxq_list.append(maxq)
    return np.array(maxq_list), np.array(pq_list)



class Corridor:
    def __init__(self, length):
        self.current_state = 1
        self.length = length
        self.goal = length-1
        self.reward = np.zeros(length)
        self.reward[0] = 0.01
        self.reward[self.goal] = 1.
    def step(self, a):
        if a == 0 and self.current_state > 0:
            self.current_state -= 1
        elif a == 1 and self.current_state < self.goal:
            self.current_state += 1
        #if self.current_state == self.goal:
            #print('goal!')
        return self.current_state, self.reward[self.current_state], self.current_state == 0 or self.current_state == self.goal
    def reset(self):
        self.current_state = 1
        return self.current_state

class Agent:
    def __init__(self, n_a, length, q, T):
        self.logit = np.zeros((length, n_a))
        self.q = q
        if q != 1:
            self.eta = np.sqrt((1-q)*n_a**q*T/(q*(n_a**(1-q)-1)))
        else:
            self.eta = np.sqrt(n_a*T/np.log(n_a))
        self.N = n_a
        self.p = np.ones((length, n_a))/n_a
        self.length = length
    def update(self, s, a, val, prob):
        self.logit[s][a] += val / prob
    def update_p(self):
        for i in range(self.length):
            maxq, self.p[i] = np_max_single_q(self.logit[i]/self.eta, self.p[i], self.q)
    def get_action(self, s):
        #maxq, pq = np_max_single_q(self.logit[s]/self.eta, self.old_p, self.q)
        a = np.random.choice(self.N, p=self.p[s])
        return a, self.p[s][a]

def mov_avg(rets, window):
    avg_ret = np.zeros(len(rets))
    for i in range(len(rets)):
        if i < window:
            avg_ret[i] = np.mean(rets[:i+1])
        else:
            avg_ret[i] = np.mean(rets[i+1-window:i+1])
    return avg_ret

def mov_avg_all(rews_all, window):
    avg_rews_all = []
    for i in range(len(rews_all)):
        avg_rews_all.append(mov_avg(rews_all[i], window))
    return np.mean(avg_rews_all, axis=0)

Ns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
qs = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
max_ep_len = 100
H = 1000
length = 5
best_qs = []
num_trial = 50
for N in Ns:
    env = Corridor(length)
    end_steps = []
    total_rets = []
    for qi in range(len(qs)):
        rets = np.zeros(shape=(num_trial, H))
        for trial in range(num_trial):
            agent = Agent(N, length, qs[qi], H)
            for h in range(H):
                s, d = env.reset(), False
                states, actions, probs = [], [], []
                ret = 0
                for t in range(max_ep_len):
                    a, prob = agent.get_action(s)
                    s_, r, d = env.step(a)
                    ret += r
                    actions.append(a)
                    probs.append(prob)
                    states.append(s)
                    s = s_
                    if d:
                        break
                for t in range(len(states)):
                    agent.update(states[t], actions[t], ret, probs[t])
                agent.update_p()
                rets[trial][h] = ret
        avg_ret = mov_avg_all(rets, 10)
        total_rets.append(avg_ret)
        for i in range(100 ,len(avg_ret)):
            if avg_ret[i] > 0.95:
                print('N : {}, q : {}, ended {}'.format(N, qs[qi], i))
                end_steps.append(i)
                break
            if i == len(avg_ret)-1:
                print('N : {}, q : {}, ended {}'.format(N, qs[qi], len(avg_ret)))
                end_steps.append(H)
    with open('mdp/mdp_N_{}'.format(N), 'w') as f:
        for i in range(H):
            if (i+1) % 10 == 0:
                f.write('{} '.format(i))
                for j in range(len(total_rets)):
                    f.write('{} '.format(total_rets[j][i]))
                f.write('\n')
    best_q = qs[np.argmin(end_steps)]
    best_qs.append(best_q)
    print('=====================')
    print('N : {}, best q : {}'.format(N, best_q))
    print('=====================')
    with open('mdp/results.txt', 'a') as f:
        f.write('{} {} \n'.format(N, best_q))

from matplotlib import pyplot as plt
plt.figure()
plt.plot(Ns, best_qs)
plt.show()