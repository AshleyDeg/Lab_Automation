# Multi Agent Reinforcement Learning (MARL)
This project implements the Q-learning algorithm in a *multi-agent environment*. Training in a multi-agent environment brings forward a challenge: the environment is not stationary, therefore we adopt an **independent learning** approach. Two competing agents (red and green) move in a grid world: green has to move toward a fixed goal and red has to catch green.  


**Q-learning algorithm**
1) Generate an action according to any policy and collect data $(x(t), u(t), r(t), x(t+1))$
2) Update the value of the pair $(x(t), u(t))$ by temporal difference $$Q_{t+1}(x(t), u(t)) = Q_{t}(x(t), u(t)) + \beta(t)[r(t) + \alpha \max_{u'} Q_t (x(t+1), u') - Q(x(t), u(t))] $$
3) Leave unchanged the values of other states $Q_{t+1}(x, u) = Q_t(x,u)$ $\forall(x,u) \!= (x(t), u(t))$

## Dependancies 

# Results 
