# Multi Agent Reinforcement Learning (MARL)
This project implements the Q-learning algorithm in a *multi-agent environment*. Training in a multi-agent environment brings forward a challenge: the environment is not stationary, therefore we adopt an **independent learning** approach. Two competing agents (red and green) move in a grid world: green has to move towards a fixed goal and red has to catch green.  

**Q-learning algorithm**
1) Generate an action according to any policy and collect data $(x(t), u(t), r(t), x(t+1))$
2) Update the value of the pair $(x(t), u(t))$ by temporal difference $$Q_{t+1}(x(t), u(t)) = Q_{t}(x(t), u(t)) + \beta(t)[r(t) + \alpha \max_{u'} Q_t (x(t+1), u') - Q(x(t), u(t))] $$
3) Leave unchanged the values of other states $Q_{t+1}(x, u) = Q_t(x,u)$ $\forall(x,u) \!= (x(t), u(t))$

The Q-learning algorithm is adapted for a multi-agent environment using an independent learning approach: one agent improves its policy by applying a RL algorithm, while the other follows a fixed policy. The learning agent is changes periodically.
 
*Dependencies*: Python version used for this project is Python 3.12.4. The libraries imported and used are: numpy, matplotlib and pygame for visualization and plotting purposes. 

# Implementation 
In agents.py two classes are defined. One for the agent and one for its adversary. In grid.py the class Grid is defined to visualize games.  We follow a grid world implementation. The good agent (green) has to move towards a fixed goal, and the adversary's (red agent) goal is to catch the other anger. The environment is a grid with nxn possible states with fixed obstacles where each agent can move according to one of four actions (left, right, up, down). 

In independent_learning.py command line arguments can be passed to set the following parameters:
- grid size nxn (default n = 10)
- number of games displayed per iteration (default g = 10)
- number of obstacles in grid (default = n)
- delay for game visualization (default d = 150ms)
- whether of not you want to print Q-tables before and after each training, default FALSE

To run the code: 
```
python independent_learning.py
```

# Results 
Please find results in the corresponding folder. 

Results include:
- a video and GIF of a couple of games recorded after 10 iterations of training
- reward plots for each iteration of training (including a moving average plot for better interpretation)
- a plot of the number of wins for each agent over 500 games

From the results we can see how the agents learn to move across the grid towards their goals. Particularly, the red agent learns where the green's goal is and tried to *beat* green by getting there first. So, since in each episode the starting position is random, the agent that wins is the closest one to the fixed goal and as can be expected, each agent wins about 50% of the time. 

