# Imports 
import numpy as np
import matplotlib.pyplot as plt
import math 


class Agent:
    n_actions = 4 # Number of possible actions (up, down, left, right)
    def __init__(self, n=10, learning_rate = 0.8, discount_factor = 0.95):
        print("\ncreating agent..\n")
        # Define parameters 
        self.n = n # Grid size will be nxn
        self.n_states = self.n**2 # Number of states in the grid world (nxn grid)
        self.current_state = 0
        self.next_state = 0 
        self.goal_state = self.n*self.n-1  # Goal state (bottom-right corner)
        self.Q_table = np.zeros((self.n_states, self.n_actions)) #Q table for Q-learning 

         # Define parameters for learning process 
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epochs = n**3 if n>10 else 2000
        self.counter_th = 10*n^2
        self.color = (24, 161, 67)
        self.name = "GREEN"
        self.rewards = np.zeros(self.epochs)

    def random_starting_position(self, obstacles):
        # Start from a random state (different from goal states and not within obstacles) 
        while True:
            rand = np.random.randint(0, self.n_states-1)
            if (rand != self.goal_state) and (rand not in obstacles):
                self.current_state = rand 
                #print("Starting position: ", current_state) 
                break

    def simulate_action(self, action):
        # Simulate the action on the grid (move to the next state)
        # Define possible actions (up, down, left, right)
        if action == 0:  # Up
            self.next_state = self.current_state - self.n if self.current_state >= self.n else self.current_state
        elif action == 1:  # Down
            self.next_state = self.current_state + self.n if self.current_state < (self.n**2-self.n) else self.current_state
        elif action == 2:  # Left
            self.next_state = self.current_state - 1 if self.current_state % self.n != 0 else self.current_state
        elif action == 3:  # Right
            self.next_state = self.current_state + 1 if (self.current_state + 1) % self.n != 0 else self.current_state 

    def explore_and_exploit(self, epoch):
        # Choose action with epsilon-greedy strategy
        # epsilon decreases with epoch 
        eps = 1/(2+epoch)
        if np.random.rand() < eps:
            action = np.random.randint(0, self.n_actions)  # Explore
        else:
            action = np.argmax(self.Q_table[self.current_state])  # Exploit
        return action 
    
    def generate_action(self, random_policy = False):
        # Choose action according to Q-table or a random policy if flag is True
        if random_policy:
            action = np.random.randint(0, self.n_actions) 
        else:
            action = np.argmax(self.Q_table[self.current_state])
        return action 

    
    def check_obstacles(self, obstacles):
        # If the next state is an obstacle, the agent doesn't move, i.e. the agent stays in the current state
        is_obs = False
        if self.next_state in obstacles:
            self.next_state = self.current_state
            is_obs = True
        return is_obs 
    
    def plot_rewards(self, iter):
        # Function to plot rewards per episode 
        window_size = 100 # Window size used to compute moving average 
 
        i = 0
        moving_averages = [] # Initialize an empty list to store moving averages
        
        while i < len(self.rewards) - window_size + 1:
            # Store elements from i to i+window_size in list to get the current window
            window = self.rewards[i : i + window_size]
        
            # Calculate the average of current window
            window_average = round(sum(window) / window_size, 2)
            
            # Store the average of current window
            moving_averages.append(window_average)
            
            # Shift window to right by one position
            i += 1

        print("\nsaving reward plots..")

        # Initialise the subplot function using number of rows and columns
        figure, axis = plt.subplots(1, 2, figsize = (16,5))

        # Reward per epoch
        x1 = np.arange(self.epochs)
        axis[0].plot(x1, self.rewards, '-o')
        axis[0].set_title("Rewards, after training phase for " + self.name + "\niter: " + str(iter) + " - grid size = " + str(self.n) +"x" + str(self.n))
        axis[0].set_xlabel("Epoch")
        axis[0].set_ylabel("Reward")

        # Moving average 
        x2 = np.arange(len(moving_averages))
        axis[1].plot(x2, moving_averages, '-o')
        axis[1].set_title("Moving average for rewards, \nwindow size: " + str(window_size))
        axis[1].set_xlabel("Epoch")
        axis[1].set_ylabel("Reward")
        
        # Save the plot to image 
        plt.savefig('rewards iter ' + str(iter)+' agent ' + self.name + '.png') 
        #plt.show()

    def Q_learning(self, obstacles, adv, goal_state, iter):
        # Implementation of the Q-learning algorithm 
        self.goal_state = goal_state
        for epoch in range(self.epochs):

            # Start from a random state (different from goal states and not within obstacles) 
            self.random_starting_position(obstacles)
            adv.random_starting_position(obstacles, self)            

            counter = 0
            self.rewards[epoch] = 0

            while (self.current_state != self.goal_state or self.current_state != adv.current_state) and counter < self.counter_th:
                # Choose action with epsilon-greedy strategy
                action = self.explore_and_exploit(epoch)

                # Simulate the environment (move to the next state)
                # Possible actions are: up, down, left, right
                self.simulate_action(action)

                if iter == 1:
                    #Random policy for adversary on first training
                    action_adv = adv.generate_action(True)
                    adv.simulate_action(action_adv)
                else:
                    #Choose action for adversary according to its Q-table
                    action_adv = adv.generate_action()
                    adv.simulate_action(action_adv)

                # If the next state is an obstacle, the agent doesn't move 
                self.check_obstacles(obstacles)
                adv.check_obstacles(obstacles)

                # Define a simple reward function (1 if the goal state is reached, -1 if eaten by adv, -0.04 otherwise)
                reward = 1 if self.next_state == self.goal_state else -1 if (self.next_state == adv.next_state) else -0.04
                self.rewards[epoch] += reward 

                # Update Q-value using the Q-learning update rule
                self.Q_table[self.current_state, action] += self.learning_rate * \
                    (reward + self.discount_factor * np.max(self.Q_table[self.next_state]) - self.Q_table[self.current_state, action])

                self.current_state = self.next_state  # Move to the next state
                adv.current_state = adv.next_state  # Move to the next state
                counter += 1

            # 
            if (epoch+1) % 1000 == 0: 
                print("Epoch: ", epoch)

        # Plot reward for each training phase
        self.plot_rewards(iter)

            
    def print_Q_table(self):
        # After training, the Q-table represents the learned Q-values
        print("Q-table shape (|states|x|actions|): ", self.Q_table.shape)
        print("Actions:\n \033[1m     up          down        left        right \033[0m")
        print(self.Q_table)


class Adversary(Agent):
    # Class for adversary (inherits from Agent) 
    def __init__(self, n=10, learning_rate = 0.8, discount_factor = 0.95): 
        super().__init__(n, learning_rate, discount_factor) 
        self.color = (255, 0, 0)
        self.name = "RED"

    def random_starting_position(self, obstacles, agent):
        # Start from a random state (different from goal states and not within obstacles) 
        while True:
            rand = np.random.randint(0, self.n_states-1)
            if (rand != agent.goal_state) and (rand not in obstacles) and (rand != agent.current_state):
                self.current_state = rand 
                #print("Starting position: ", current_state) 
                break
    
    def Q_learning(self, obstacles, agent, iter):
        for epoch in range(self.epochs):

            # Start from a random state (not goal states and not within obstacles) 
            agent.random_starting_position(obstacles)
            self.random_starting_position(obstacles, agent)

            counter = 0
            self.rewards[epoch] = 0

            while (self.current_state != self.goal_state or agent.current_state != agent.goal_state) and counter < self.counter_th:
                # Choose action with epsilon-greedy strategy
                action = self.explore_and_exploit(epoch)
                # Simulate the environment (move to the next state), possible actions (up, down, left, right)
                self.simulate_action(action)

                #Choose action for adversary according to its Q-table 
                action_agent = agent.generate_action()
                agent.simulate_action(action_agent)

                # If the next state is an obstacle, the agent doesn't move 
                is_obs = self.check_obstacles(obstacles)
                agent.check_obstacles(obstacles)
                self.goal_state = agent.next_state

                # Define a simple reward function (1 if the goal state is reached, -1 for obstacles, -0.04 otherwise)
                
                #reward = 1 if self.next_state == self.goal_state else -1 if self.next_state == current_state and current_state in self.obstacles else -0.04

                # Define a simple reward function (1 if the goal state is reached, -1 if agent reaches its goal, -0.04 otherwise)
                reward = 1 if self.next_state == self.goal_state else -1 if agent.next_state == agent.goal_state else 0
                reward = 1 if (self.next_state == self.goal_state or counter == self.counter_th-3)  else -1 if agent.next_state == agent.goal_state else 0
                self.rewards[epoch] += reward
                
                
                # Update Q-value using the Q-learning update rule
                self.Q_table[self.current_state, action] += self.learning_rate * \
                    (reward + self.discount_factor * np.max(self.Q_table[self.next_state]) - self.Q_table[self.current_state, action])

                self.current_state = self.next_state  # Move to the next state
                agent.current_state = agent.next_state  # Move to the next state
                counter += 1

            # 
            if (epoch+1) % 1000 == 0: 
                print("Epoch: ", epoch)
        
        # Plot reward for each training phase
        self.plot_rewards(iter)