# Imports 
import numpy as np
import argparse
from agents import Agent, Adversary
from grid import Grid 
import matplotlib.pyplot as plt

# Setup argparse to handle command-line arguments
parser = argparse.ArgumentParser(description="Q-learning Grid World with Obstacles. Independent Learning. \nTwo agents: green has to move towards a fixed goal, the other (red) has to chase and catch green. ")
parser.add_argument('--n', type=int, default=10, help='type: INT - an integer n to specify grid size (nxn), default n = 10')
parser.add_argument('--g', type=int, default=10, help='type: INT - number of games displayed per iteration, default g = 10')
parser.add_argument('--p', type=bool, default=False, help='type: BOOL - print Q-tables before and after each training, default FALSE')
parser.add_argument('--d', type=int, default=150, help='type: INT - delay for game visualization, default d = 150')
parser.add_argument('--o', type=int, help='type: INT - number of obstacles in grid, default = n')
args = parser.parse_args()

# Define the environment
viz_Q = args.p
n = args.n 
n_obs = args.o
if n < 4:
    print("Grid size cannot be smaller than 4. Setting grid size to 4.")
    n = 4
n_games = args.g
delay = args.d

# Create agents and set goal for green 
agent = Agent(n) 
goal_state = n*n-1  # Goal state (bottom-right corner)
adv = Adversary(n)

# Define grid and obstacles (states where the agent cannot go)
grid = Grid(n)
grid.generate_obstacles(n_obs)
print("Obstacles: ", grid.obstacles)
print("Epochs: ", agent.epochs)
print("Games displayed for each training phase: ", n_games)


# Independent learning 
# - in odd iterations only green is learning and updating its Q-table
# - in even iterations only red is learning and updating its Q-table 
for iter in range(1, 11): 
    if iter % 2: 
        print("\n\033[1m Training phase for GREEN, iteration: ", iter, "\033[0m")
        if viz_Q:             
            print("\nQ-tables before training")
            print("Agent's Q-table: ")
            agent.print_Q_table()
            print("\nAdversary's Q-table: ")
            adv.print_Q_table()
        agent.Q_learning(grid.obstacles, adv, goal_state, iter)
        if viz_Q: 
            print("\nQ-tables after training.")
            print("Agent's Q-table: ")
            agent.print_Q_table()
            print("\nAdversary's Q-table: ")
            adv.print_Q_table()
        grid.draw_game(agent, adv, iter, n_games, delay)
    else:
        print("\033[1m Training phase for RED, iteration: ", iter, "\033[0m")
        if viz_Q:             
            print("\nQ-tables before training")
            print("Agent's Q-table: ")
            agent.print_Q_table()
            print("\nAdversary's Q-table: ")
            adv.print_Q_table()
        adv.Q_learning(grid.obstacles, agent, iter)
        if viz_Q: 
            print("\nQ-tables after training.")
            print("Agent's Q-table: ")
            agent.print_Q_table()
            print("\nAdversary's Q-table: ")
            adv.print_Q_table()
        grid.draw_game(agent, adv, iter, n_games, delay)

print("Wins for red (adv): ", grid.red_wins)
print("Wins for green (agent): ", grid.green_wins)
print("Nobody won: ", grid.no_wins)


# Plot and save statistics on number of wins for red and green throughout trainings 
figure, axis = plt.subplots(1, 2, figsize = (12,6))

# Wins for red
x1 = np.arange(10)
axis[0].plot(x1, grid.red_wins[0:10], '-ro')
axis[0].set_title("Wins for red")
axis[0].set_xlabel("Iter")
axis[0].set_ylabel("Wins")

# Wins for green  
axis[1].plot(x1, grid.green_wins[0:10], '-go')
axis[1].set_title("Wins for green")
axis[1].set_xlabel("Iter")
axis[1].set_ylabel("Wins")

# Save the plot to a file
plt.savefig('wins.png')