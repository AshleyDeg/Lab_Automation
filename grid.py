# Imports 
import numpy as np
import pygame
import sys
from agents import Agent, Adversary 

pygame.init()

class Grid:
    # Define colors
    BG = (200, 200, 200) #background color 
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 128) 
    RED = (255, 0, 0)
    

    def __init__(self, n = 10):
        self.grid_size  = n
        self.n_states = self.grid_size**2 # Number of states in the grid world (nxn grid)
        self.cell_size = 50 if self.grid_size <15 else 25
        self.screen_size = self.grid_size * self.cell_size

        print("\nGrid size: ", n, "x" , n, "\n")

        # Load and scale the images
        self.goal_image = pygame.image.load('goal.png') # Load the image 
        self.goal_image = pygame.transform.scale(self.goal_image, (self.cell_size, self.cell_size))  # Scale the image to your needed size

        self.obstacles = [] 
        self.red_wins = np.zeros(11)
        self.green_wins = np.zeros(11)
        self.no_wins = np.zeros(11)
    

    def draw_grid(self):
        for x in range(0, self.screen_size, self.cell_size):
            for y in range(0, self.screen_size, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.WHITE, rect, 1)

    def clear_screen(self):
        # Clear screen
        self.screen.fill(self.BG)

    def state_to_coords(self, state):
        # Convert state to grid coordinates
        row = state // self.grid_size
        col = state % self.grid_size
        return col * self.cell_size, row * self.cell_size

    def draw_goal(self, goal_state):
        # Draw goal
        goal_x, goal_y = self.state_to_coords(goal_state)
        self.screen.blit(self.goal_image, (goal_x, goal_y)) 

    def draw_agent(self, agent):
        # Draw agents
        agent_x, agent_y = self.state_to_coords(agent.current_state)
        pygame.draw.rect(self.screen, agent.color, (agent_x, agent_y, self.cell_size, self.cell_size))

    def generate_obstacles(self, n_obs=None):
        # Generate obstacles 
        x=0
        if n_obs == None: 
            n_obs = self.grid_size
        while x<n_obs:
            rand = np.random.randint(0, self.n_states-1)
            if rand not in self.obstacles: 
                self.obstacles.append(rand)
                x += 1
    
    def draw_obstacles(self):
        # Draw obstacles
        for obstacle in self.obstacles:
            obs_x, obs_y = self.state_to_coords(obstacle)
            pygame.draw.rect(self.screen, self.BLACK, (obs_x, obs_y, self.cell_size, self.cell_size))
    
    def draw_banner(self, text):
        # Draw a banner to understand who wins the game 
        # Create a banner at the top of the screen

        # Create a font object
        font = pygame.font.Font('freesansbold.ttf', 30)
        
        # Create a text surface object
        text = font.render(text, True, self.WHITE, self.BLUE)
        
        # Create a rectangular object for the text surface object
        textRect = text.get_rect()
        textRect.center = (self.screen_size // 2, self.screen_size // 2)
        self.screen.blit(text, textRect)

    def draw_game(self, agent, adv, iter, n_games, delay=150): 
        # Function to draw a game 
        running = True
        counter_viz = 0
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))

        # Clock object 
        self.clock = pygame.time.Clock()
        self.FPS = 10

        agent.random_starting_position(self.obstacles)
        adv.random_starting_position(self.obstacles, agent) 
        move_counter = 0 
        move_limit = self.grid_size**2

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Choose best action based on the learned Q-table
            action = np.argmax(agent.Q_table[agent.current_state])
            pygame.display.set_caption("Q-learning Grid World, iter: "+ str(iter) + ", game: " + str(counter_viz+1))

            if iter == 1: 
                # In the first iteration adversary moves randomly
                action_adv = adv.generate_action(1)
            else:
                # In other iterations choose according to policy 
                action_adv = adv.generate_action(0)

            agent.simulate_action(action)
            adv.simulate_action(action_adv)

            # If the next state is an obstacle, the agent doesn't move 
            agent.check_obstacles(self.obstacles)
            adv.check_obstacles(self.obstacles)

            move_counter += 1 

            # Clear screen
            self.screen.fill(self.BG) 
            # Draw grid
            self.draw_grid()
            # Draw obstacles
            self.draw_obstacles()
            # Draw goal state
            self.draw_goal(agent.goal_state)

            # Draw agents
            self.draw_agent(agent)
            self.draw_agent(adv)

            # Update display
            pygame.display.flip()

            # Move to next state
            agent.current_state = agent.next_state
            adv.current_state = adv.next_state

            # Add delay for visualization
            pygame.time.delay(delay)

            # If the game finishes reset to a random state
            # The game finishes if:
            # - green reaches goal state and green wins 
            # - red catches green and red wins 
            # - after x moves no agent has reached its goal and red wins 
            if agent.current_state == agent.goal_state or adv.current_state == agent.current_state or move_counter > move_limit:
                # Clear screen
                self.screen.fill(self.BG) 
                # Draw grid
                self.draw_grid()
                # Draw obstacles
                self.draw_obstacles()
                
                if adv.current_state == agent.current_state:
                    # Adversary wins 
                    # Red catches green 
                    self.draw_agent(adv)
                    self.draw_goal(agent.goal_state)
                    # Update display
                    pygame.display.flip()
                    pygame.time.delay(200) #300
                    self.draw_banner("RED WINS")
                    pygame.display.update()
                    # Add delay for visualization
                    pygame.time.delay(500) #1000
                    # New random starting position 
                    agent.random_starting_position(self.obstacles)
                    adv.random_starting_position(self.obstacles, agent) 
                    counter_viz += 1
                    self.red_wins[iter-1] += 1/n_games    
                elif agent.current_state == agent.goal_state: 
                    # Agent wins 
                    # Goal reached 
                    self.draw_agent(agent)
                    self.draw_agent(adv)
                    # Update display
                    pygame.display.flip()
                    pygame.time.delay(200) #300
                    self.draw_banner("GREEN WINS")
                    pygame.display.update()
                    # Add delay for visualization
                    pygame.time.delay(500) #1000 
                    # New random starting position 
                    agent.random_starting_position(self.obstacles)
                    adv.random_starting_position(self.obstacles, agent) 
                    counter_viz += 1
                    self.green_wins[iter-1] += 1/n_games
                elif move_counter > move_limit: 
                    # Red wins 
                    self.draw_agent(agent)
                    self.draw_agent(adv)
                    self.draw_goal(agent.goal_state) 
                    # Update display
                    pygame.display.flip()
                    pygame.time.delay(200) #300
                    self.draw_banner("RED WINS")
                    pygame.display.update()
                    # Add delay for visualization
                    pygame.time.delay(500) #1000
                    # New random starting position 
                    agent.random_starting_position(self.obstacles)
                    adv.random_starting_position(self.obstacles, agent) 
                    counter_viz += 1
                    self.red_wins[iter-1] += 1/n_games
                    self.no_wins[iter-1] += 1/n_games

                move_counter = 0 

            # Exit the loop after n games            
            if counter_viz >= n_games:
                # Show n games 
                running = False

        if iter == 11: 
            # After the loop, quit pygame
            pygame.display.quit() 