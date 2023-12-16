'''
This assignment done by
Peng Zhou Z5443641

'''
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class World(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.R = np.zeros(self.x*self.y)
        self.agentPos = 0
    
    def idx2xy(self,idx):
        x = int(idx / self.y)
        y = idx % self.y
        return x, y

    def xy2idx(self,x,y):
        return x*self.y + y

    def resetAgent(self, pos):
        self.agentPos = int(pos)

    def setReward(self, x, y, r):
        goalState = self.xy2idx(x, y)
        self.R[goalState] = r

    def getState(self):
        return self.agentPos

    def getReward(self):
        return self.R[self.agentPos]

    def getNumOfStates(self):
        return self.x*self.y
 
    def getNumOfActions(self):
        return 4

    def move(self,id):
        x_, y_ = self.idx2xy(self.agentPos)
        tmpX = x_
        tmpY = y_
        if id == 0: # move DOWN
            tmpX += 1
        elif id == 1: # move UP
            tmpX -= 1
        elif id == 2: # move RIGHT
            tmpY += 1
        elif id == 3: # move LEFT
            tmpY -= 1
        else:
            print("ERROR: Unknown action")

        if self.validMove(tmpX, tmpY):
            self.agentPos = self.xy2idx(tmpX,tmpY)

    def validMove(self,x,y):
        valid = True
        if x < 0 or x >= self.x:
            valid = False
        if y < 0 or y >= self.y:
            valid = False
        return valid

class Agent:
    def __init__(self, world, action_selection_method='epsilon-greedy', learning_algorithm='q-learning'):
        self.world = world
        self.numofStates = self.world.getNumOfStates()
        self.numofActions = self.world.getNumOfActions()
        self.Q = np.loadtxt("initial_Q_values.txt")
        self.random_numbers = np.loadtxt("random_numbers.txt")
        self.alpha = 0
        self.gamma = 0.4
        self.epsilon = 0.25
        self.tau = 0.1
        self.counter=0
        self.accumulated_reward=[]
        self.num_steps=[]
        
        # store the action selection method and learning algorithm
        self.action_selection_method = action_selection_method
        self.learning_algorithm = learning_algorithm

    def actionSelection(self, state):
        if self.action_selection_method == 'epsilon-greedy':
            # epsilon-greedy logic
            if self.counter >= len(self.random_numbers):
                raise ValueError
            rnd = self.random_numbers[self.counter]
            self.counter += 1
            
            if rnd <= self.epsilon:
                action_rnd = self.random_numbers[self.counter]
                self.counter += 1
                if action_rnd <= 0.25:
                    action = 0
                elif action_rnd <= 0.5:
                    action = 1
                elif action_rnd <= 0.75:
                    action = 2
                else:
                    action = 3
            else:
                action = np.argmax(self.Q[state,:])
                
            return action
        elif self.action_selection_method == 'softmax':
            # softmax logic
            # Get the random number from the pre-loaded list based on a counter
            random_number = self.random_numbers[self.counter]
            # Increment the counter
            self.counter += 1
            
            # Compute the softmax probabilities
            q_values = self.Q[state, :]
            probabilities = np.exp(q_values / self.tau) / np.sum(np.exp(q_values / self.tau))
            
            # Compute cumulative probabilities
            cumulative_probabilities = np.cumsum(probabilities)
            
            # Find the action using np.searchsorted
            action = np.searchsorted(cumulative_probabilities, random_number)
            return action

    def sarastrain(self, iter):
        for itr in range(iter):
            state = 0
            self.world.resetAgent(state)
            a = self.actionSelection(state)
            
            total_reward = 0
            steps = 0
            
            episode = True
            while episode:
                self.world.move(a)
                reward = self.world.getReward()
                next_state = self.world.getState()
                next_a = self.actionSelection(next_state)

 
                # Update Q values
                self.Q[state, a] += self.alpha * (reward + self.gamma * self.Q[next_state, next_a] \
                                                       - self.Q[state, a])

                
                total_reward += reward
                steps += 1
                
                state = next_state
                a = next_a
                
                if reward == 1.0:
                    self.Q[next_state, :] = 0
                    episode = False
                    
            self.accumulated_reward.append(int(total_reward))
            self.num_steps.append(steps)
        print(self.Q)
    
    def qtrain(self, iter):
        for itr in range(iter):
            state = 0
            self.world.resetAgent(state)
            
            
            total_reward = 0
            steps = 0
            
            episode = True
            while episode:
                a = self.actionSelection(state)
                self.world.move(a)
                reward = self.world.getReward()
                next_state = self.world.getState()
                
                
                # Update Q values
                self.Q[state, a] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state, :]) \
                                                       - self.Q[state, a])

                
                total_reward += reward
                steps += 1
                
                state = next_state
                
                if reward == 1.0:
                    self.Q[next_state, :] = 0
                    episode = False
                    
            self.accumulated_reward.append(int(total_reward))
            self.num_steps.append(steps)
        print(self.Q)

    def train(self, iter):
        if self.learning_algorithm == 'q-learning':
            self.qtrain(iter)
        else:
            self.sarastrain(iter)
            
    def plotQValue(self, title=''):
        plt.rcParams.update({'font.size': 11})
        plt.imshow(self.Q, cmap='Oranges', interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.title(title + ' Q values')
        plt.xlabel('Actions')
        plt.ylabel('States')
        plt.xticks(np.arange(4), ('Down', 'Up', 'Right', 'Left'))
        plt.yticks(np.arange(self.numofStates), np.arange(self.numofStates))
        plt.show()
        
    def plot_accumulatedReward(self, title=''):
        plt.rcParams.update({'font.size': 11})
        plt.plot(self.accumulated_reward)
        plt.title(title + ' Accumulated Reward')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.show()

    def plot_numofSteps(self, title=''):
        plt.rcParams.update({'font.size': 11})
        plt.plot(self.num_steps)
        plt.title(title + ' Steps per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Steps')
        plt.show()
        
if __name__ == "__main__":
    world = World(3, 4)
    world.setReward(2, 3, 1.0)  # Goal state
    world.setReward(1, 1, -1.0) # Fear region

    # Softmax - Q-learning
    learner = Agent(world, action_selection_method='softmax', learning_algorithm='q-learning')
    learner.train(1000)
    learner.plotQValue(title='Softmax - Q-learning')
    learner.plot_accumulatedReward(title='Softmax - Q-learning')
    learner.plot_numofSteps(title='Softmax - Q-learning')
    
'''  
    # Epsilon-greedy - Sarsa
    learner = Agent(world, action_selection_method='epsilon-greedy', learning_algorithm='sarsa')
    learner.train(1000)
    learner.plotQValue(title='Epsilon-greedy - Sarsa')
    learner.plot_accumulatedReward(title='Epsilon-greedy - Sarsa')
    learner.plot_numofSteps(title='Epsilon-greedy - Sarsa')

    # Softmax - Sarsa
    learner = Agent(world, action_selection_method='softmax', learning_algorithm='sarsa')
    learner.train(1000)
    learner.plotQValue(title='Softmax - Sarsa')
    learner.plot_accumulatedReward(title='Softmax - Sarsa')
    learner.plot_numofSteps(title='Softmax - Sarsa')

    # Epsilon-greedy - Q-learning
    learner = Agent(world, action_selection_method='epsilon-greedy', learning_algorithm='q-learning')
    learner.train(1000)
    learner.plotQValue(title='Epsilon-greedy - Q-learning')
    learner.plot_accumulatedReward(title='Epsilon-greedy - Q-learning')
    learner.plot_numofSteps(title='Epsilon-greedy - Q-learning')
'''
    

  

