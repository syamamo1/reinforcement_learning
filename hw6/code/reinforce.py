import os
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense 

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        
        # TODO: Define network parameters and optimizer
        # num_inputs, state_size --> num_inputs, P(num_actions)
        self.network = Sequential()
        self.network.add(Dense(64, input_dim = (self.state_size), activation='relu')) 
        self.network.add(Dense(64, input_dim = (64), activation='relu')) 
        self.network.add(Dense(self.num_actions, input_dim = (64), activation='softmax')) 

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        # TODO: implement this 
        probs = self.network(states)
        return probs

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a Tensorflow scalar
        """
        # TODO: implement this
        # Hint: Use gather_nd to get the probability of each action that was actually taken in the episode.
        
        inds = np.column_stack((np.arange(len(actions)), actions))
        probs = tf.gather_nd(self.call(states), inds)

        loss = -tf.reduce_sum(tf.math.log(probs)*discounted_rewards)

        return loss

