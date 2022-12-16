import datetime
import operator
import os
from pathlib import Path

import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from env.utils.probes_reader import ProbesReader


class Agent:
    # Print results every 100 step
    PRINT_INTERVAL = 100

    def __init__(self, state_size, action_size, policy_file_name):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = 64

        self.memory_size = 500000  # Buffer size
        self.memory = np.zeros((self.memory_size, 2 * state_size + 3))
        self.memory_counter = 0

        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999

        self.gamma = 0.9  # Discount factor
        self.learning_rate = 0.001  # Alpha

        self.target_update_interval = 10  # Update target network every 10 time step
        self.training_interval = 10  # Train network every 10 time step
        self.learning_start = 1000  # Start learning from time step 1000

        self.value_model = self.build_model(name='value_model', compile=True)
        self.target_model = self.build_model(name='target_model', compile=False)

        self.save_weights_path = Path(str(Path(__file__).parent.resolve()) + os.sep + "shared" + os.sep + policy_file_name)

    def build_model(self, name, compile):
        """
        Generate models of neural network.
        If value model is generating it must be compiled.
        If target model is generating, there is no need to compile.
        Model has 3 layer each with 64 node and with activation 'relu'
        :param name: name of model
        :param compile: compile flag that specify model compilation
        :return: created model based on type (value model or target model)
        """
        state = Input(shape=(self.state_size,), name='states')
        x = state
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        state_action_value = Dense(self.action_size, activation='linear')(x)
        model = Model(state, state_action_value, name=name)
        if compile:
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        """
        Choose action based on epsilon greedy algorithm and do it in environment
        :param state: current state of agent
        :return: selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)

        # Get index of the action with greater value return action index
        action = np.argmax(self.value_model(np.expand_dims(state, axis=0)))
        return action

    def memorize(self, state, action, reward, next_state, done):
        """
        Keep agent achieved data
        :param state: current state of agent
        :param action: action that is done in state
        :param reward: reward that achieved by doing action in environment
        :param next_state: next state that agent is in it after doing action
        :param done: is episode finished or not
        :return:
        """
        index = self.memory_counter % self.memory_size  # Which memory index to save?
        self.memory[index] = np.array(list(state) + [action, reward] + list(next_state) + [done])
        self.memory_counter += 1

    def get_batch(self):
        """
        Get batch data from memory
        What if samples are still less than memory but more than batch size
        :return: saved data in memory
        """
        random_indices = np.random.randint(0, min(self.memory_size, self.memory_counter), size=self.batch_size)
        return self.memory[random_indices]

    def update_weights(self):
        """
        Update target model weights using value model weights
        :return:
        """
        self.target_model.set_weights(self.value_model.get_weights())

    def update_epsilon(self):
        """
        Decrease epsilon until arrive to minimum value
        :return:
        """
        epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = epsilon if epsilon > self.epsilon_min else self.epsilon_min

    def train(self, total_steps):
        """
        Train neural network
        :return:
        """
        if total_steps <= self.learning_start:
            return

        if self.memory_counter < self.batch_size:
            return

        # Train network in every training interval
        if total_steps % self.training_interval == 0:
            data = self.get_batch()
            states = data[:, :self.state_size]
            actions = data[:, self.state_size].astype('int')
            rewards = data[:, self.state_size + 1]
            next_states = data[:, self.state_size + 2:-1]
            dones = data[:, -1]

            x = states
            target_value = np.max(self.target_model.predict(next_states), axis=1) * (
                    1 - dones)  # To make a 0 value for target
            target_value = target_value * self.gamma + rewards

            y = self.value_model.predict(x)  # Copy value to y
            y[np.arange(self.batch_size), actions] = target_value  # Replace y's with ys

            self.value_model.train_on_batch(x, y)

        # Update target network and epsilon in every target update interval
        if total_steps % self.target_update_interval == 0:
            self.update_weights()
            self.update_epsilon()

    def save_weights(self):
        """
        Save target model network weights in a file
        :return:
        """
        self.target_model.save_weights(self.save_weights_path)

    def load_weights(self):
        """
        Load value model and target model network weights from a file.
        File has been generated before with target model weights.
        :return:
        """
        self.target_model.load_weights(self.save_weights_path)
        self.value_model.load_weights(self.save_weights_path)

    def start_training(self, env, episodes, win_reward):
        """
        Start leaning in given environment and save learned policy and then return average of rewards
        """

        rewards = list()
        mean_rewards = list()
        steps = list()
        mean_steps = list()
        rlimits = list()
        mean_rlimits = list()
        probes = list()
        mean_probes = list()

        # Start training
        for episode in range(1, episodes + 1):
            state = env.reset()
            done = False
            total_rewards = 0
            total_steps = 0
            total_rlimits = 0
            total_probes = [0 for i in ProbesReader.REWARD_PROBES_RANGE]
            while not done:
                action = self.act(state)
                next_state, reward, done, rlimit, step_probes = env.step(action)
                self.memorize(state, action, reward, next_state, done)
                state = next_state
                total_rewards += reward
                total_steps += 1
                total_rlimits += rlimit
                total_probes = list(map(operator.add, total_probes, step_probes[:ProbesReader.REWARD_PROBES_RANGE.stop]))
                self.train(sum(steps))

            rewards.append(total_rewards)
            mean_score = np.mean(rewards[-100:])  # Get average of last 100 rewards
            mean_rewards.append(mean_score)
            steps.append(total_steps)
            mean_step = np.mean(steps[-50:])  # Get average of last 50 steps
            mean_steps.append(mean_step)
            rlimits.append(total_rlimits)
            mean_rlimit = np.mean(rlimits[-100:])  # Get average of last 100 rlimits
            mean_rlimits.append(mean_rlimit)
            probes.append(total_probes)
            mean_probe = list(map(np.mean, zip(*probes[-100:])))  # Get average of last 100 probes
            mean_probes.append(mean_probe)

            if episode % Agent.PRINT_INTERVAL == 0:
                print("{}: DQN Agent: Episode {}-{}: Mean score = {} in {} episodes"
                      .format(datetime.datetime.now(),
                              abs(episode - Agent.PRINT_INTERVAL) + 1, episode, mean_score, Agent.PRINT_INTERVAL))
        # Save policy
        self.save_weights()

        return mean_rewards, mean_steps, mean_rlimits, mean_probes

    def __str__(self):
        """ Serialize agent object as json string """
        return str({"state_size": self.state_size, "action_size": self.action_size}).replace("\'", "\"")
