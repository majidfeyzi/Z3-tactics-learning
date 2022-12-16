import os
from pathlib import Path

import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class AgentLearningTest:
    """
    Agent that is using to render learned policy.
    """

    def __init__(self, state_size, action_size, policy_file_name):
        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = 0.001  # Alpha

        self.value_model = self.build_model(name='value_model', compile=True)

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

    def act(self, state, n):
        """
        Choose action based on learned policy and do it in environment
        :param state: current state of agent
        :param n: number of actions that we want to return
        :return: selected actions indices
        """

        # Get index of the action with greater value return action index
        actions = self.__argmaxs(self.value_model(np.expand_dims(state, axis=0)), n)
        return actions

    def load_weights(self):
        """
        Load value model and target model network weights from a file.
        File has been generated before with target model weights.
        :return:
        """
        self.value_model.load_weights(self.save_weights_path)

    @staticmethod
    def __argmaxs(ls, n):
        return np.flip(np.argsort(ls[0])[-n:])
