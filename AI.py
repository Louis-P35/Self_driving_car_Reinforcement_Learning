import random

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np

import gym
from gym import spaces


class CustomEnv(gym.Env):
    def __init__(self, nbAction, nbInput, valuesScale):
        self.nbAction = nbAction
        self.nbInput = nbInput
        self.angle = 0

        # Définir l'espace d'observation (observation_space)
        low = np.zeros((self.nbInput,))
        high = np.ones((self.nbInput,)) * valuesScale
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Définir l'espace d'action (action_space)
        self.action_space = spaces.Discrete(self.nbAction)

        # Initialisation de l'état courant (current_state)
        self.current_state = None

    def reset(self):
        # Réinitialisation de l'état courant (current_state)
        self.current_state = np.zeros((self.nbInput,))
        return self.current_state

    def step(self, action, isDead, hasPassCheckpoint, angle):
        # Vérification de l'action
        assert self.action_space.contains(action)

        # Calcul de la récompense (reward) en fonction de l'état courant et de l'action choisie
        reward = 0
        # if isDead:
        #     reward = -1
        # elif hasPassCheckpoint:
        #     reward = 1

        # print(angle, ' ', self.angle)
        # if angle > self.angle:
        #     reward = 1
        # else:
        #     reward = -1
        # self.angle = angle

        # Debug test
        # if action == 1:
        #     reward = 1
        # else:
        #     reward = -1

        # Mise à jour de l'état courant (current_state) en fonction de l'action choisie
        self.current_state[action % self.nbInput] = 1 # Not used in my case

        # Vérification si le jeu est fini (done)
        done = False

        # Retourner l'observation (observation), la récompense (reward), l'indicateur de fin de jeu (done) et des informations supplémentaires (info)
        return self.current_state, reward, done, {}



"""
Replay buffer for the agent
Handle storage of the state action reward and state transition tuples
discrete mean that the actions can be an integer set of values
inputShape: Input shape of the environment (sensors, position, velocity, etc)
"""
class ReplayBuffer(object):
    def __init__(self, maxSize, inputShape, nbActions, discrete=False):
        # Maximum size of the memory
        self.mem_size = maxSize
        self.discrete = discrete

        # Store the states from the environment
        self.state_memory = np.zeros((self.mem_size, inputShape))
        # Keep track of the new state we get after taking an action
        self.new_state_memory = np.zeros((self.mem_size, inputShape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, nbActions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        # Terminal flags from the environment because when we transition into the terminal state
        # we don't want to take into account the reward at the next state
        # When the episode is over you don't want to take into account the reward from the next state because
        # that is a reward from another episode
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.mem_counter = 0


    # Storing the state action reward, new state and done flag into our memory
    def storeTransition(self, state, action, reward, newState, done):
        # First available memory
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = newState
        self.reward_memory[index] = reward
        # The done flag is False when the episode isn't over
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            discretActions = np.zeros(self.action_memory.shape[1])
            discretActions[action] = 1.0
            self.action_memory[index] = discretActions
        else:
            self.action_memory[index] = action
        self.mem_counter += 1


    # batch_size because we don't want to sample the entirety of the memory that is too slow
    def sample_buffer(self, batch_size):
        # Ensure we do not read beyond the array size
        max_mem = min(self.mem_counter, self.mem_size)
        # Random batch of that memory
        batch = np.random.choice(max_mem, batch_size)

        # Acces sub arrays of the memory
        states = self.state_memory[batch]
        newStates = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, newStates, terminal





"""
Deep Q Learning model
inputDims: input dimentions
fc1_dim: fully connected layer 1 dimentions
fc2_dim: fully connected layer 2 dimentions
"""
def build_dqn(learningRate, nbActions, inputDims, fc1_dim, fc2_dim):
    # Construct a sequence of layers
    model = Sequential([
        # Dense layer
        Dense(fc1_dim, input_shape=(inputDims, )),
        # Activation layer
        Activation('relu'),
        # Second dense layer
        Dense(fc2_dim),
        # Second activation layer
        Activation('relu'),
        # Last dense layer
        Dense(nbActions)])

    # Compile the model
    # mse = mean squared error
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='mse')

    return model


class Agent(object):
    # epsilon_dec: Epsilon decrement
    def __init__(self, alpha, gamma, nbActions, epsilon, batchSize, inputDims, epsilon_dec=0.996,
                 epsilon_end=0.01, mem_size=1000000, fileName='dqn_model.h5'):
        # Set of available actions (useful to select a random action)
        self.action_space = [i for i in range(nbActions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batchSize
        self.model_file = fileName

        self.memory = ReplayBuffer(mem_size, inputDims, nbActions, discrete=True)

        # Q network for evaluating our actions
        # fc = 256, can play around with thoses values
        self.q_eval = build_dqn(alpha, nbActions, inputDims, 256, 256)


    # Interfacing with our memory to save a new state transition
    # Interface fonction
    def remember(self, state, action, reward, new_state, done):
        self.memory.storeTransition(state, action, reward, new_state, done)

    def choose_action(self, state):
        # Re-shape it (add an axis to the vector)
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # Take a greedy action
            # Pass the state through the network, get the value of all the actions for that particular state
            # and select the action that has the maximum value
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action


    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        # Feed the set of states through the model
        # Calculate the value of the current state as well as the nest states
        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1) * done

        # Perform the fit and pass in the states and the Q target
        # Passes the batch of states through the network, calculate their values based on the current estimate
        # and then compare those to Q target
        # q_traget is a current reward + best possible action for the next state, it is kinda like the delta between
        # where we want to be and where we are
        _ = self.q_eval.fit(state, q_target, verbose=0)

        # Handle epsilon
        # Decrement epsilon over time until it reach epsilon_min
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def saveModel(self):
        self.q_eval.save(self.model_file)

    def loadModel(self, file_name):
        self.q_eval = load_model(file_name)