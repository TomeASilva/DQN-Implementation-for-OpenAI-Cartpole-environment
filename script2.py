import tensorflow as tf
from tensorflow import keras
from keras import layers, initializers, regularizers
import gym
import numpy as np
import time
import random
import os
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ReplayBuffer():
    def __init__(self, size=int(10e5)):
        self.size = size
        self.memory = deque(maxlen=size)

    def save_sample(self, sample):
        """
        Inputs:
        sample - list [s, a, r, s', done]

        """

        self.memory.append(sample)

    def draw_batch(self, minibatch_size=64):
        """
        Inputs:
        minibatch_size -- scalar
        Outputs:
        _states -- an array size [minibatch_size, state space dimension]
        _actions -- an array size [minibatch_size]
        _rewards -- an array size [minibatch_size]
        _next_states -- an array size [minibatch_size, state space dimension]
        _dones -- an array with ones and zeros [minibatch_size]
        """
        batch = random.sample(self.memory, minibatch_size)

        _states, _actions, _rewards, _next_states, _dones = zip(*batch)
        _states = np.asarray(_states).reshape([minibatch_size, -1])
        _actions = np.asarray(_actions)
        _rewards = np.asarray(_rewards)
        _dones = np.asarray(_dones, dtype=int)
        _next_states = np.asarray(_next_states).reshape([minibatch_size, -1])

        return _states, _actions, _rewards, _next_states, _dones


def build_networks(network_name, num_layers, activations, layer_sizes, input_layer):
    """
    Creates network and performs forward propagation

    Inputs:
    network_name-- string that will be used in the tensor flow graph as the name of the ANN
    num_layers -- scalar, input layer not included
    input_layer -- tensor_flow placeholder: will contain the training examples to fed into the ANN
    layer_sizes -- list with the size of each layer not including input layer
    activations -- list with the name of the activation function for each layer

    num_layers, layer_sizes, activations list will have the same size

    Outputs:
    network -- tensor
    nn_variables -- list with tf.Variable objects created in the network: weights and biases
    """

    assert(num_layers == (len(activations)) and num_layers ==
           len(layer_sizes)),  "Check the number of activations and layer_sizes provided "

    with tf.variable_scope(network_name):
        network = layers.Dense(
            layer_sizes[0], activation=activations[0], kernel_initializer=layers.initializers.glorot_normal(), name="Layer_1")(input_layer)

        for layer in range(1, num_layers):

            network = layers.Dense(units=layer_sizes[layer], kernel_initializer=layers.initializers.glorot_normal(), activation=activations[layer], name=(
                "Layer_" + str(layer + 1)))(network)

        nn_variable_names = tf.trainable_variables(scope=network_name)

    return nn_variable_names, network


class ComputationGraph:
    """Will build all the operations necessary for the DQN Agent"""

    def __init__(self, args):
        self.tau = args["soft_update_tau"]

        self.st_placeholder = tf.placeholder(tf.float32, shape=[
                                             None, args["state_space_size"]], name='state')  # Sample of Sates
        self.st_1_placeholder = tf.placeholder(tf.float32, shape=[
                                               None, args["state_space_size"]], name="state_1")  # Sample of States
        self.rewards_placeholder = tf.placeholder(
            tf.float32, shape=[None, ], name="rewards")
        self.actions_placeholder = tf.placeholder(
            tf.int32, shape=[None, ], name="actions")
        self.dones_placeholder = tf.placeholder(
            tf.float32, shape=[None, ], name="dones")

        self.nn_variables, self.Q_hat = build_networks(
            "DQN", args["num_layers"], args["activations"], args["layers_sizes"], self.st_placeholder)  # DQN Network

        self.target_nn_variables, self.Q_target_net = build_networks(
            "Target_DQN", args["num_layers"], args["activations"], args["layers_sizes"], self.st_1_placeholder)  # DQN target Network

        # Create summaries for weitghts and biases
        summaries = []
        for variable in self.nn_variables:

            varname = variable.name.replace(
                "kernel:0", "W").replace("bias:0", "b")
            summaries.append(tf.summary.histogram(varname, variable))

        self.parameters_summary = tf.summary.merge(summaries)
        ###

        # Compute the predictions for Q(s,a)
        with tf.variable_scope("Q_Training"):
            encoded_action_tensor = tf.one_hot(
                self.actions_placeholder, args["action_space_size"])

            self.Q_training = tf.reduce_sum(tf.multiply(
                self.Q_hat, encoded_action_tensor), axis=1)

       # Find Q(s', a')  where a' = argmax Q(s',a)
        with tf.variable_scope("a_maxQ"):
            self.a_maxQ = tf.reduce_max(self.Q_target_net, axis=1)

        # Computes the bellman operator
        with tf.variable_scope("yi"):
            self.yi = self.rewards_placeholder + \
                args["gamma"] * self.a_maxQ * (1 - self.dones_placeholder)

        # Computes the loss function and creates a summary for the lost function
        with tf.variable_scope("Loss_MSE"):

            self.loss = tf.losses.mean_squared_error(self.yi, self.Q_training)
            # Create summary for loss function
            self.summary_loss = tf.summary.scalar("Loss", self.loss)
        # Train opearation
        with tf.variable_scope("train_DQN"):

            self.train_op = tf.train.AdamOptimizer(learning_rate=args["learning_rate"]).minimize(
                self.loss, var_list=self.nn_variables)

        # Opeartion that performs the update to the target network
        with tf.variable_scope("update_target_network"):
            self.update_target_network = [tf.assign(target_variable, target_variable * self.tau +
                                                    (1 - self.tau) * variable) for target_variable, variable in zip(self.target_nn_variables, self.nn_variables)]


class DQN_Agent(ComputationGraph):
    """ Implements all the methods necessary for DQN_agent, as long a DQN Agent that 
    will have acess to the computationGraph

    Arguments: 
    args: a dictionary with Problem parameters with expected keys:
            "num_layers": 3,
            "layers_sizes": [32, 64, 2],
            "activations": ["relu", "relu", "linear"],
            "gamma": 0.9,
            "state_space_size": env.observation_space.shape[0],
            "action_space_size": env.action_space.n,
            "learning_rate": 0.001,
            "epsilon_start": 1,
            "minibatch_size": 64,
            "C_step_target_up": 1,
            "max_steps_episode": 200,
            "max_number_episodes": 50,
            "soft_update_tau": 8e-2,
            "epsilon_stop": 0.01,
            "epsilon_decay": 0.999}

    sess: a Tensor Flow Session
    """

    def __init__(self, args, session):
        super().__init__(args)
        self.action_space_size = args["action_space_size"]
        self.state_space_size = args["state_space_size"]
        self.epsilon_start = args["epsilon_start"]
        self.epsilon_stop = args["epsilon_stop"]
        self.epsilon_decay = args["epsilon_decay"]
        self.session = session
        self.steps = 0
        self.epsilon = self.epsilon_start

    def take_action(self, state):
        """"Computes the function that will get the highest reward according to parameters of ANN

        Arguments: 
        state-- array size [, state_space_dimension]: a state
        Outputs
        action --  int: action 1 or 0 
        """

        state = state.reshape(-1, self.state_space_size)
        if random.uniform(0, 1) >= self.epsilon:

            Q_hat = self.session.run([self.Q_hat], feed_dict={
                self.st_placeholder: state})
            action = np.argmax(Q_hat[0])

        else:
            action = random.randint(0, self.action_space_size - 1)

        return action

    def train_batch(self, _states, _actions, _rewards, _next_state, _dones, summary_writer):
        """ Performs one step of gradient descet over a batch 
        Arguments:
        _states -- an array size [minibatch_size, state space dimension]
        _actions -- an array size [minibatch_size]
        _rewards -- an array size [minibatch_size]
        _next_state -- an array size [minibatch_size, state space dimension]
        _dones -- an array with ones and zeros [minibatch_size]
        summary_writer--  tf.summary.FileWriter object: will serve to add the loss and NN parameters summaries

        Ouputs:
        _loss -- a float with the value of the loss function over this batch of trainning samples
        """

        _, summary_loss, summary_parameters, _loss = self.session.run([self.train_op, self.summary_loss, self.parameters_summary, self.loss], feed_dict={
            self.st_placeholder: _states, self.st_1_placeholder: _next_state, self.actions_placeholder: _actions, self.rewards_placeholder: _rewards, self.dones_placeholder: _dones})
        summary_writer.add_summary(summary_loss, self.steps)
        summary_writer.add_summary(summary_parameters, self.steps)
        self.steps += 1

        if self.epsilon > self.epsilon_stop:
            self.epsilon *= self.epsilon_decay

        return _loss

    def build_summaries(self):
        """ Creates a summary for storing the total reward at every episode and to store the total predicted reward

        Outputs:

        _placeholder_ep_reward-- tensorflow placeholder:to be fed with values of episode rewards
        placeholder_ep_predicted_reward-- tensorflow placeholder: to be fed with the predictd reward of each episode
        _reward_summaries-- op that represents 2 merged scalar summaries
        """

        _placeholder_ep_reward = tf.placeholder(
            tf.float32, name="Episode_Reward")

        summary1 = tf.summary.scalar(
            _placeholder_ep_reward.name, _placeholder_ep_reward)

        placeholder_ep_predicted_reward = tf.placeholder(
            tf.float32, name="Episode_Predicted_Reward")

        summary2 = tf.summary.scalar(
            placeholder_ep_predicted_reward.name, placeholder_ep_predicted_reward)

        _reward_summaries = tf.summary.merge([summary1, summary2])

        return _placeholder_ep_reward, placeholder_ep_predicted_reward, _reward_summaries

    def evaluate_Q_sa(self, state, action):
        """ Evaluates the Q(s,a) for a given state and action

        Arguments: 
        state: array, size [1, state_space_size]
        action: array, size[1]
        Output:
        _Qsa[0]: a float 

        """

        state = state.reshape(-1, self.state_space_size)

        action = np.asarray(action).reshape(-1)
        _Qsa = self.session.run([self.Q_training], feed_dict={
            self.st_placeholder: state, self.actions_placeholder: action})  # returns a list with an array of Q values and the type of values in the array

        return float(_Qsa[0])


if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    args = {"num_layers": 3,
            "layers_sizes": [32, 64, 2],
            "activations": ["relu", "relu", "linear"],
            "gamma": 0.9,
            "state_space_size": env.observation_space.shape[0],
            "action_space_size": env.action_space.n,
            "learning_rate": 0.001,
            "epsilon_start": 1,
            "minibatch_size": 64,
            "C_step_target_up": 1,
            "max_steps_episode": 200,
            "max_number_episodes": 50,
            "soft_update_tau": 8e-2,
            "epsilon_stop": 0.01,
            "epsilon_decay": 0.999}

    ######DQN_Loop######

    cart_replay_buffer = ReplayBuffer()
    sess = tf.Session()
    agent_DQN = DQN_Agent(args, sess)
    losses = []
    episode_rewards = []
    max_step = args["max_steps_episode"]
    num_episodes = args["max_number_episodes"]
    sess.run(tf.global_variables_initializer())
    subdir = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    logdir = "./summary/" + subdir
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(sess.graph)

    placeholder_ep_reward, placeholder_predicted_reward, reward_summaries = agent_DQN.build_summaries()

    for i in range(num_episodes):
        state = env.reset()
        step = 0
        done = False
        episode_comulative_reward = 0

        while (step < max_step) and (not done):

            # Choose action

            action = agent_DQN.take_action(state)

            # Find the predicted Episode Reward

            if step == 0:
                Qsa = agent_DQN.evaluate_Q_sa(state, action)

            next_state, reward, done, _ = env.step(
                action)  # We still need to take the action
            step += 1
            episode_comulative_reward += reward

            # save training example

            cart_replay_buffer.save_sample(
                (state, action, reward, next_state, done))

            state = next_state

            # Run a Step of gradient Descent

            if len(cart_replay_buffer.memory) > args["minibatch_size"]:

                states, actions, rewards, next_states, dones = cart_replay_buffer.draw_batch(
                    args["minibatch_size"])

                loss = agent_DQN.train_batch(
                    states, actions, rewards, next_states, dones, writer)
                losses.append(loss)

                if agent_DQN.steps % args["C_step_target_up"] == 0:
                    _ = sess.run(agent_DQN.update_target_network)

        # End of Episode

        episode_rewards.append(episode_comulative_reward)

        # add summary of reward
        str_summary = sess.run(reward_summaries, feed_dict={
                               placeholder_ep_reward: episode_comulative_reward, placeholder_predicted_reward: Qsa})

        writer.add_summary(str_summary, i)

        if i % 10 == 0:
            print(
                f"Episode {i}/{num_episodes}, Number of steps:{step} , Reward sum {episode_comulative_reward}")
