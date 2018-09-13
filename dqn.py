from random import random

import gym
import numpy as np
import tensorflow as tf


class GymDQNLearner:
    def __init__(self):
        self.saving_path = './saved_models/dqn/'
        self.epochs = 10000
        self.gamma = .9
        self.epsilon = 1.
        self.train_per_epoch = 1
        self.n_generating_trajectories_per_epoch = 1
        self.max_memory_size = 5000
        self.max_trajectory_length = 1000
        self.batch_size = 64

        self.env = gym.make('CartPole-v0').env
        self.state_embedding_size = self.env.observation_space.shape[0]
        self.number_of_actions = self.env.action_space.n
        print(self.state_embedding_size, self.number_of_actions)
        self.layer_units = [512, 128, self.number_of_actions]
        self.layer_activations = ['relu', 'relu', None]
        self.initialize_experience_replay_memory()

        self.create_model()
        self.load()

    def initialize_experience_replay_memory(self):
        self.experience_replay_memory = np.array([])

    def get_epsilon(self, i):
        # alpha = 1e-5
        # return 1.0 - (i / np.sqrt(1 + alpha * (i ** 2))) * np.sqrt(alpha)
        # return 1.0 - float(i) / epochs
        return max(0.1, self.epsilon * (0.9989 ** i))
        # return 1

    def get_state_weights(self, trajectory):
        total_reward = len(trajectory)
        return [total_reward for i in range(total_reward)]

    def add_to_memory(self, trajectory):
        weights = self.get_state_weights(trajectory)
        for (from_state, action, reward, to_state, done, q_value), weight in zip(trajectory, weights):
            if self.experience_replay_memory.shape[0] >= self.max_memory_size:
                # self.experience_replay_memory = \
                #     np.delete(self.experience_replay_memory, np.random.randint(0, self.experience_replay_memory.shape[0]))
                # self.experience_replay_memory = self.experience_replay_memory[1:]
                min_element = np.argmin([exp['weight'] for exp in self.experience_replay_memory])
                self.experience_replay_memory = \
                    np.delete(self.experience_replay_memory, min_element)
            self.experience_replay_memory = np.append(self.experience_replay_memory, [
                {'from': from_state, 'action': action,
                 'reward': reward, 'done': done,
                 'to': to_state,
                 'q_value': q_value,
                 'weight': weight}])

    def sample_from_memory(self):
        if self.experience_replay_memory.shape[0] > 1:
            weights = np.array([exp['weight'] for exp in self.experience_replay_memory])
            p = weights / np.sum(weights)
            return np.random.choice(self.experience_replay_memory,
                                    np.min([self.batch_size, self.experience_replay_memory.shape[0]]), p=p)
        else:
            return self.experience_replay_memory

    def create_multilayer_dense(self, layer_input, layer_units, layer_activations, scope, reuse_vars=None):
        with tf.variable_scope(scope, reuse=reuse_vars):
            last_layer = None
            for i, (layer_size, activation) in enumerate(zip(layer_units, layer_activations)):
                if i == 0:
                    last_layer = tf.layers.dense(layer_input, layer_size, activation)
                else:
                    last_layer = tf.layers.dense(last_layer, layer_size, activation)
        return last_layer

    def create_model(self):
        self.inputs = tf.placeholder(np.float32, [None, self.state_embedding_size], name='inputs')
        self.outputs = tf.placeholder(np.float32, [None, self.number_of_actions], name='outputs')

        self.output_layer = \
            self.create_multilayer_dense(self.inputs, self.layer_units, self.layer_activations, 'q_func')

        self.loss = tf.losses.mean_squared_error(self.outputs, self.output_layer, scope='q_func')

        trainable_variables = tf.trainable_variables('q_func')
        self.train_op = tf.train.AdamOptimizer(1e-3, name='optimizer').minimize(self.loss, var_list=trainable_variables)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    def get_action(self, epoch, q_value):
        if random() < self.get_epsilon(epoch):
            action = self.env.action_space.sample()
        else:
            action = np.argmax(q_value)
        return action

    def generate_new_trajectories(self, epoch):
        for _ in range(self.n_generating_trajectories_per_epoch):
            observation = self.env.reset()
            done = False
            trajectory = []
            while not done:
                q_value = self.sess.run(self.output_layer, {self.inputs: [observation]})[0]
                action = self.get_action(epoch, q_value)
                new_observation, reward, done, info = self.env.step(action)
                trajectory.append((observation, action, reward, new_observation, done, q_value))
                observation = new_observation
                if len(trajectory) > self.max_trajectory_length:
                    break
            self.add_to_memory(trajectory)

    def create_batch(self):
        batch_q_values = []
        batch_observations = []
        for experience in self.sample_from_memory():
            action = experience['action']
            new_q_value = np.copy(experience['q_value'])
            new_q_value[action] = experience['reward']
            if not experience['done']:
                update_value = np.max(self.sess.run(self.output_layer, {self.inputs: [experience['to']]})[0])
                new_q_value[action] += self.gamma * update_value
            batch_q_values.append(new_q_value)
            batch_observations.append(experience['from'])
        return batch_observations, batch_q_values

    def train(self):
        epoch = 0
        # while loss_value > 0.002:
        while epoch < self.epochs:
            self.generate_new_trajectories(epoch)
            epoch_loss = None
            for sub_epoch_id in range(self.train_per_epoch):
                batch_observations, batch_q_values = self.create_batch()
                _, epoch_loss = self.sess.run((self.train_op, self.loss),
                                              {self.inputs: batch_observations, self.outputs: batch_q_values})
            self.save()
            epoch_total_reward = self.play()
            print(
                "*********** epoch {} ***********\n"
                "memory size: {}\n"
                "total loss: {}\n"
                "total reward gained: {}\n"
                "epsilon: {}".format(epoch, self.experience_replay_memory.shape[0],
                                     epoch_loss, epoch_total_reward, self.get_epsilon(epoch)))
            epoch += 1

    def play(self, render=False):
        total_reward = 0
        done = False
        observation = self.env.reset()
        while not done:
            if render:
                self.env.render()
            q_value = self.sess.run(self.output_layer, {self.inputs: [observation]})[0]
            # action = self.get_action(0, q_value) # random action
            action = np.argmax(q_value)
            mod = total_reward % 1000
            # if total_reward == 1000:
            if mod in (0, 1, 2, 3):
                print(total_reward)
                # action = self.env.action_space.sample()
                action = 1 - action
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    def save(self):
        self.saver.save(self.sess, self.saving_path)

    def load(self):
        import os
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
        if not tf.train.checkpoint_exists(self.saving_path + 'checkpoint'):
            print('Saved temp_models not found! Randomly initialized.')
        else:
            self.saver.restore(self.sess, self.saving_path)
            print('Model loaded!')


model = GymDQNLearner()
# model.train()
rewards = []
while True:
    total_reward = model.play(False)
    rewards.append(total_reward)
    print('total reward: %d' % total_reward)
    print('reward mean: %f, std: %f' % (np.mean(rewards), np.std(rewards)))
