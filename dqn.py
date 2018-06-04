from os.path import exists
from random import random

import gym
import numpy as np
import tensorflow as tf


class GymDQNLearner:
    def __init__(self):
        self.epochs = 3000
        self.gamma = 0.9
        self.epsilon = .5

        self.env = gym.make('CartPole-v0')
        self.state_embedding_size = 4 * self.env.observation_space.shape[0]
        self.number_of_actions = self.env.action_space.n
        print(self.state_embedding_size, self.number_of_actions)
        self.layer_units = [self.state_embedding_size, 128, 128, self.number_of_actions]
        self.experience_replay_memory = np.array([])

        self.inputs = self.outputs = self.input_layer = self.hidden1_layer = self.hidden2_layer = self.output_layer = \
            self.loss = self.optimizer = self.saver = None

    def initialize_experience_replay_memory(self):
        self.experience_replay_memory = np.array([])

    def get_epsilon(self, i):
        # alpha = 1e-5
        # return 1.0 - (i / np.sqrt(1 + alpha * (i ** 2))) * np.sqrt(alpha)
        # return 1.0 - float(i) / epochs
        return (1.0 - float(i) / self.epochs) * self.epsilon
        # return .1

    def sample_from_memory(self):
        min_batch_size = 64
        if self.experience_replay_memory.shape[0] > 1:
            return self.experience_replay_memory[
                np.random.randint(0, self.experience_replay_memory.shape[0] - 1,
                                  np.min([min_batch_size, self.experience_replay_memory.shape[0]]), int)]
        else:
            return self.experience_replay_memory

    def get_last_state(self, current_observation):
        if self.experience_replay_memory.shape[0] == 0:
            return np.array([0 for _ in range(3 * self.env.observation_space.shape[0])] + list(current_observation),
                            dtype=np.float32)
        return self.experience_replay_memory[-1]['to']

    def create_model(self):
        self.inputs = tf.placeholder(np.float32, [None, self.state_embedding_size], name='inputs')
        self.outputs = tf.placeholder(np.float32, [None, self.number_of_actions], name='outputs')

        self.input_layer = tf.layers.dense(inputs=self.inputs, units=self.layer_units[0], activation=tf.nn.tanh,
                                           name='in_layer')
        self.hidden1_layer = tf.layers.dense(inputs=self.input_layer, units=self.layer_units[1], activation=tf.nn.tanh,
                                             name='h1_layer')
        self.hidden2_layer = tf.layers.dense(inputs=self.hidden1_layer, units=self.layer_units[2],
                                             activation=tf.nn.tanh,
                                             name='h2_layer')
        self.output_layer = tf.layers.dense(inputs=self.hidden2_layer, units=self.layer_units[3], activation=None,
                                            name='out_layer')

        self.loss = tf.losses.mean_squared_error(self.outputs, self.output_layer)
        # optimizer = tf.train.AdamOptimizer(1e-3, name='optimizer').minimize(loss)
        self.optimizer = tf.train.AdamOptimizer(1e-3, name='optimizer').minimize(self.loss)
        self.saver = tf.train.Saver()

    def load_checkpoint_or_initialize(self, sess):
        if exists('./model/checkpoint'):
            self.saver.restore(sess, './model/vars.ckpt')
            print('variable new values restored!')
        else:
            sess.run(tf.global_variables_initializer())

    def train(self):
        with tf.Session() as sess:
            self.load_checkpoint_or_initialize(sess)
            epoch = 0
            # while loss_value > 0.002:
            while epoch < self.epochs:
                epsilon = self.get_epsilon(epoch)
                observation = self.env.reset()
                done = False
                t = 0
                epoch_total_reward = 0.0
                while not done:
                    # print(t)
                    q_value = sess.run(self.output_layer, {self.inputs: [self.get_last_state(observation)]})[
                        0]
                    if random() < epsilon:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(q_value)
                    new_observation, reward, done, info = self.env.step(action)
                    self.experience_replay_memory = np.append(self.experience_replay_memory, [
                        {'from': self.get_last_state(observation), 'action': action,
                         'reward': reward, 'done': done,
                         'to': np.append(
                             self.get_last_state(observation)[self.env.observation_space.shape[0]:],
                             new_observation),
                         'q_value': q_value}])
                    batch_q_values = []
                    batch_observations = []
                    for experience in self.sample_from_memory():
                        action = experience['action']
                        new_q_value = np.copy(experience['q_value'])
                        new_q_value[action] = experience['reward']
                        if not experience['done']:
                            update_value = np.max(sess.run(self.output_layer, {self.inputs: [experience['to']]})[0])
                            new_q_value[action] += self.gamma * update_value
                        batch_q_values.append(new_q_value)
                        batch_observations.append(experience['from'])
                        self.optimizer.run({self.inputs: batch_observations, self.outputs: batch_q_values})
                    epoch_total_reward += reward
                    observation = new_observation
                    t += 1
                    if done:
                        if epoch % 100 == 0:
                            loss_value = sess.run(self.loss,
                                                  {self.inputs: batch_observations, self.outputs: batch_q_values})
                            self.saver.save(sess, './model/vars.ckpt')
                            print('variable new values saved!')
                            print(
                                "Episode {} training finished after {} timesteps\n"
                                "total loss: {}\n"
                                "total reward gained: {}\n"
                                "epsilon: {}".format(epoch, t, loss_value, epoch_total_reward, epsilon))
                            # print(q_value_history[0])
                        break
                epoch += 1
            self.saver.save(sess, './model/vars.ckpt')

    def play(self):
        with tf.Session() as sess:
            self.load_checkpoint_or_initialize(sess)
            for _ in range(10):
                observation = self.env.reset()
                done = False
                t = 0
                self.initialize_experience_replay_memory()
                last_state = self.get_last_state(observation)
                while not done:
                    self.env.render()
                    q_value = sess.run(self.output_layer, {self.inputs: [last_state]})[0]
                    action = np.argmax(q_value)
                    observation, reward, done, info = self.env.step(action)
                    last_state = np.append(last_state[self.env.observation_space.shape[0]:], observation)
                    t += 1
                    if done:
                        print("Test Episode finished after {} timesteps".format(t))
                        break


model = GymDQNLearner()
model.create_model()
model.train()
model.play()
