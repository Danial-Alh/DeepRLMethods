from os.path import exists
from random import random

import gym
import numpy as np
import tensorflow as tf

# env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v0')
state_embedding_size = 4 * env.observation_space.shape[0]
action_embedding_size = env.action_space.n
print(state_embedding_size, action_embedding_size)
layer_units = [state_embedding_size, 256, 128, action_embedding_size]

inputs = tf.placeholder(np.float32, [None, state_embedding_size], name='inputs')
outputs = tf.placeholder(np.float32, [None, action_embedding_size], name='outputs')

input_layer = tf.layers.dense(inputs=inputs, units=layer_units[0], activation=tf.nn.relu, name='in_layer')
hidden1_layer = tf.layers.dense(inputs=input_layer, units=layer_units[1], activation=tf.nn.tanh, name='h1_layer')
hidden2_layer = tf.layers.dense(inputs=hidden1_layer, units=layer_units[2], activation=tf.nn.relu, name='h2_layer')
output_layer = tf.layers.dense(inputs=hidden2_layer, units=layer_units[3], activation=tf.nn.relu, name='out_layer')

loss = tf.losses.mean_squared_error(outputs, output_layer)
# optimizer = tf.train.AdamOptimizer(1e-3, name='optimizer').minimize(loss)
optimizer = tf.train.AdamOptimizer(1e-3, name='optimizer').minimize(loss)
saver = tf.train.Saver()

epochs = 10000
gamma = 0.9
epsilon = .5


def get_epsilon(i):
    # alpha = 1e-5
    # return 1.0 - (i / np.sqrt(1 + alpha * (i ** 2))) * np.sqrt(alpha)
    return 1.0 - float(i) / epochs


def sample_from_memory(memory):
    min_batch_size = 64
    if memory.shape[0] > 1:
        return memory[np.random.randint(0, memory.shape[0] - 1, np.min([min_batch_size, memory.shape[0]]), int)]
    else:
        return memory


def get_last_state(memory, current_observation):
    if memory.shape[0] == 0:
        return np.array([0 for _ in range(3 * env.observation_space.shape[0])] + list(current_observation),
                        dtype=np.float32)
    return memory[-1]['to']


with tf.Session() as sess:
    if exists('./model/checkpoint'):
        saver.restore(sess, './model/vars.ckpt')
        print('variable new values restored!')
    else:
        sess.run(tf.global_variables_initializer())
    loss_value = 1
    epoch = 0
    # while loss_value > 0.002:
    while epoch < epochs:
        epsilon = get_epsilon(epoch)
        observation = env.reset()
        done = False
        t = 0
        epoch_total_reward = 0.0
        experience_replay_memory = np.array([])
        while not done:
            # print(t)
            q_value = sess.run(output_layer, {inputs: [get_last_state(experience_replay_memory, observation)]})[0]
            if random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_value)
            new_observation, reward, done, info = env.step(action)
            experience_replay_memory = np.append(experience_replay_memory, [
                {'from': get_last_state(experience_replay_memory, observation), 'action': action,
                 'reward': reward, 'done': done,
                 'to': np.append(get_last_state(experience_replay_memory, observation)[env.observation_space.shape[0]:],
                                 new_observation),
                 'q_value': q_value}])
            batch_q_values = []
            batch_observations = []
            for experience in sample_from_memory(experience_replay_memory):
                new_q_value = np.copy(experience['q_value'])
                new_q_value[action] = experience['reward']
                if not experience['done']:
                    update_value = np.max(sess.run(output_layer, {inputs: [experience['to']]})[0])
                    new_q_value[action] += gamma * update_value
                batch_q_values.append(new_q_value)
                batch_observations.append(experience['from'])
            optimizer.run({inputs: batch_observations, outputs: batch_q_values})
            epoch_total_reward += reward
            observation = new_observation
            t += 1
            if done:
                if epoch % 1000 == 0:
                    loss_value = sess.run(loss, {inputs: batch_observations, outputs: batch_q_values})
                    saver.save(sess, './model/vars.ckpt')
                    print('variable new values saved!')
                    print(
                        "Episode {} training finished after {} timesteps\n"
                        "total loss: {}\n"
                        "total reward gained: {}\n"
                        "epsilon: {}".format(epoch, t, loss_value, epoch_total_reward, epsilon))
                    # print(q_value_history[0])
                break
        epoch += 1

    for _ in range(10):
        observation = env.reset()
        done = False
        t = 0
        epoch_total_reward = 0.0
        last_state = get_last_state(np.array([]), observation)
        while not done:
            env.render()
            q_value = sess.run(output_layer, {inputs: [observation]})[0]
            action = np.argmax(q_value)
            observation, reward, done, info = env.step(action)
            last_state = np.append(last_state[env.observation_space.shape[0]:], observation)
            t += 1
            if done:
                print("Test Episode finished after {} timesteps".format(t))
                break
