import gym
import numpy as np
import tensorflow as tf


class CartPolePolicyGradient:
    def __init__(self):
        self.env = gym.make('CartPole-v0')

        self.state_embedding_size = self.env.observation_space.shape[0]
        self.number_of_actions = self.env.action_space.n
        self.action_space = [i for i in range(self.number_of_actions)]
        print(self.state_embedding_size, self.number_of_actions)

        self.learning_rate = 1e-3
        self.total_epochs = 500
        self.max_time_step = 100
        self.number_of_trajectory_per_epoch = 64
        self.layer_units = [128, 128, self.number_of_actions]
        self.layer_activations = ['relu', 'relu', 'relu']

        self.build_placeholders()
        self.build_model()
        self.build_loss()
        self.build_ops()

    def build_placeholders(self):
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.state_input = tf.placeholder(tf.float32, (None, self.max_time_step, self.state_embedding_size),
                                          'state_input')
        self.selected_actions = tf.placeholder(tf.int32, (None, self.max_time_step), 'actions')
        self.reward_input = tf.placeholder(tf.float32, (None, self.max_time_step), 'reward')
        self.trajectory_lengths = tf.placeholder(tf.int32, (None,), 'trajectory_lengths')
        self.test_state_input = tf.placeholder(tf.float32, (None, self.state_embedding_size),
                                               'test_state_input')

    def build_model(self):
        with tf.variable_scope('model'):
            self.logits = None
            for i, (layer_size, activation) in enumerate(zip(self.layer_units, self.layer_activations)):
                if i == 0:
                    self.logits = tf.layers.dense(self.state_input, layer_size, activation)
                else:
                    self.logits = tf.layers.dense(self.logits, layer_size, activation)
        with tf.variable_scope('model', reuse=True):
            self.test_logits = None
            for i, (layer_size, activation) in enumerate(zip(self.layer_units, self.layer_activations)):
                if i == 0:
                    self.test_logits = tf.layers.dense(self.test_state_input, layer_size, activation)
                else:
                    self.test_logits = tf.layers.dense(self.test_logits, layer_size, activation)
            self.test_softmax = tf.nn.softmax(self.test_logits)

    def build_loss(self):
        actions_one_hot = tf.one_hot(self.selected_actions, self.number_of_actions, axis=-1)
        time_step_mask = tf.sequence_mask(self.trajectory_lengths, maxlen=self.max_time_step, dtype=tf.float32)
        cumulative_rewards = tf.cumsum(self.reward_input, axis=-1)
        likelihood_weight = time_step_mask * cumulative_rewards

        self.nll = tf.nn.softmax_cross_entropy_with_logits_v2(labels=actions_one_hot, logits=self.logits)
        self.loss = tf.reduce_mean(tf.reduce_sum(self.nll * likelihood_weight, axis=-1))

    def build_ops(self):
        trainable_variables = tf.trainable_variables()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name='adam_optimizer')
        gradients = tf.gradients(self.loss, trainable_variables)
        gradients = self.get_maximizer_gradients(gradients)
        self.train_op = self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_maximizer_gradients(self, gradients):
        maximizer_gradients = []
        for grad in gradients:
            maximizer_gradients.append(-1 * grad)
        return maximizer_gradients

    def generate_trajectories(self, number_of_trajectories):
        states = [[] for _ in range(number_of_trajectories)]
        actions = [[] for _ in range(number_of_trajectories)]
        rewards = [[] for _ in range(number_of_trajectories)]
        lengths = []

        for i in range(number_of_trajectories):
            last_observation = self.env.reset()
            for j in range(self.max_time_step):
                action_probs = self.sess.run(self.test_softmax, {self.test_state_input: [last_observation],
                                                                 self.batch_size: 1})[0]
                sampled_action = self.choose_action(action_probs)
                new_observation, reward, done, info = self.env.step(sampled_action)
                states[i].append(last_observation)
                actions[i].append(sampled_action)
                rewards[i].append(reward)
                last_observation = new_observation
                if done:
                    break
            lengths.append(j + 1)
            fake_state = np.zeros_like(last_observation)
            for _ in range(self.max_time_step - j - 1):
                states[i].append(fake_state)
                actions[i].append(-1)
                rewards[i].append(0)
        return np.array(states), np.array(actions), np.array(rewards), np.array(lengths)

    def train(self):
        for epoch in range(self.total_epochs):
            states, actions, rewards, lengths = self.generate_trajectories(self.number_of_trajectory_per_epoch)
            epoch_loss, _ = self.sess.run([self.loss, self.train_op], {self.state_input: states,
                                                                       self.selected_actions: actions,
                                                                       self.reward_input: rewards,
                                                                       self.trajectory_lengths: lengths,
                                                                       self.batch_size: self.number_of_trajectory_per_epoch})
            expected_reward = np.mean(np.sum(rewards, axis=-1))
            print('epoch %d loss: %.2f, E(R): %.2f' % (epoch, epoch_loss, expected_reward))

    def choose_action(self, p):
        return np.random.choice(self.action_space, size=1, p=p)[0]
        # return np.argmax(p)

    def play(self):
        # for _ in range(10):
        while True:
            last_observation = self.env.reset()
            done = False
            t = 0
            while not done:
                self.env.render()
                action_probs = self.sess.run(self.test_softmax, {self.test_state_input: [last_observation],
                                                                 self.batch_size: 1})[0]
                sampled_action = self.choose_action(action_probs)
                last_observation, reward, done, info = self.env.step(sampled_action)
                t += 1
                if done:
                    print("Test Episode finished after {} timesteps".format(t))
                    break


model = CartPolePolicyGradient()
model.train()
model.play()
