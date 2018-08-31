import gym
import numpy as np
import tensorflow as tf


class CartPolePolicyGradient:
    def __init__(self):
        self.env = gym.make('CartPole-v1')

        self.state_embedding_size = self.env.observation_space.shape[0]
        self.number_of_actions = self.env.action_space.n
        self.action_space = [i for i in range(self.number_of_actions)]
        print(self.state_embedding_size, self.number_of_actions)

        self.saving_path = './saved_models/pg/'
        self.learning_rate = 1e-3
        self.gamma = .99
        self.total_epochs = 100
        self.reward_threshold = 250
        self.repeat_per_epoch = 5
        self.policy_training_repeat_per_epoch = 5
        self.value_training_repeat_per_epoch = 5
        self.max_time_step = 300
        self.number_of_trajectory_per_epoch = 64
        self.value_batch_size = 64
        self.layer_units = {'policy': [32, 32, self.number_of_actions],
                            'value': [32, 32, 1]}
        self.layer_activations = {'policy': ['tanh', 'tanh', 'softplus'],
                                  'value': ['tanh', 'tanh', 'softplus']}

        self.build_placeholders()
        self.build_model()
        self.build_loss()
        self.build_ops()
        self.load()

    def build_placeholders(self):
        self.state_over_time_input = tf.placeholder(tf.float32, (None, self.max_time_step, self.state_embedding_size),
                                                    'state_time_input')
        self.selected_actions = tf.placeholder(tf.int32, (None, self.max_time_step), 'actions')
        self.advantage_input = tf.placeholder(tf.float32, (None, self.max_time_step), 'reward')
        self.trajectory_lengths = tf.placeholder(tf.int32, (None,), 'trajectory_lengths')
        self.state_input = tf.placeholder(tf.float32, (None, self.state_embedding_size),
                                          'state_input')
        self.value_labels = tf.placeholder(tf.float32, (None,),
                                           'value_labels')

    def create_multilayer_dense(self, layer_input, layer_units, layer_activations, scope, reuse_vars=None):
        with tf.variable_scope(scope, reuse=reuse_vars):
            last_layer = None
            for i, (layer_size, activation) in enumerate(zip(layer_units, layer_activations)):
                if i == 0:
                    last_layer = tf.layers.dense(layer_input, layer_size, activation)
                else:
                    last_layer = tf.layers.dense(last_layer, layer_size, activation)
        return last_layer

    def build_model(self):
        self.policy_logits = self.create_multilayer_dense(self.state_over_time_input, self.layer_units['policy'],
                                                          self.layer_activations['policy'], 'policy')
        self.test_policy_softmax = tf.nn.softmax(
            self.create_multilayer_dense(self.state_input, self.layer_units['policy'],
                                         self.layer_activations['policy'], 'policy', True))
        self.value_estimates = self.create_multilayer_dense(self.state_input, self.layer_units['value'],
                                                            self.layer_activations['value'], 'value')[:, 0]

    def build_loss(self):
        actions_one_hot = tf.one_hot(self.selected_actions, self.number_of_actions, axis=-1)
        time_step_mask = tf.sequence_mask(self.trajectory_lengths, maxlen=self.max_time_step, dtype=tf.float32)
        likelihood_weight = -time_step_mask * self.advantage_input

        nll = tf.nn.softmax_cross_entropy_with_logits_v2(labels=actions_one_hot, logits=self.policy_logits)
        self.policy_loss = tf.reduce_mean(tf.reduce_sum(nll * likelihood_weight, axis=-1))

        self.value_loss = tf.reduce_mean(tf.squared_difference(self.value_estimates, self.value_labels))

    def build_ops(self):
        policy_trainable_variables = tf.trainable_variables('policy')
        value_trainable_variables = tf.trainable_variables('value')

        self.policy_optimizer = tf.train.AdamOptimizer(self.learning_rate, name='policy_optimizer')
        gradients = tf.gradients(self.policy_loss, policy_trainable_variables)
        gradients = self.get_maximizer_gradients(gradients)
        self.policy_train_op = self.policy_optimizer.apply_gradients(zip(gradients, policy_trainable_variables))

        self.value_train_op = tf.train.AdamOptimizer(self.learning_rate, name='value_optimizer'). \
            minimize(self.value_loss, var_list=value_trainable_variables)

        self.saver = tf.train.Saver(tf.global_variables() + tf.local_variables(), save_relative_paths=True)

        self.sess = tf.Session()

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
                action_probs = self.sess.run(self.test_policy_softmax, {self.state_input: [last_observation]})[0]
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

    def estimate_trajectory_values(self, trajectory_states):
        trajectory_values = []
        for states in trajectory_states:
            values = self.sess.run([self.value_estimates],
                                   {self.state_input: states})[0]
            trajectory_values.append(values)
        return np.array(trajectory_values)

    def prepare_value_estimator_training_data(self, trajectory_states, trajectory_state_values, trajectory_rewards):
        states = []
        values = []
        for i in range(trajectory_states.shape[0]):
            for j in range(trajectory_states[i].shape[0]):
                from_state = trajectory_states[i, j]
                from_state_value = trajectory_state_values[i, j]
                if j == trajectory_states[i].shape[0] - 1:
                    to_state = None
                    to_state_value = 0.0
                else:
                    to_state = trajectory_states[i, j + 1]
                    to_state_value = trajectory_state_values[i, j + 1]
                reward = trajectory_rewards[i, j]
                states.append(from_state)
                values.append(reward + self.gamma * to_state_value)
        return np.array(states), np.array(values)

    def calculate_advantages(self, trajectory_rewards, trajectory_state_values):
        advantages = [[] for _ in range(trajectory_rewards.shape[0])]
        for i in range(trajectory_rewards.shape[0]):
            for j in range(trajectory_rewards[i].shape[0]):
                reward = trajectory_rewards[i, j]
                from_state_value = trajectory_state_values[i, j]
                if j == (trajectory_rewards[i].shape[0] - 1):
                    to_state_value = 0.0
                else:
                    to_state_value = trajectory_state_values[i, j + 1]
                advantages[i].append(reward + to_state_value - from_state_value)
        return np.array(advantages)

    def choose_action(self, p, mode='sample'):
        if mode == 'sample':
            return np.random.choice(self.action_space, size=1, p=p)[0]
        elif mode == 'greedy':
            return np.argmax(p)

    def train_value_estimator(self, states, value_labels):
        total_batches = int(states.shape[0] / self.value_batch_size)
        for batch_id in range(total_batches):
            curr_slice = slice(batch_id * self.value_batch_size, (batch_id + 1) * self.value_batch_size)
            value_loss, _ = self.sess.run([self.value_loss, self.value_train_op],
                                          {self.state_input: states[curr_slice],
                                           self.value_labels: value_labels[curr_slice]})
            if batch_id % 100 == 0:
                print('### batch %d/%d, value loss: %.2f' % (total_batches, batch_id, value_loss))

    def train_policy(self, states, actions, advantages, lengths):
        policy_loss, _ = self.sess.run([self.policy_loss, self.policy_train_op],
                                       {self.state_over_time_input: states,
                                        self.selected_actions: actions,
                                        self.advantage_input: advantages,
                                        self.trajectory_lengths: lengths})
        print('### policy loss: %.2f' % (policy_loss))

    def train(self):
        epoch = -1
        expected_reward = 0.0
        while expected_reward <= self.reward_threshold:
            epoch += 1
            # for epoch in range(self.total_epochs):
            trajectory_states, trajectory_actions, trajectory_rewards, trajectory_lengths = \
                self.generate_trajectories(self.number_of_trajectory_per_epoch)
            trajectory_state_values = self.estimate_trajectory_values(trajectory_states)
            all_states, value_labels = self.prepare_value_estimator_training_data(trajectory_states,
                                                                                  trajectory_state_values,
                                                                                  trajectory_rewards)
            trajectory_advantages = self.calculate_advantages(trajectory_rewards, trajectory_state_values)
            expected_reward = np.mean(np.sum(trajectory_rewards, axis=-1))
            print('@@@@@@@@@ epoch %d, E(R): %.2f' % (epoch, expected_reward))
            if expected_reward >= self.reward_threshold:
                print('epoch %d, reached goal -> E(R): %.2f' % (epoch, expected_reward))
                break
            for repeat in range(self.repeat_per_epoch):
                print('****** repeat: %d' % repeat)
                for _ in range(self.value_training_repeat_per_epoch):
                    self.train_value_estimator(all_states, value_labels)
                for _ in range(self.policy_training_repeat_per_epoch):
                    self.train_policy(trajectory_states, trajectory_actions, trajectory_advantages, trajectory_lengths)
            self.save()

    def play(self):
        # for _ in range(10):
        while True:
            last_observation = self.env.reset()
            done = False
            t = 0
            while not done:
                self.env.render()
                action_probs = self.sess.run(self.test_policy_softmax, {self.state_input: [last_observation]})[0]
                sampled_action = self.choose_action(action_probs, 'greedy')
                last_observation, reward, done, info = self.env.step(sampled_action)
                t += 1
                if done:
                    print("Test Episode finished after {} timesteps".format(t))
                    break

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


model = CartPolePolicyGradient()
model.train()
model.play()
