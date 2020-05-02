import tensorflow as tf
import gc
import math
import statistics
from random import random
import numpy as np
import tensorflow_probability as tfp
import gym



class Policy(object):

    def __init__(self, env, nn_dims):

        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        print('Action Space', self.action_space)
        if isinstance(self.action_space, gym.spaces.box.Box):
            print('Action Space High', self.action_space.high)
            print('Action Space Low', self.action_space.low)
            
        print('\nObservation Space', self.observation_space, '\nHigh',
              self.observation_space.high, '\nLow', self.observation_space.low,
              end='\n\n')
        
        self.epoch = 0
        self.nn_dims = nn_dims
        self.build_nn()


        
    def build_nn(self):
        layers = [tf.keras.Input(shape=self.observation_space.shape)]

        for d in self.nn_dims:
            layers.extend([
                tf.keras.layers.Dense(d, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotNormal()),
                tf.keras.layers.Dropout(0.1),
            ])

        layers.append(tf.keras.layers.Dense(self.action_space.n, activation='softmax'))

        self.model = tf.keras.models.Sequential(layers)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.95, beta_2=0.99)
        
        

    def get_action(self, ob):
        epsilon = 0.3 * math.pow(0.99, self.epoch)
        if random() < epsilon:
            action = self.action_space.sample()
            return action
            
        action_tensor = self.model(tf.Variable([ob]))
        probs = action_tensor[0].numpy() / sum(action_tensor[0].numpy())
        action = np.random.choice(np.arange(len(action_tensor[0])), p=probs)
        return action


    
    def gradient_step(self, gae, eps_prev_obs, eps_actions, eps_obs, eps_rewards, batch=0, gamma=0.99):
        mean_as = 0
        obs_pred = []
        next_obs_pred = []

        obs = tf.Variable(eps_prev_obs)
        
        if isinstance(self.action_space, gym.spaces.box.Box):
            actions = tf.Variable(eps_actions)
        else:
            actions = tf.Variable(tf.one_hot(eps_actions, self.action_space.n))

        rewards = tf.Variable(eps_rewards, dtype=float)
            
        with tf.GradientTape() as tape:
            logits = self.model(obs, training=True)
            negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=actions, logits=logits)
            advantage = tf.Variable(gae, dtype=float)
            weighted_negative_likelihoods = tf.multiply(negative_likelihoods, advantage )
            loss = tf.reduce_mean(weighted_negative_likelihoods)
            grads = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        if batch % 100 == 9:
            print('Actions', actions[:5].numpy())
            print('Logits', logits[:5].numpy())
            print('Rewards', rewards[:5].numpy())
            print('A(st)', advantage[:5].numpy())
            print('NLs', negative_likelihoods[:5].numpy())
            print('WNLs', weighted_negative_likelihoods[:5].numpy())
            print('Loss', loss.numpy())
        
        return rewards, advantage, negative_likelihoods, weighted_negative_likelihoods, loss
    
    
    
