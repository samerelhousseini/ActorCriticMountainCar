import tensorflow as tf
import numpy as np
import time
import gc


class Critic(object):

    def __init__(self, env, nn_dims):

        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.nn_dims = nn_dims

        self.build_nn()
        
        
    def build_nn(self):

        layers = [tf.keras.Input(shape=self.observation_space.shape)]
        for d in self.nn_dims:
            layers.extend([
                tf.keras.layers.Dense(d, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal()),
                tf.keras.layers.Dropout(0.1),
            ])
        layers.append(tf.keras.layers.Dense(1, activation='linear'))
        self.model = tf.keras.models.Sequential(layers)

        opt = tf.keras.optimizers.Adam(learning_rate = 0.003, beta_1=0.95, beta_2=0.99)
        
        self.model.compile(optimizer=opt,
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=['mse'])



            
    def gradient_step(self, eps_prev_obs, eps_obs, eps_rewards, eps_terminals, 
                      epoch=0, batch=0, step=False, gamma=0.99):
        
        rv = []
        val_split = 0.1
        
        prev_obs = np.array(eps_prev_obs)
        
        for i in range(len(eps_rewards)):
            if epoch < 1:
                rv.append(0)
            else:
                v = self.model(tf.Variable([eps_obs[i]]))
                rv.append(eps_rewards[i] + gamma * v.numpy()[0][0])     
                    
            if eps_terminals[i] == 1:
                self.model.fit(np.array([eps_obs[i]]), np.array([eps_rewards[i]]), epochs=1,
                               batch_size=1, verbose=0)
        
        
        rvt = np.array(rv)

        if batch % 100 == 9:
            print("Critic::Training Values", prev_obs[:1], rvt[:4], '\n', '\n')
            
        hist = self.model.fit(prev_obs, rvt, epochs=1,
                              batch_size=len(prev_obs), verbose=0)
        return hist