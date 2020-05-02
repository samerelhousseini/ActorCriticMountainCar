from Critic import Critic
from Policy import Policy
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

from IPython import display
import numpy as np
import math

from Episode import Episode


class Agent(object):

    def __init__(self, env, env_name, nn_dims, cr_dims):
        
        self.env = env
        
        self.policy = Policy(env, nn_dims)
        self.critic = Critic(env, cr_dims)
        
        self.rewards = 0
        
        self.loss_history = []
        self.weighted_loss_history = []
        self.reward_history = []
        self.baseline_history = []
        self.variance_history = []
        self.tr_per_run_arr = []
        self.acc_history = []
        self.critic_loss_history = []
        self.sum_rewards = []

    
    
    def get_weights_filenames(self, filename = None):
        if filename == None:
            cr = ''
            ac = ''

            for n in self.policy_dims:
                ac += str(n) + '-'

            for n in self.critic_dims:
                cr += str(n) + '-'
            
            ac_filename = './Weights/'+self.env_name+'Actor-'+ac
            cr_filename = './Weights/'+self.env_name+'Critic-'+cr
        else:
            ac_filename = './Weights/Actor-'+filename
            cr_filename = './Weights/Critic-'+filename
        
        return ac_filename, cr_filename
            
        
    def save_weights(self, filename = None):
        ac_filename, cr_filename = self.get_weights_filenames(filename)
        self.policy.model.save_weights(ac_filename)
        self.critic.model.save_weights(cr_filename)
        
        
    def load_weights(self, filename = None):
        ac_filename, cr_filename = self.get_weights_filenames(filename)
        self.policy.model.load_weights(ac_filename)
        self.critic.model.load_weights(cr_filename)      
    
        
    def take_action(self, ob):
        return self.policy.get_action(ob)

        
    def simulate(self, eps=10, render=True):
        for i in range(eps):
            ep = Episode(self.env, self, render=render)
            _, _, _, _, rewards, _ = ep.run()
            print(i, len(rewards), sum(rewards), max(rewards))
            
            
            
    def train_episode(self, episode=0, render=False):
        
        ep = Episode(self.env, self, render=render)
        t, self.prev_obs, self.obs, self.actions, self.rewards, self.terminals = ep.run()
    
        batch_size = 8
        batches = math.ceil(len(self.prev_obs) / batch_size) 
        
        gamma = 0.99
        critic_gamma = 0.99
        
        self.sum_rewards.append(sum(self.rewards))
        

        print("\n\nNew Episode", episode, "Length", t)
        print("*******************************")
        
        for ii in range(batches):

            if ii == batches - 1:
                start = ii*batch_size
                finish = len(self.prev_obs) 
            else:
                start = ii*batch_size
                finish = (ii+1)*batch_size

            
            print("**** Batch T(start):T(end)", start, finish, "out of", t)
            eps_prev_obs = self.prev_obs[start:finish]
            eps_obs = self.obs[start:finish]
            eps_rewards = self.rewards[start:finish]
            eps_terminals = self.terminals[start:finish]
            eps_actions = self.actions[start:finish]
                        
            obs_pred = self.critic.model(tf.Variable(eps_prev_obs)).numpy()
            next_obs_pred = self.critic.model(tf.Variable(eps_obs)).numpy()
            
            obs_pred = obs_pred.reshape((obs_pred.shape[0],))
            next_obs_pred = next_obs_pred.reshape((next_obs_pred.shape[0],))
            
            
            if ii % 100 == 9:
                print('O(st)', eps_prev_obs[:1])
                print('O(st+1)', eps_obs[:1])
                print('V(st)', obs_pred[:4])
                print('V(st+1)', next_obs_pred[:4])
            

            eps_advantage = eps_rewards + gamma * next_obs_pred - obs_pred

            rewards, advantage, negative_likelihoods, weighted_negative_likelihoods, loss =\
            self.policy.gradient_step(eps_advantage, eps_prev_obs, eps_actions, eps_obs, eps_rewards, 
                                      batch=ii, gamma=gamma)
            
            self.policy.epoch = episode
            

            hist = self.critic.gradient_step(eps_prev_obs, eps_obs, eps_rewards, 
                                             eps_terminals, epoch=episode, 
                                             batch=ii, step=True, gamma=critic_gamma)
            
        self.acc_history.extend(hist.history['mse'])
        self.loss_history.append(tf.reduce_mean(negative_likelihoods))
        self.weighted_loss_history.append(loss.numpy().mean())
        self.reward_history.append(sum(rewards.numpy()) / len(rewards.numpy()))
        self.variance_history.append(tfp.stats.variance(weighted_negative_likelihoods))
        self.tr_per_run_arr.append(t)
    
        return t
    
    
    
    def train(self, episodes = 500):
        
        streaks = []
        max_streaks = 200
        
        for i in range(episodes):
            t = self.train_episode(episode=i)
            streaks.append(t)
        
            if len(streaks) > max_streaks: streaks = streaks[-max_streaks:]
                
            if sum(streaks) < 0.7 * len(streaks) * self.env.spec.max_episode_steps:
                print("Model Fully Trained")
                self.print_results(clear = False)
                break
            
            if i % 5 == 4:
                self.print_results()
    
    
    
    def print_results(self, clear=True):
                    
        if clear == True:
            display.clear_output(wait=True)

        self.fig, self.axs = plt.subplots(2, 3, figsize=(15, 6))
        self.axs[0, 0].plot(self.tr_per_run_arr)
        self.axs[0, 0].set_title('Transitions Per Episode')

        self.axs[0, 1].plot(self.reward_history)
        self.axs[0, 1].set_title('Rewards [History]')

        self.axs[1, 0].plot(self.loss_history)
        self.axs[1, 0].set_title('Loss [History]')

        self.axs[1, 1].plot(self.weighted_loss_history)
        self.axs[1, 1].set_title('Weighted Loss [History]')

        self.axs[1, 2].plot(self.sum_rewards)
        self.axs[1, 2].set_title('Rewards [History]')

        self.axs[0, 2].plot(self.acc_history)
        self.axs[0, 2].set_title('Critic Accuracy [History]')

        plt.show()


