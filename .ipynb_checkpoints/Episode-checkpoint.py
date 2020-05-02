import math


class Episode(object):
    def __init__(self, env, agent, render):

        self.env = env
        self.agent = agent
        self.render = render

        self.prev_obs = []
        self.obs = []
        self.actions = []
        self.rewards = []
        self.terminals = []


    def run(self):
        observation = self.env.reset()
        done = False
        t = 0
        
        self.prev_obs = []
        self.obs = []
        self.actions = []
        self.rewards = []
        self.terminals = []
        
        
        while not done:
            if self.render == True: self.env.render()
            
            action = self.agent.take_action(observation)
            prev_ob = observation
            observation, reward, done, info = self.env.step(action)

            t = t + 1

            reward = reward + math.pow(observation[1] * 100, 2)
            
            if done: 
                reward = -60
                
                if t < self.env.spec.max_episode_steps: 
                    reward = 300        

            self.rewards.append(reward)
            self.prev_obs.append(prev_ob)
            self.obs.append(observation)
            self.actions.append(action)
            self.terminals.append(done)


        return t, self.prev_obs, self.obs, self.actions, self.rewards, self.terminals