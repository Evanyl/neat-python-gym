import gym

class CartPole:
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.env.reset()
        self.env._max_episode_steps = 2000
        self.actions = []

    def render(self):
        self.env.render()

    def step(self,):
        if len(self.actions) == 0:
            action = self.env.action_space.sample()
        else:
            action = self.actions.pop()

        return self.env.step(action)

    def addInstruction(self, newAction):
        self.actions.append(newAction)

    def done(self):
        self.env.close()
