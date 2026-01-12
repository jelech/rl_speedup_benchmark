import numpy as np

class RandomAgent:
    def __init__(self, action_space_n=2):
        self.action_space_n = action_space_n
        
    def act(self, state):
        # Using numpy random is faster than random.choice usually for simple ints
        return np.random.randint(0, self.action_space_n)
    
    def learn(self, *args):
        # Simulate a tiny bit of learning overhead if needed, 
        # or keep empty to focus purely on Env speed.
        pass

class SimplePolicyAgent:
    """A dummy agent that does a matrix multiplication to simulate inference time"""
    def __init__(self, obs_dim=4, act_dim=2):
        self.weights = np.random.randn(obs_dim, act_dim)
        
    def act(self, state):
        # Simple linear policy
        logits = np.dot(state, self.weights)
        return np.argmax(logits)
        
    def learn(self, *args):
        # Simulate a small backprop/update op
        # y = mx + b type calculation
        self.weights += 0.001 * np.random.randn(*self.weights.shape)
