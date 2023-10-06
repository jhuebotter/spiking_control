

class BaseAgent:
    def __init__(self, env, config, device):
        self.env = env
        self.config = config
        self.device = device

    def train(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError
    
    def load(self):
        raise NotImplementedError