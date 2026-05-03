import numpy as np

class Genome:
    def __init__(self):
        self.data = np.random.rand(10)  # Example genome data, can be more complex

    

class Agent:
    def __init__(self, id):
        self.id = id 
        self.name = f"Agent {id}"
        
        self.wealth = 0
        self.pop = 100
        self.genome = Genome()

