import numpy as np

parameters = {
    "genome_length": 2,
    "population" : (5, 30)
}

class Genome():
    def __init__(self, length = 10):
        self.length = length
        self.genes = np.random.rand(length)
class Agent:
    def __init__(self, id, parameters = parameters):
        self.id = id
        self.genome = Genome(length=parameters["genome_length"])
        self.pop = np.random.randint(parameters["population"][0], parameters["population"][1])

    def act(self, maps):
        pass