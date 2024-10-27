import numpy as np
from cec2017.functions import f2, f13


class Individual:
    LIMITATION = 100
    def __init__(self, length, function):
        self.genes = np.random.uniform(-Individual.LIMITATION, Individual.LIMITATION, size=length)
        self.function = function
    
    def evaluate(self):
        return self.function(self.genes)
    

class Population:
    def __init__(self, size, function):
        self.individuals = np.zeros(len(size))
        self.function = function
        for i in range(size):
            ind = Individual(10, self.function)
            self.individuals[i] = ind
    
    def evaluatePopulation(self):
        evaluation = np.zeros(len(self.individuals))
        for i, ind in enumerate(self.individuals):
            evaluation[i] = ind.evaluate()
        return evaluation

    def findTheBest(self) -> Individual:
        theBest = self.individuals[0]
        valueOfTheBest = theBest.evaluate()
        for ind in self.individuals:
            if ind.evaluate() < valueOfTheBest:
                theBest = ind
                valueOfTheBest = ind.evaluate()
        return theBest
    
    def reproduction(self):
        pass


def evolutionaryAlgorithm(function ,population: Population, sigma: float, tMax: int):
    evaluation = population.evaluatePopulation()
    theBest = population.findTheBest()
    for i in range(tMax):
        population.reproduction()


def main():
    BUDGET = 10000
    SIZE = 20
    SIGMA = 3
    function = f2
    population = Population(SIZE, function)
    tMax = BUDGET/SIZE
    evolutionaryAlgorithm(function, population, SIGMA, tMax)


if __name__ == "__main__":
    main()
