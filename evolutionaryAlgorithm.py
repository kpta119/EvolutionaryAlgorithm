import numpy as np
import random
from cec2017.functions import f2, f13


class Individual:
    LIMITATION = 100
    def __init__(self, length, function):
        self.genes = np.random.uniform(-Individual.LIMITATION, Individual.LIMITATION, size=length)
        self.function = function
    
    def evaluate(self):
        return self.function(self.genes)
    
    def mutation(self, sigma):
        self.genes += np.random.normal(0, sigma, size=self.genes.shape)
        self.genes = np.clip(self.genes, -Individual.LIMITATION, Individual.LIMITATION)
        
            

class Population:
    def __init__(self, size, function):
        self.size = size
        self.individuals = [Individual(10, function) for _ in range(size)]
        self.function = function
    
    
    def evaluatePopulation(self, population=None):
        if population is None:
            population = self.individuals
        evaluation = np.zeros(self.size)
        for i, ind in enumerate(population):
            evaluation[i] = ind.evaluate()
        return evaluation

    def findTheBest(self, evaluation, population=None) -> Individual:
        if population is None:
            population = self.individuals
        theBestIndex = np.argmin(evaluation)
        theBest = population[theBestIndex]
        return theBest
    
    def reproduction(self): #tournament selection
        Rpopulation = []
        for i in range(self.size):
            ind1 = self.individuals[random.randint(0,self.size-1)]
            ind2 = self.individuals[random.randint(0,self.size-1)]
            if ind1.evaluate() <= ind2.evaluate():
                Rpopulation.append(ind1)
            else:
                Rpopulation.append(ind2)
        return Rpopulation

    def mutationPopulation(self, population, sigma):
        for ind in population:
            ind.mutation(sigma)
        return population

    def succsession(self, newPopulation):
        self.individuals = newPopulation

def evolutionaryAlgorithm(function, population: Population, sigma: float, tMax: int):
    evaluation = population.evaluatePopulation()
    theBest = population.findTheBest(evaluation)
    theBestValue = function(theBest.genes)
    for i in range(tMax):
        Rpopulation = population.reproduction()
        Mpopulation = population.mutationPopulation(Rpopulation, sigma)
        evaluation2 = population.evaluatePopulation(Mpopulation)
        theBestCandidate = population.findTheBest(evaluation2, Mpopulation)
        if function(theBestCandidate.genes) <= theBestValue:
            theBest = theBestCandidate
            theBestValue = function(theBestCandidate.genes)
        population.succsession(Mpopulation)
    return theBest.genes, theBestValue

def main():
    BUDGET = 10000
    SIZE = 20
    SIGMA = 0.3
    function = f2
    population = Population(SIZE, function)
    tMax = int(BUDGET/SIZE)
    print(evolutionaryAlgorithm(function, population, SIGMA, tMax))


if __name__ == "__main__":
    main()
