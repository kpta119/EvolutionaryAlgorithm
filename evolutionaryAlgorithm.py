import numpy as np
import random
import copy
from matplotlib import pyplot as plt
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
    def __init__(self, size, function, individuals=None):
        if individuals is None:
            self.individuals = [Individual(10, function) for i in range(size)]
        else:
            self.individuals = individuals
        self.size = size
        self.function = function


    def evaluatePopulation(self):
        evaluation = np.zeros(self.size)
        for i, ind in enumerate(self.individuals):
            evaluation[i] = ind.evaluate()
        return evaluation

    def findTheBest(self, evaluation) -> Individual:
        theBestIndex = np.argmin(evaluation)
        theBest = self.individuals[theBestIndex]
        return theBest

    def reproduction(self): #tournament selection
        Rpopulation = []
        for i in range(self.size):
            ind1 = self.individuals[random.randint(0,self.size-1)]
            ind2 = self.individuals[random.randint(0,self.size-1)]
            win = ind1 if ind1.evaluate() <= ind2.evaluate() else ind2
            Rpopulation.append(copy.deepcopy(win))
        return Population(self.size, self.function, Rpopulation)

    def mutationPopulation(self, sigma):
        for ind in self.individuals:
            ind.mutation(sigma)
        return self

    def succsession(self, newPopulation):
        self.individuals = newPopulation.individuals



def evolutionaryAlgorithm(population: Population, sigma: float, tMax: int):
    f_values = []
    evaluation = population.evaluatePopulation()
    theBest = population.findTheBest(evaluation)
    theBestValue = theBest.evaluate()
    for i in range(tMax):
        Rpopulation = population.reproduction()
        Mpopulation = Rpopulation.mutationPopulation(sigma)
        evaluation2 = Mpopulation.evaluatePopulation()   #array of assessed individuals
        theBestCandidate = Mpopulation.findTheBest(evaluation2)
        if theBestCandidate.evaluate() <= theBestValue:
            theBest = theBestCandidate
            theBestValue = theBestCandidate.evaluate()
        population.succsession(Mpopulation)
        f_values.append(theBestValue)
    return theBest.genes, theBestValue, f_values


def generatePlot(initialPopulation, sigma, tMax, function):
    all_values = []
    for i in range(100):
        _, _, f_values = evolutionaryAlgorithm(initialPopulation, sigma, tMax)
        all_values.append(f_values)
    avg_values = np.mean(all_values, axis=0)
    plt.plot(avg_values)
    plt.xlabel("Iterations")
    plt.ylabel(f"Average values of function {function.__name__} (100 starts)")
    plt.yscale('log')
    plt.show()


def main():
    BUDGET = 10000
    SIZE = 6
    SIGMA = 0.5
    function = f2
    initialPopulation = Population(SIZE, function)
    tMax = int(BUDGET/SIZE)
    generatePlot(initialPopulation, SIGMA, tMax, function)


if __name__ == "__main__":
    main()
