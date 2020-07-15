import numpy as np


class _Creature:
    def __init__(self):
        self.time = 0

    def mutate(self):
        pass

    def crossover(self):
        """can only crossover with same subclass/'species'"""
        pass

    def getfitness(self):
        pass


class BowCreature(_Creature):
    def getfitness(self):
        """f(t, kills, arrows left)"""


class SquareCreature(_Creature):
    def getfitness(self):
        """f(t), Increases most initially, plateaus later"""
        return np.sqrt(self.time)


class EvolutionManager:

    def __init__(self):
        pass

    def cycle(self):
        pass

    def select(self):
        pass

