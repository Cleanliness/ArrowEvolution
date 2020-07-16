import numpy as np
import pygame
import GameObjects
import NN


class Creature:
    def __init__(self, surface):
        self.time = 0
        self.threshold = 0.7
        self.brain = NN.NN([4, 5, 4, 3])
        self.surface = surface

    def drawNN(self, x=600, y=40):
        h_offset = 0
        self.brain.classify([1, 2, 3, 4])
        v_offset = 0
        old_v = 0

        for layer in range(0, len(self.brain.activations)):

            # centering each layer at middle of first layer
            if h_offset > 0:
                old_v = v_offset
                v_offset = (40*len(self.brain.activations[0]) - 40*len(self.brain.activations[layer])) / 2

            # drawing neurons
            for act in range(0, len(self.brain.activations[layer])):
                rgb = self.brain.activations[layer][act]*255
                c = (rgb, rgb, rgb)

                pygame.draw.ellipse(self.surface, (0, 0, 0), pygame.Rect(x + h_offset, y + v_offset, 25, 25))
                pygame.draw.ellipse(self.surface, c, pygame.Rect(x + h_offset + 2, y + v_offset + 2, 21, 21))

                # draw weights connecting each neuron
                if layer > 0:
                    npos = 0
                    for weight in self.brain.weights[layer-1][act]:
                        # only draw strong weights
                        if abs(weight) > 1:
                            rgb = (weight+2)/4*255
                            c = (rgb, 255-rgb, 0)
                            start = (x + h_offset - 48, y + old_v - 40 * len(self.brain.activations[layer - 1]) + npos + 13)
                            end = (x + h_offset, y + v_offset + 13)
                            pygame.draw.aaline(self.surface, c, start, end)

                        npos += 40
                v_offset += 40
            h_offset += 70

    def mutate(self):
        pass

    def crossover(self, creature):
        """can only crossover with same subclass/'species'"""
        pass

    def get_fitness(self):
        pass


class BowCreature(Creature):
    """Represents a bow controlled by a NN. Input and output layers are set up like this:
    -----------------------------------------------------------------
    O = neuron

    ==Input Layer==                                 ==Output Layer==
    O - bow strength
    O - theta
    O - enemy's x position                           O - mouse x position
    O - enemy's y position   ..................      O - mouse y position
    O - enemy's dx                                   O - mouse clicked (if activation > 0.7)
    O - enemy's dy
    """
    def __init__(self, surface, bow):
        super().__init__(surface)
        self.brain = NN.NN([6, 4, 4, 3])
        self.game_model = bow

    def get_fitness(self):
        """f(time(exponential decay), arrows left(exp))"""

    def update_nn(self, x, y, dx, dy):
        inp_layer = [self.game_model.draw_dx/80, self.game_model.draw_dy/20]
        inp_layer.extend([(x-255)/50, (y-255)/50, (dx)*3, (dy)*3])

        self.brain.classify(inp_layer)

    def move_model(self):
        self.game_model.pull_bow(self.brain.activations[-1][0:2])

        if self.brain.activations[-1][2] > 0.7:
            # TODO call fire bow correctly from NN
            print('fireeee!!')


class SquareCreature(Creature):
    """Represents an enemy controlled by a NN. Input and output layers are set up like this:
    ------------------------------------------------------------
    O = neuron

    ==Input Layer==                              ==Output Layer==
    O - bow draw strength in x direction
    O - bow draw strength y                          O - up
    O - arrow in q1                                  O - down
    O - arrow in q2   ..........................     O - left
    O - arrow in q3                                  O - right
    O - arrow in q4

    """
    def __init__(self, surface, enemy):
        super().__init__(surface)
        self.brain = NN.NN([6, 5, 5, 4])
        self.game_model = enemy

    def update_nn(self, x, y):
        inp_layer = [x/80, y/20]
        for q in self.game_model.quadstat:
            if q:
                inp_layer.append(200)
            else:
                inp_layer.append(-200)
        self.brain.classify(inp_layer)

    def move_model(self):
        out = self.brain.activations[-1]
        processed = []

        for i in out:
            if i > self.threshold:
                processed.append(1)
            else:
                processed.append(0)

        # self.game_model.player_control(processed)

    def get_fitness(self):
        """f(t), Increases most initially, plateaus later"""
        return 8*np.sqrt(self.time)


class EvolutionManager:

    def __init__(self, surface, mutation_rate=0.05, pop_size=15):
        self.generations = 1
        self.individuals = 1
        self.pop_size = pop_size
        self.mut_rate = mutation_rate
        self.surface = surface

        self.bow_gen = [BowCreature(self.surface, GameObjects.Bow(self.surface)) for i in range(0, self.pop_size)]
        self.enemy_gen = [SquareCreature(self.surface, GameObjects.Enemy(self.surface)) for i in
                            range(0, self.pop_size)]

    def cycle(self):
        while self.individuals <= self.pop_size:
            for i in range(0, len(self.bow_gen)):
                self.bow_gen[i].update_nn()
                self.enemy_gen[i].update_nn()

    def select(self):
        pass

