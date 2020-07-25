import numpy as np
import pygame
import GameObjects
import NN
import random
import neat


class Creature:
    def __init__(self, surface):
        self.time = 0.01
        self.threshold = 0.7
        self.surface = surface
        self.fitness = 0.01


class BowCreature(Creature):
    """Represents a bow controlled by a NN. Input and output layers are set up like this:
    -----------------------------------------------------------------
    O = neuron

    ==Input Layer==                                 ==Output Layer==
    O - bow strength x
    O - bow strength y
    O - enemy's x position                           O - mouse x position
    O - enemy's y position   ..................      O - mouse y position
    O - enemy's dx                                   O - mouse clicked (if activation > self.threshold)
    O - enemy's dy
    """

    def __init__(self, surface, bow):
        super().__init__(surface)
        self.game_model = bow
        self.threshold = 0.9
        self.killed = False

    def set_fitness(self, disp):
        """ Sets and returns fitness of this BowCreature instance. Fitness is a function of time and average distance
        between all arrows and the enemy per game tick.
        """
        if disp < 0:
            return 0

        # calculate fitness on average arrow distance to enemy per tick
        d_comp = 20000/(disp + 10)
        tcomp = 10 * np.exp(0.1*(-1*self.time + 28))

        if self.killed:
            kcomp = 8*(np.random.random_sample()*2 - 1) + 30

        else:
            kcomp = -5*(np.random.random_sample()*2)

        self.fitness = (d_comp + tcomp + kcomp)
        return self.fitness

    def move_model(self, out):
        """Moves bow creature given an output of a neural network. <out> must be a list of length 3. Returns an arrow
        object when shot successfully"""
        self.time += 0.05

        # updating pull position of game model
        inp = out
        inp[0] = inp[0]*300 + 100
        inp[1] = inp[1]*100 + 598
        self.game_model.pull_bow(False, inp)

        # shooting check
        if out[-1] > self.threshold:

            mp = self.game_model.drawpos
            dx = (247-mp[0])/20
            dy = (573-mp[1])/20
            spd = np.sqrt(np.square(dx) + np.square(dy))

            if spd > 5:
                arrow = self.game_model.shoot(mp, dx, dy)
                return arrow


class SquareCreature(Creature):
    """Represents an enemy controlled by a NN. Input and output layers are set up like this:
    ------------------------------------------------------------
    O = neuron

    ==Input Layer==                              ==Output Layer==
    O - bow draw strength in x direction
    O - bow draw strength y
    O - x position                                   O - Up
    O - y position      ......................       O - down
    O - arrow in q1                                  O - left
    O - arrow in q2                                  O - Right
    O - arrow in q3
    O - arrow in q4

    """

    def __init__(self, surface, enemy):
        super().__init__(surface)
        self.game_model = enemy
        self.avoided = 0

    def move_model(self, outp):

        self.time += 0.05
        out = outp
        processed = []

        for i in out:
            if i > self.threshold:
                processed.append(1)
            else:
                processed.append(0)

        self.game_model.player_control(False, processed)

    def set_fitness(self):
        """f(t), Increases most initially, plateaus later, sets fitness"""
        self.fitness = 8 * np.sqrt(self.time) + self.avoided
        return self.fitness


class EvolutionManager:

    def __init__(self, surface, b_config, e_config):
        # UI stuff
        self.stfont = pygame.font.SysFont('Consolas', 15, bold=False)
        self.drawing = True
        self.surface = surface

        # neat-python variables
        self.b_conf = b_config
        self.e_conf = e_config
        self.b_pop = neat.Population(b_config)
        self.e_pop = neat.Population(e_config)
        self.b_brains = [neat.nn.FeedForwardNetwork.create(self.b_pop.population[i], b_config) for i in self.b_pop.population]
        self.e_brains = [neat.nn.FeedForwardNetwork.create(self.e_pop.population[i], e_config) for i in self.e_pop.population]

        # game objects
        self.bow_gen = [BowCreature(self.surface, GameObjects.Bow(self.surface))
                        for i in range(0, len(self.b_pop.population))]
        self.enemy_gen = [SquareCreature(self.surface, GameObjects.Enemy(self.surface, 100)) for i in
                          range(0, len(self.e_pop.population))]
        self.managers = [GameObjects.WaveManager([self.enemy_gen[i].game_model], self.bow_gen[i].game_model)
                         for i in range(0, len(self.bow_gen))]
        self.timers = [0 for i in range(0, len(self.bow_gen))]

        # statistics
        self.generations = 1
        self.avg_b_fitness = []
        self.avg_e_fitness = []

    def multicycle(self):
        """Runs and draws a game cycle on multiple enemies and bows.
        Restarts when all timers reach 100/all enemies die."""
        view_single = False
        mp = pygame.mouse.get_pos()
        mouse_ind = None

        if int(mp[1]) in range(12, 12 + 10*len(self.bow_gen)) and int(mp[0]) in range(860, 990):
            mouse_ind = (mp[1]-12) // 10
            view_single = True

        # game loop
        for i in range(0, len(self.bow_gen)):
            if self.timers[i] < 100:
                # updating game objects
                b = self.bow_gen[i].game_model
                e = self.enemy_gen[i].game_model
                self.managers[i].detectCollisions()
                if len(self.managers[i].enemies) < 1:
                    self.bow_gen[i].killed = True
                self.enemy_gen[i].avoided = self.managers[i].missed

                # setting input for enemy neural networks
                e_inp = [b.draw_dx, b.draw_dy, (e.pos[0] - 255), (e.pos[1] - 255)]
                for q in e.quadstat:
                    if q:
                        e_inp.append(100)
                    else:
                        e_inp.append(0)
                self.e_brains[i].activate(e_inp)

                # setting input for bow neural networks
                b_inp = [b.draw_dx, b.draw_dy]
                b_inp.extend([(e.pos[0] - 255), (e.pos[1] - 255), e.vel[0] * 3, e.vel[1] * 3])
                self.b_brains[i].activate(b_inp)

                # drawing game objects and timer bars
                time_rect = pygame.Rect(860, 12 + 10 * i, (100 - self.timers[i]) * 1.3, 5)
                pygame.draw.rect(self.surface, (0, 255, 0), time_rect)

                if (not view_single) or (i == mouse_ind):
                    self.managers[i].draw()

                # listening for NN interaction, and updating game models
                b_out = [self.b_brains[i].values[ind] for ind in range(0, 3)]
                e_out = [self.e_brains[i].values[ind] for ind in range(0, 4)]
                self.enemy_gen[i].move_model(e_out)
                arrow = self.bow_gen[i].move_model(b_out)
                if arrow is not None:
                    self.managers[i].add_arrow(arrow)

                # countdown when stuck/unstuck, ending timer when enemy dies
                if not e.is_moving() or self.managers[i].failed_shots > 15 or b.arrows > 18 or b.time_between_shots > 500:
                    self.timers[i] += 0.2
                elif len(self.managers[i].enemies) < 1:
                    self.timers[i] = 150
                else:
                    self.timers[i] = 0
        self.drawstats()

        # checking all generations finished, select, breed, and mutate
        for i in self.timers:
            if i < 100:
                return None
        self.b_pop.run(self.bow_ff, 1)
        self.e_pop.run(self.enemy_ff, 1)

        # reset timers and game models, brains
        self.bow_gen = [BowCreature(self.surface, GameObjects.Bow(self.surface))
                        for i in range(0, len(self.b_pop.population))]
        self.enemy_gen = [SquareCreature(self.surface, GameObjects.Enemy(self.surface, 100)) for i in
                          range(0, len(self.e_pop.population))]
        self.managers = [GameObjects.WaveManager([self.enemy_gen[i].game_model], self.bow_gen[i].game_model)
                         for i in range(0, len(self.bow_gen))]
        self.timers = [0 for i in range(0, len(self.bow_gen))]
        self.b_brains = [neat.nn.FeedForwardNetwork.create(self.b_pop.population[i], self.b_conf) for i in self.b_pop.population]
        self.e_brains = [neat.nn.FeedForwardNetwork.create(self.e_pop.population[i], self.e_conf) for i in self.e_pop.population]
        self.generations += 1

    def bow_ff(self, genomes, config):
        """bow fitness function for neat-python. Updates NEAT's genome fitnesses"""

        tot_fit = 0
        i = 0
        for b_id, genome in genomes:
            ft = self.bow_gen[i].set_fitness(self.managers[i].avg_dist)
            tot_fit += ft
            genome.fitness = ft
            i += 1

        avg_fit = tot_fit / len(self.b_pop.population)
        self.avg_b_fitness.append(avg_fit)

    def enemy_ff(self, genomes, config):
        """Enemy fitness function for neat-python. Updates NEAT's genome fitnesses"""

        tot_fit = 0
        i = 0
        for e_id, genome in genomes:
            ft = self.enemy_gen[i].set_fitness()
            tot_fit += ft
            genome.fitness = ft
            i += 1

        avg_fit = tot_fit / len(self.e_pop.population)
        self.avg_e_fitness.append(avg_fit)

    def drawstats(self):
        """Draws stats contained in this EvolutionManager instance onto surface"""

        # drawing best and avg fitness of last generation as text
        b_avg_str = "Average bow fitness: "
        e_avg_str = "Average enemy fitness: "

        bbest_str = "Best bow fitness: "
        ebest_str = "Best enemy fitness: "

        if self.generations > 1:
            b_avg_str += str(round(self.avg_b_fitness[-1], 2))
            e_avg_str += str(round(self.avg_e_fitness[-1], 2))

            bbest_str += str(round(self.b_pop.best_genome.fitness, 2))
            ebest_str += str(round(self.e_pop.best_genome.fitness, 2))

        gen_lbl = self.stfont.render("Generation " + str(self.generations), False, (0, 0, 0))
        b_avg_lbl = self.stfont.render(b_avg_str, False, (0, 0, 0))
        e_avg_lbl = self.stfont.render(e_avg_str, False, (0, 0, 0))
        bbest_fit_lbl = self.stfont.render(bbest_str, False, (0, 0, 0))
        ebest_fit_lbl = self.stfont.render(ebest_str, False, (0, 0, 0))

        self.surface.blit(gen_lbl, (1020, 10))
        self.surface.blit(b_avg_lbl, (1020, 25))
        self.surface.blit(e_avg_lbl, (1020, 40))
        self.surface.blit(bbest_fit_lbl, (1020, 55))
        self.surface.blit(ebest_fit_lbl, (1020, 70))

        # graphing fitnesses
        self.graph_fitnesses()

    def graph_fitnesses(self):
        """plots enemy and bow fitness on a graph"""

        # drawing graph borders and box
        pygame.draw.aaline(self.surface, (0, 0, 0), (1000, 100), (1400, 100))
        pygame.draw.aaline(self.surface, (0, 0, 0), (1000, 400), (1400, 400))
        graph_rect = pygame.Rect(1020, 120, 360, 260)
        pygame.draw.rect(self.surface, (0, 0, 0), graph_rect, 1)

        if len(self.avg_b_fitness) > 0:
            # set up horizontal offset and scaling factor
            h_off = 360 / (len(self.avg_b_fitness) + 1)
            scale = 0.85*260/max(self.avg_b_fitness + self.avg_e_fitness)

            start_b = (1020, 380)
            start_e = (1020, 380)
            for i in range(0, len(self.avg_b_fitness)):
                end_b = (1020 + h_off*(i+1), 120 + 260 - self.avg_b_fitness[i]*scale)
                end_e = (1020 + h_off * (i + 1), 120 + 260 - self.avg_e_fitness[i]*scale)

                # bow line (red), enemy line (green)
                pygame.draw.aaline(self.surface, (255, 0, 0), start_b, end_b)
                pygame.draw.aaline(self.surface, (0, 255, 0), start_e, end_e)

                # update start points
                start_b = end_b
                start_e = end_e