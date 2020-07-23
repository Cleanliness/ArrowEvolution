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
        self.brain = NN.NN([4, 5, 4, 3])
        self.surface = surface
        self.fitness = 0.01

    def copy_creature(self):
        pass

    def drawNN(self, x=600, y=40):
        h_offset = 0
        self.brain.classify([1, 2, 3, 4])
        v_offset = 0
        old_v = 0

        for layer in range(0, len(self.brain.activations)):

            # centering each layer at middle of first layer
            if h_offset > 0:
                old_v = v_offset
                v_offset = (40 * len(self.brain.activations[0]) - 40 * len(self.brain.activations[layer])) / 2

            # drawing neurons
            for act in range(0, len(self.brain.activations[layer])):
                rgb = self.brain.activations[layer][act] * 255
                c = (rgb, rgb, rgb)

                pygame.draw.ellipse(self.surface, (0, 0, 0), pygame.Rect(x + h_offset, y + v_offset, 25, 25))
                pygame.draw.ellipse(self.surface, c, pygame.Rect(x + h_offset + 2, y + v_offset + 2, 21, 21))

                # draw weights connecting each neuron
                if layer > 0:
                    npos = 0
                    for weight in self.brain.weights[layer - 1][act]:
                        # only draw strong weights
                        if abs(weight) > 0:
                            if np.sign(weight) < 0:
                                c = (255, 0, 0)
                            else:
                                c = (0, 255, 0)
                            start = (
                            x + h_offset - 48, y + old_v - 40 * len(self.brain.activations[layer - 1]) + npos + 13)
                            end = (x + h_offset, y + v_offset + 13)
                            pygame.draw.aaline(self.surface, c, start, end)

                        npos += 40
                v_offset += 40
            h_offset += 70

    def mutate(self, mut_rate=0.05):
        """Mutates weights and biases of creature with a chance of <mut_rate>"""

        # mutating weights
        for w_mat in range(0, len(self.brain.weights)):
            for layer in range(0, len(self.brain.weights[w_mat])):
                for weight in range(0, len(self.brain.weights[w_mat][layer])):
                    # rolling mutation chance
                    if np.random.random_sample() < mut_rate:
                        # rolling mutation type (addition, multiplication, replacement)
                        mutroll = np.random.random_sample()
                        if mutroll < 0.33:
                            self.brain.weights[w_mat][layer][weight] += 30*(np.random.random_sample()*2-1)

                        elif mutroll < 0.66:
                            self.brain.weights[w_mat][layer][weight] *= 2*(np.random.random_sample()*2-1)

                        else:
                            self.brain.weights[w_mat][layer][weight] = 1*(np.random.random_sample()*2-1)

        # crossing over biases
        for bias_layer in range(0, len(self.brain.biases)):
            for bias in range(0, len(self.brain.biases[bias_layer])):

                # rolling mutation chance
                if np.random.random_sample() < 0.5:
                    if np.random.random_sample() < mut_rate:
                        # rolling mutation type (addition, multiplication, replacement)
                        mutroll = np.random.random_sample()
                        if mutroll < 0.3333:
                            self.brain.biases[bias_layer][bias] += 2 * (np.random.random_sample() * 2 - 1)

                        elif mutroll < 0.6666:
                            self.brain.biases[bias_layer][bias] *= 2 * (np.random.random_sample() * 2 - 1)

                        else:
                            self.brain.biases[bias_layer][bias] = 4 * (np.random.random_sample() * 2 - 1)

    def crossover(self, creature):
        """Swaps genes between 2 species, i.e breeds them.can only crossover with same subclass/'species'. Randomly
        swap weights and biases for each neuron. Results in 2 'children' between self and the creature.
        Returns the 2 'children' in a list"""

        # crossing over weights
        for w_mat in range(0, len(self.brain.weights)):
            for layer in range(0,len(self.brain.weights[w_mat])):
                # roll 50% chance to crossover
                if np.random.random_sample() < 0.3:
                    a = self.brain.weights[w_mat][layer]
                    b = creature.brain.weights[w_mat][layer]

                    self.brain.weights[w_mat][layer] = b
                    creature.brain.weights[w_mat][layer] = a

        # crossing over biases
        for bias_layer in range(0, len(self.brain.biases)):
            for bias in range(0, len(self.brain.biases[bias_layer])):
                if np.random.random_sample() < 0.5:
                    a = self.brain.biases[bias_layer][bias]
                    b = creature.brain.biases[bias_layer][bias]

                    self.brain.biases[bias_layer][bias] = b
                    creature.brain.biases[bias_layer][bias] = a

    def set_fitness(self):
        pass


class BowCreature(Creature):
    """Represents a bow controlled by a NN. Input and output layers are set up like this:
    -----------------------------------------------------------------
    O = neuron

    ==Input Layer==                                 ==Output Layer==
    O - bow strength x
    O - bow strength y
    O - enemy's x position                           O - mouse x position
    O - enemy's y position   ..................      O - mouse y position
    O - enemy's dx                                   O - mouse clicked (if activation > 0.7)
    O - enemy's dy
    """

    def __init__(self, surface, bow):
        super().__init__(surface)
        self.brain = NN.NN([6, 4, 3])
        self.game_model = bow
        self.threshold = 0.9
        self.killed = False

    def copy_creature(self):
        new_c = BowCreature(self.surface, GameObjects.Bow(self.surface))
        new_c.time = self.time
        new_c.brain = self.brain.copy_NN()
        new_c.fitness = self.fitness

        return new_c

    def set_fitness(self, disp):
        """f(time(exponential decay), **arrows left(lognormal dist. on time), closest arrow distance to enemy)
        put in desmos to see:

        arrow component: 200\frac{1}{xb}e^{\left(-\frac{\left(\ln x-u\right)^{2}}{2b^{2}}\right)}
            - b = 0.5
            - u = 2.8
            
        time component: 6.5\cdot1.03^{\left(-x+a\right)}
            - c = 0.06
            - a = 28
        """
        if disp < 0:
            return 0

        # calculate fitness on average arrow distance to enemy per tick
        d_comp = 9000/(disp + 10)
        tcomp = 10 * np.exp(0.1*(-1*self.time + 28))

        if self.killed:
            kcomp = 8*(np.random.random_sample()*2 - 1) + 0

        else:
            kcomp = 5*(np.random.random_sample()*2 - 1) - 0

        self.fitness = d_comp + tcomp + kcomp
        return self.fitness

    def update_nn(self, x, y, dx, dy):
        inp_layer = [-self.game_model.draw_dx / 80, -self.game_model.draw_dy / 20]
        inp_layer.extend([(x - 255) /50, (y - 255) /50, (dx) * 3, (dy) * 3])

        self.brain.classify(inp_layer)

    def move_model(self, out):
        self.time += 0.05

        # [1*(self.brain.activations[-1][0]*300 + 100), (self.brain.activations[-1][1]*200 + 598)]
        inp = out
        inp[0] = inp[0]*300 + 100
        inp[1] = inp[1]*100 + 598
        self.game_model.pull_bow(False, inp)

        # if self.brain.activations[-1][2] > 0.7
        if out[-1] > self.threshold:

            mp = self.game_model.drawpos
            dx = (247-mp[0])/20
            dy = (573-mp[1])/20
            spd = np.sqrt(np.square(dx) + np.square(dy))

            if spd > 5:
                arrow = self.game_model.shoot(mp, dx, dy)
                return arrow

        return None


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
        self.brain = NN.NN([8, 5, 4])
        self.game_model = enemy
        self.avoided = 0

    def copy_creature(self):
        new_c = SquareCreature(self.surface, GameObjects.Enemy(self.surface, pos=(250,250)))
        new_c.time = self.time
        new_c.brain = self.brain.copy_NN()
        new_c.fitness = self.fitness

        return new_c

    def update_nn(self, bowx, bowy):
        inp_layer = [bowx/20, bowy / 100, (self.game_model.pos[0]-255)/80, (self.game_model.pos[1]-255)/50]
        for q in self.game_model.quadstat:
            if q:
                inp_layer.append(-200)
            else:
                inp_layer.append(200)
        self.brain.classify(inp_layer)

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

    def __init__(self, surface, b_config, e_config, mutation_rate=0.05):
        self.generations = 1
        self.mut_rate = mutation_rate
        self.surface = surface

        # neat-python variables
        self.b_conf = b_config
        self.e_conf = e_config

        self.b_pop = neat.Population(b_config)
        self.e_pop = neat.Population(e_config)

        self.b_brains = [neat.nn.FeedForwardNetwork.create(self.b_pop.population[i], b_config) for i in self.b_pop.population]
        self.e_brains = [neat.nn.FeedForwardNetwork.create(self.e_pop.population[i], e_config) for i in self.e_pop.population]

        self.best_bcreature = None
        self.best_ecreature = None
        self.best_b_brain = None
        self.best_e_brain = None
        self.best_manager = None

        self.timer = 0


        # game logic stuff
        self.bow_gen = [BowCreature(self.surface, GameObjects.Bow(self.surface))
                        for i in range(0, len(self.b_pop.population))]
        self.enemy_gen = [SquareCreature(self.surface, GameObjects.Enemy(self.surface, 100, (250, 250))) for i in
                          range(0, len(self.e_pop.population))]

        self.managers = [GameObjects.WaveManager([self.enemy_gen[i].game_model], self.bow_gen[i].game_model)
                         for i in range(0, len(self.bow_gen))]


        self.timers = [0 for i in range(0, len(self.bow_gen))]


        # UI stuff
        self.stfont = pygame.font.SysFont('Consolas', 15)
        self.drawing = True

    def view_most_fit(self):
        """shows game between most fit bow against most fit enemy"""

        if self.timer < 100 and self.generations > 1 and len(self.best_manager.enemies) > 0:
            b = self.best_manager.bow
            e = self.best_manager.enemies[0]
            self.best_manager.detectCollisions()

            # setting input for enemy neural networks
            e_inp = [b.draw_dx, b.draw_dy, (e.pos[0] - 255), (e.pos[1] - 255)]
            for q in e.quadstat:
                if q:
                    e_inp.append(100)
                else:
                    e_inp.append(0)
            self.best_e_brain.activate(e_inp)

            # setting input for bow neural networks
            b_inp = [b.draw_dx / 80, b.draw_dy]
            b_inp.extend([(e.pos[0] - 255), (e.pos[1] - 255), e.vel[0] * 3, e.vel[1] * 3])
            self.best_b_brain.activate(b_inp)

            # drawing game objects
            self.best_manager.draw()

            # listening for NN interaction, and updating game models
            b_out = [self.best_b_brain.values[ind] for ind in range(0, 3)]
            e_out = [self.best_e_brain.values[ind] for ind in range(0, 4)]
            self.best_ecreature.move_model(e_out)

            arrow = self.best_bcreature.move_model(b_out)
            if arrow is not None:
                self.best_manager.add_arrow(arrow)

            # countdown when stuck/unstuck, ending timer when enemy dies
            if not e.is_moving() or self.best_manager.failed_shots > 15 or b.arrows > 18 or b.time_between_shots > 500:
                self.timer += 0.2
            elif len(self.best_manager.enemies) < 1:
                self.timer = 200
            else:
                self.timer = 0

        else:
            self.best_bcreature = BowCreature(self.surface, GameObjects.Bow(self.surface))
            self.best_ecreature = SquareCreature(self.surface, GameObjects.Enemy(self.surface, 100, (250, 250)))
            self.best_manager = GameObjects.WaveManager([self.best_ecreature.game_model],
                                                        self.best_bcreature.game_model)
            self.timer = 0

    def multicycle(self):
        """Runs a game cycle on multiple enemies and bows"""
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

                # drawing game objects, timer bars
                if self.drawing:
                    self.managers[i].draw()
                    time_rect = pygame.Rect(860, 12 + 10*i, (100 - self.timers[i]) * 1.3, 5)
                    pygame.draw.rect(self.surface, (0, 255, 0), time_rect)
                else:
                    print("generation " + (str(self.generations)))

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
                    self.timers[i] = 200
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
        self.enemy_gen = [SquareCreature(self.surface, GameObjects.Enemy(self.surface, 100, (250, 250))) for i in
                          range(0, len(self.e_pop.population))]
        self.managers = [GameObjects.WaveManager([self.enemy_gen[i].game_model], self.bow_gen[i].game_model)
                         for i in range(0, len(self.bow_gen))]
        self.timers = [0 for i in range(0, len(self.bow_gen))]
        self.b_brains = [neat.nn.FeedForwardNetwork.create(self.b_pop.population[i], self.b_conf) for i in self.b_pop.population]
        self.e_brains = [neat.nn.FeedForwardNetwork.create(self.e_pop.population[i], self.e_conf) for i in self.e_pop.population]

        # updating best creatures
        self.best_b_brain = neat.nn.FeedForwardNetwork.create(self.b_pop.best_genome, self.b_conf)
        self.best_e_brain = neat.nn.FeedForwardNetwork.create(self.e_pop.best_genome, self.e_conf)
        self.best_bcreature = BowCreature(self.surface, GameObjects.Bow(self.surface))
        self.best_ecreature = SquareCreature(self.surface, GameObjects.Enemy(self.surface, 100, (250, 250)))
        self.best_manager = GameObjects.WaveManager([self.best_ecreature.game_model],
                                                    self.best_bcreature.game_model)
        self.timer = 0
        self.generations += 1

    def bow_ff(self, genomes, config):
        """bow fitness function for neat-python. Updates NEAT's genome fitnesses"""
        print(len(genomes))
        print(len(self.bow_gen))

        i = 0
        for b_id, genome in genomes:
            genome.fitness = self.bow_gen[i].set_fitness(self.managers[i].avg_dist)
            i += 1

    def enemy_ff(self, genomes, config):
        """Enemy fitness function for neat-python. Updates NEAT's genome fitnesses"""

        i = 0
        for e_id, genome in genomes:
            genome.fitness = self.enemy_gen[i].set_fitness()
            i += 1

    def drawstats(self):
        gen_lbl = self.stfont.render("generation " + str(self.generations), False, (0, 0, 0))
        self.surface.blit(gen_lbl, (1020, 10))
        b_str = "best bow fitness: "
        e_str = "best enemy fitness: "

        if self.b_pop.best_genome is not None and self.e_pop.best_genome is not None:
            b_str += str(self.b_pop.best_genome.fitness)
            e_str += str(self.e_pop.best_genome.fitness)

        b_fit_lbl = self.stfont.render(b_str, False, (0, 0, 0))
        e_fit_lbl = self.stfont.render(e_str, False, (0, 0, 0))
        self.surface.blit(b_fit_lbl, (1150, 10))
        self.surface.blit(e_fit_lbl, (1150, 25))

    def sort_by_fitness(self):
        self.enemy_gen.sort(key=lambda x: x.fitness, reverse=True)
        self.bow_gen.sort(key=lambda x: x.fitness, reverse=True)

    def select(self):
        """Selects and breeds the fittest creatures of the generation. Assume bowcreatures and squarecreatures are
        sorted by fitness in descending order."""
        # keep best performing individual of each generation
        new_bow_gen = [self.bow_gen[0].copy_creature()]
        new_e_gen = [self.enemy_gen[0].copy_creature()]

        # shuffle and breed top 50% of population
        bow_breed = self.bow_gen[0:1*len(self.bow_gen)//2]
        e_breed = self.enemy_gen[0:1*len(self.enemy_gen)//2]


        # breeding individuals with highest fitness, copy into a new generation
        i = 0
        while i < len(bow_breed)-2:
            bow_breed[i].crossover(bow_breed[i+1])
            e_breed[i].crossover(e_breed[i+1])

            new_bow_gen.extend([bow_breed[i].copy_creature(), bow_breed[i + 1].copy_creature()])
            new_e_gen.extend([e_breed[i].copy_creature(), e_breed[i+1].copy_creature()])
            i += 2
        # populating remainder of new generation with randomly generated creatures
        while len(new_e_gen) < self.pop_size:
            new_e_gen.append(SquareCreature(self.surface, GameObjects.Enemy(self.surface, 100, (250,250))))
            new_bow_gen.append(BowCreature(self.surface, GameObjects.Bow(self.surface)))

        # mutate everyone
        for i in range(0, len(new_e_gen)):
            new_e_gen[i].mutate(self.mut_rate)
            new_bow_gen[i].mutate(self.mut_rate)

        # assign new generations to evolution manager, update game managers
        self.enemy_gen = new_e_gen
        self.bow_gen = new_bow_gen
        self.managers = [GameObjects.WaveManager([self.enemy_gen[i].game_model], self.bow_gen[i].game_model) for i in
                         range(0, len(self.bow_gen))]

        # reset timers of all creatures after breeding
        self.timers = [0 for i in range(0, self.pop_size)]
        self.generations += 1



