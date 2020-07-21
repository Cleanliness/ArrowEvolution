import neat
import os

enemypath = os.path.join("enemy-config-feedforward.txt")

conf = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, enemypath)
pop = neat.Population(conf)

for i in pop.population:
    b = pop.population[i]

