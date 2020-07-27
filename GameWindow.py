import pygame
import numpy as np
import Evolution
import sys
import os
import neat


ENEMY_PATH = os.path.join("enemy-config-feedforward.txt")
BOW_PATH = os.path.join("bow-config-feedforward.txt")
e_conf = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, ENEMY_PATH)
b_conf = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, BOW_PATH)


# game setup
pygame.init()
pygame.font.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((1400, 700))
header = pygame.font.SysFont('Times New Roman', 25)
bow_img = pygame.image.load('sprites/bow.png').convert_alpha()


e_manager = Evolution.EvolutionManager(screen, b_conf, e_conf)

view_fit = False
# game loop

while True:
    screen.fill((255, 255, 255))
    mousepos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            view_fit = not view_fit

    # ========= updating screen==============
    # draw dividers and ui
    pygame.draw.aaline(screen, (0, 0, 0), (550, 0), (550, 700))
    pygame.draw.aaline(screen, (0, 0, 0), (1000, 0), (1000, 700))
    pygame.draw.aaline(screen, (0, 0, 0), (850, 0), (850, 700))
    pygame.draw.aaline(screen, (0, 0, 0), (550, 340), (850, 340))
    header.render('Slingshot', False, (0, 0, 0))
    enemy_label = header.render('Enemy', False, (0, 0, 0))
    bow_label = header.render('Slingshot', False, (0, 0, 0))

    e_manager.multicycle()

    screen.blit(enemy_label, (670, 20))
    screen.blit(bow_label, (660, 360))
    screen.blit(bow_img, (190, 550))
    pygame.display.flip()
    pygame.display.update()

    # set framerate
    clock.tick(300)


