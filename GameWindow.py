import pygame
import GameObjects
import numpy as np
import sys

# game setup
pygame.init()
screen = pygame.display.set_mode((500, 700))

fun = GameObjects.rand_funct()

e = GameObjects.Enemy(screen, 100, (250, 250))
bow = GameObjects.Bow(screen)
manager = GameObjects.WaveManager([e])

r = e.model

arrlist = []
# game loop
while True:
    screen.fill((255, 255, 255))
    mousepos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEBUTTONUP and int(mousepos[0]) in range(0, 500) and int(mousepos[1]) in range(500, 700):
            # print('fire')
            mp = bow.drawpos
            dx = (247-mp[0])/35
            dy = (573-mp[1])/35

            manager.add_arrow(bow.shoot(mp, dx, dy))


    # update screen
    manager.detectCollisions()
    bow.draw()
    manager.draw()
    e.player_control()

    if pygame.mouse.get_pressed()[0] and int(mousepos[0]) in range(0, 500) and int(mousepos[1]) in range(500, 700):
        bow.pull_bow()

    pygame.display.flip()
    pygame.display.update()



