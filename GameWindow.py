import pygame
import GameObjects
import Evolution
import sys

# game setup
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((1400, 700))
header = pygame.font.SysFont('Times New Roman', 25)

e = GameObjects.Enemy(screen, 100, (250, 250))
bow = GameObjects.Bow(screen)
manager = GameObjects.WaveManager([e])

enemy = Evolution.SquareCreature(screen, e)
bow_ai = Evolution.BowCreature(screen, bow)

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
            dx = (247-mp[0])/45
            dy = (573-mp[1])/45

            manager.add_arrow(bow.shoot(mp, dx, dy))

    # ========= updating screen==============
    # draw dividers, and generation info
    pygame.draw.aaline(screen, (0, 0, 0), (550, 0), (550, 700))
    pygame.draw.aaline(screen, (0, 0, 0), (850, 0), (850, 700))
    pygame.draw.aaline(screen, (0, 0, 0), (550, 340), (850, 340))
    header.render('Slingshot', False, (0, 0, 0))
    enemy_label = header.render('Enemy', False, (0, 0, 0))
    bow_label = header.render('Slingshot', False, (0, 0, 0))

    # drawing game objects
    manager.detectCollisions()
    bow.draw()
    manager.draw()
    e.player_control()
    bow.pull_bow()

    # drawing + updating neural nets
    enemy.update_nn(bow.draw_dx, bow.draw_dy)
    bow_ai.update_nn(e.pos[0], e.pos[1], e.vel[0], e.vel[1])
    enemy.drawNN()
    bow_ai.drawNN(600, 420)

    screen.blit(enemy_label, (670, 20))
    screen.blit(bow_label, (670, 390))
    pygame.display.flip()
    pygame.display.update()



