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
manager = GameObjects.WaveManager([e], bow)

enemy = Evolution.SquareCreature(screen, e)
bow_ai = Evolution.BowCreature(screen, bow)

e_manager = Evolution.EvolutionManager(screen)

clock = pygame.time.Clock()
# game loop
while True:
    screen.fill((255, 255, 255))
    mousepos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            e_manager.drawing = True

        if event.type == pygame.MOUSEBUTTONDOWN:
            e_manager.drawing = False

    # ========= updating screen==============
    # draw dividers, and generation info
    pygame.draw.aaline(screen, (0, 0, 0), (550, 0), (550, 700))
    pygame.draw.aaline(screen, (0, 0, 0), (1000, 0), (1000, 700))
    pygame.draw.aaline(screen, (0, 0, 0), (850, 0), (850, 700))
    pygame.draw.aaline(screen, (0, 0, 0), (550, 340), (850, 340))
    header.render('Slingshot', False, (0, 0, 0))
    enemy_label = header.render('Enemy', False, (0, 0, 0))
    bow_label = header.render('Slingshot', False, (0, 0, 0))

    e_manager.cycle()

    screen.blit(enemy_label, (670, 20))
    screen.blit(bow_label, (670, 390))
    pygame.display.flip()
    pygame.display.update()

    # set framerate
    clock.tick(300)



