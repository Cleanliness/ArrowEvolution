import pygame
import GameObjects
import sys

# game setup
pygame.init()
screen = pygame.display.set_mode((500, 700))


e = GameObjects.Enemy()
b = GameObjects.Bow(screen)
r = e.model
t = 0

arrlist = []
# game loop
while True:
    screen.fill((255, 255, 255))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEBUTTONUP:
            print('fire')
            mp = pygame.mouse.get_pos()
            dx = (247-mp[0])/30
            dy = (573-mp[1])/30
            arrlist.append(b.shoot(mp, dx, dy))

    e.move(t)
    pygame.draw.rect(screen, (0, 0, 0), r)

    for a in arrlist:
        a.move()

    t += 0.01
    # update screen
    b.update()
    if pygame.mouse.get_pressed()[0]:
        b.pull_bow()

    pygame.display.flip()
    pygame.display.update()



