import pygame
import numpy as np

class Enemy:
    """represents the target of the bow and arrow"""

    def __init__(self, hp=100):
        self.pos = [0, 0]
        self.vel = [0, 0]
        self.hp = hp
        self.model = pygame.Rect(200, -15, 15, 15)

    def move(self, time, m_funct=np.sin, yspeed=0.1):
        """Moves enemy, m_funct represents a function for how the x movement of enemy changes over time, must
        be in the range of (0,1)"""
        old = self.pos[:]
        self.pos[0] += m_funct(time)
        self.pos[1] += yspeed
        dx = self.pos[0] - old[0]

        if dx <= 0:
            dx = self.pos[0] - np.ceil(old[0])
        else:
            dx = self.pos[0] - np.floor(old[0])
        self.model.move_ip(dx, self.pos[1] - np.floor(old[1]))


class WaveManager:
    """Represents a wave of enemies"""

    def __init__(self, enemies):
        self.enemies = enemies
        self.arrows = []
        self.bow = Bow(int(len(enemies)*1.5 // 1))

    def detectCollisions(self):
        pass


class Bow:
    def __init__(self, surface, arrow_count=12):
        self.arrows = arrow_count
        self.draw_len = 0
        self.model = [pygame.image.load('sprites/bow.png')]
        self.surface = surface

    def pull_bow(self):
        mp = pygame.mouse.get_pos()

        pygame.draw.aaline(self.surface, (0, 0, 0), (193, 573), mp)
        pygame.draw.aaline(self.surface, (0, 0, 0), (303, 573), mp)
        pygame.draw.aaline(self.surface, (235, 64, 52), mp, (247 + 3*(247-mp[0]), 573 + 3*(573-mp[1])))

        ball = pygame.Rect(mp[0] - 6, mp[1] - 10, 16, 16)
        pygame.draw.ellipse(self.surface, (27, 163, 3), ball)

    def shoot(self, end_pos, dx, dy):
        """drawing arrow longer = take more time, faster arrow, slower fire rate, more distance and dmg
            shorter = less time, slower arrow & faster fire rate, less distance and dmg. Returns the arrow shot"""
        shot = Arrow(dx, dy, end_pos, self.surface)
        return shot

    def update(self):
        """updates the bow model"""
        self.surface.blit(self.model[0], (190, 550))

class Arrow:
    def __init__(self, dx, dy, pos, surface):
        self.dx = dx
        self.dy = dy
        self.pos = list(pos)
        self.surface = surface
        self.model = pygame.Rect(self.pos[0] - 6, self.pos[1] - 10, 16, 16)

    def move(self):

        old = self.pos[:]
        self.pos[0] += self.dx
        self.pos[1] += self.dy
        dx = self.pos[0] - old[0]
        dy = self.pos[1] - old[1]

        if dx <= 0:
            dx = self.pos[0] - np.ceil(old[0])
        else:
            dx = self.pos[0] - np.floor(old[0])

        if dy <= 0:
            dy = self.pos[1] - np.ceil(old[1])
        else:
            dy = self.pos[1] - np.floor(old[1])
        self.model.move_ip(dx, dy)
        pygame.draw.ellipse(self.surface, (27, 163, 3), self.model)


def default(t):
    return 0