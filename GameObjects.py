import pygame
import numpy as np


# ================= Movement functions to use ========================
def defaultx(x):
    return 0


def defaulty(y):
    return 0


def jitter(y):
    """eventually increasing function, use for y only"""
    return 0.1

def periodic(x):
    """random periodic function for x"""

    return (np.sin(0.5*x) + np.sin(2*x) + np.sin(3*x))/4


def rand_funct():
    """create a random periodic function for x movement"""
    def funct(x):
        b = np.random.random_sample()+1
        c = np.random.random_sample()+1

        return np.sin(x/100) + np.cos((x/100)*b)+ np.sin((x/100)*c)

    return funct


class Enemy:
    """represents the target of the bow and arrow, x and y funct represent function describing movement of enemy
    with respect to time."""

    def __init__(self, surface, hp=100, pos=(0, 0), x_funct=defaultx, y_funct=defaulty):
        self.pos = list(pos)
        self.vel = [0, 0]
        self.hp = hp
        self.model = pygame.Rect(self.pos[0] - 7, self.pos[1] - 7, 15, 15)
        self.surface = surface
        self.accel = 0.012
        self.fx = x_funct
        self.fy = y_funct

    def move(self, time):
        """Moves enemy, m_funct represents a function for how the x movement of enemy changes over time, must
        be in the range of (0,1), and updates drawing"""

        old = self.pos[:]
        self.pos[0] += self.fx(time)
        self.pos[1] += self.fy(time)
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

        self.vel = [dx, dy]
        self.model.move_ip(dx, dy)

    def player_control(self):
        """control enemy with arrow buttons"""
        keys = pygame.key.get_pressed()

        # check if accelerating in x and y
        ax = (keys[275] - keys[276])
        ay = (keys[274] - keys[273])

        # stop accelerating when at max velocity while pressing buttons
        if abs(self.vel[0]) > 0.9 and abs(ax) > 0 and np.sign(ax) == np.sign(self.vel[0]):
            ax = 0
        if abs(self.vel[1]) > 0.9 and abs(ay) > 0 and np.sign(ay) == np.sign(self.vel[1]):
            ay = 0

        dx = self.vel[0] + self.accel*ax
        dy = self.vel[1] + self.accel*ay

        # decelerate when keys are let go and object is moving
        if keys[275] < 1 and keys[276] < 1 and abs(self.vel[0]) > 0:
            dx = dx - np.sign(dx)*self.accel
        if keys[274] < 1 and keys[273] < 1 and abs(self.vel[1]) > 0:
            dy = dy - np.sign(dy)*self.accel

        print(self.pos)

        self.vel = [dx, dy]
        old = self.pos[:]
        vx = self.pos[0] + dx
        vy = self.pos[1] + dy

        self.pos[0] += dx
        self.pos[1] += dy

        if dx <= 0:
            dx = vx - np.ceil(old[0])
        else:
            dx = vx - np.floor(old[0])

        if dy <= 0:
            dy = vy - np.ceil(old[1])
        else:
            dy = vy - np.floor(old[1])

        if int(self.pos[0]) not in range(1, 499):
            dx = 0
            if self.pos[0] < 1:
                self.pos[0] = 2
            else:
                self.pos[0] = 498

            self.vel[0] = 0

        if int(self.pos[1]) not in range(1, 599):
            dy = 0
            if self.pos[1] < 1:
                self.pos[1] = 2
            else:
                self.pos[1] = 598

            self.vel[1] = 0

        self.model.move_ip(dx, dy)

    def draw(self):
        pygame.draw.rect(self.surface, (0, 0, 0), self.model)

class WaveManager:
    """Represents a wave of enemies"""

    def __init__(self, enemies):
        self.enemies = enemies
        self.arrows = []
        self.bow = Bow(int(len(enemies)*1.5 // 1))


    def detectCollisions(self):
        """detects if arrows hit any enemies, and despawns arrows out of bounds, updates arrows and enemies"""

        for arr in self.arrows:
            for e in self.enemies:
                # print('checking fired')
                if int(arr.pos[0]) in range(int(e.pos[0] - 20), int(e.pos[0] + 20)) and int(arr.pos[1]) in range(int(e.pos[1] - 20), int(e.pos[1] + 20)):
                    self.arrows.remove(arr)
                    self.enemies.remove(e)
                    # print('hit!')

        # update the survivors/arrow positions
        [a.move() for a in self.arrows]
        [e.player_control() for e in self.enemies]


    def draw(self):
        [a.draw() for a in self.arrows]
        [e.draw() for e in self.enemies]

    def add_enemy(self, enemy):
        """adds an enemy to the enemies list"""
        self.enemies.append(enemy)

    def add_arrow(self, arrow):
        """adds arrow to arrow list"""
        if arrow is not None:
            self.arrows.append(arrow)


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

        if self.arrows > 0:
            ball = pygame.Rect(mp[0] - 6, mp[1] - 10, 16, 16)
            pygame.draw.ellipse(self.surface, (27, 163, 3), ball)

    def shoot(self, end_pos, dx, dy):
        """drawing arrow longer = take more time, faster arrow, slower fire rate, more distance and dmg
            shorter = less time, slower arrow & faster fire rate, less distance and dmg. Returns the arrow shot"""
        if self.arrows > 0:
            shot = Arrow(dx, dy, end_pos, self.surface)
            self.arrows -= 1
            return shot

        return None

    def draw(self):
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

    def draw(self):
        pygame.draw.ellipse(self.surface, (27, 163, 3), self.model)
