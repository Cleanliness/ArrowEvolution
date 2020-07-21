import pygame
import numpy as np


# ================= Movement functions to use ========================
def defaultx(x):
    return 0


def defaulty(y):
    return 0


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

    def __init__(self, surface, hp=100, pos=(0, 0)):
        self.pos = list(pos)
        self.vel = [0, 0]
        self.hp = hp
        self.model = pygame.Rect(self.pos[0] - 7, self.pos[1] - 7, 15, 15)
        self.quads = [pygame.Rect(self.pos[0] - 50, self.pos[1] - 50, 50, 50),
                      pygame.Rect(self.pos[0] - 50, self.pos[1]-1, 50, 50),
                      pygame.Rect(self.pos[0]-1, self.pos[1] - 50, 50, 50),
                      pygame.Rect(self.pos[0] - 1, self.pos[1] - 1, 50, 50)]
        self.quadborder = self.quads[:]
        self.quadstat = [False for i in range(0, 4)]
        self.surface = surface
        self.accel = 0.05

    def copy_enemy(self):
        new_e = Enemy(self.surface, self.hp, self.pos[:])
        new_e.vel = self.vel[:]
        new_e.model = pygame.Rect(self.pos[0] - 7, self.pos[1] - 7, 15, 15)
        new_e.quads = [pygame.Rect(self.pos[0] - 50, self.pos[1] - 50, 50, 50),
                      pygame.Rect(self.pos[0] - 50, self.pos[1] - 1, 50, 50),
                      pygame.Rect(self.pos[0] - 1, self.pos[1] - 50, 50, 50),
                      pygame.Rect(self.pos[0] - 1, self.pos[1] - 1, 50, 50)]
        new_e.quadborder = self.quads[:]
        new_e.quadstat = self.quadstat[:]

    def player_control(self, mouse=True, inp=(0, 0, 0, 0)):
        """control enemy with arrow buttons"""

        if mouse:
            allkeys = pygame.key.get_pressed()
            keys = [allkeys[275], allkeys[276], allkeys[274], allkeys[273]]

        else:
            keys = list(inp)

        # check if accelerating in x and y
        ax = (keys[0] - keys[1])
        ay = (keys[2] - keys[3])

        # stop accelerating when at max velocity while pressing buttons
        if abs(self.vel[0]) > 1.5 and abs(ax) > 0 and np.sign(ax) == np.sign(self.vel[0]):
            ax = 0
        if abs(self.vel[1]) > 1.5 and abs(ay) > 0 and np.sign(ay) == np.sign(self.vel[1]):
            ay = 0

        dx = self.vel[0] + self.accel*ax
        dy = self.vel[1] + self.accel*ay

        # decelerate when keys are let go and object is moving
        if keys[0] < 1 and keys[1] < 1 and abs(self.vel[0]) > 0:
            dx = dx - np.sign(dx)*self.accel
        if keys[2] < 1 and keys[3] < 1 and abs(self.vel[1]) > 0:
            dy = dy - np.sign(dy)*self.accel

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

        # stop moving when hitting a wall
        if int(self.pos[0]) not in range(1, 499):
            dx = 0
            if self.pos[0] < 1:
                self.pos[0] = 2
            else:
                self.pos[0] = 498

            self.vel[0] = 0

        if int(self.pos[1]) not in range(1, 499):
            dy = 0
            if self.pos[1] < 1:
                self.pos[1] = 2
            else:
                self.pos[1] = 498
            self.vel[1] = 0

        # TODO reimplement quadrant moving
        # really dumb way of updating positions, but pygame's move(dx,dy) doesn't work for some reason
        for i in range(0, len(self.quads)):
            self.quads = [pygame.Rect(self.pos[0] - 50, self.pos[1] - 50, 50, 50),
                          pygame.Rect(self.pos[0] - 50, self.pos[1] - 1, 50, 50),
                          pygame.Rect(self.pos[0] - 1, self.pos[1] - 50, 50, 50),
                          pygame.Rect(self.pos[0] - 1, self.pos[1] - 1, 50, 50)]
            self.quadborder = self.quads[:]

        self.model.move_ip(dx, dy)

    def check_quads(self, arrows):

        for i in range(0, len(self.quads)):
            tmp_pos = [int(self.quads[i].x), int(self.quads[i].y)]
            for pos in arrows:
                a_pos = [int(i) for i in pos]
                if a_pos[0] in range(tmp_pos[0], tmp_pos[0] + 50) and a_pos[1] in range(tmp_pos[1], tmp_pos[1] + 50):
                    self.quadstat[i] = True
                    break
                self.quadstat[i] = False

    def draw(self):
        # drawing quadrants of enemy + borders
        for i in range(0, len(self.quads)):
            c = (0, 255, 68)
            if self.quadstat[i]:
                c = (255, 0, 0)
            pygame.draw.rect(self.surface, c, self.quads[i])
            pygame.draw.rect(self.surface, (0, 0, 0), self.quads[i], 1)

        # draw main enemy body (black square), and dy/dx
        pygame.draw.rect(self.surface, (0, 0, 0), self.model)
        pygame.draw.aaline(self.surface, (0, 26, 255), self.pos, (self.pos[0] + 120*self.vel[0], self.pos[1] + 120*self.vel[1]))

    def is_moving(self):
        """returns if enemy is moving"""
        return not (int(abs(self.vel[0]*10)) in range(0, 3) and int(abs(self.vel[1]*10)) in range(0, 3))


class Bow:
    """Represents the slingshot"""
    def __init__(self, surface, arrow_count=0):
        self.arrows = arrow_count
        self.draw_len = 0
        self.model = [pygame.image.load('sprites/bow.png').convert_alpha()]
        self.surface = surface
        self.drawpos = [247, 573]
        self.draw_dx = 0
        self.draw_dy = 0

        self.time_between_shots = 0

    def copy_bow(self):
        new_bow = Bow(self.surface, self.arrows)
        new_bow.draw_len = self.draw_len
        new_bow.drawpos = self.drawpos[:]
        new_bow.draw_dx = self.draw_dx
        new_bow.draw_dy = self.draw_dy
        new_bow.time_between_shots = self.time_between_shots

        return new_bow

    def pull_bow(self, mouse=True, neuron_pos=(247, 573)):
        mousepos = pygame.mouse.get_pos()
        if mouse and not (pygame.mouse.get_pressed()[0] and int(mousepos[0]) in range(0, 500) and int(mousepos[1]) in range(500, 700)):
            return None

        if mouse:
            mp = pygame.mouse.get_pos()
        else:
            mp = neuron_pos

        # setting initial dx and dy
        if self.drawpos[0] == 247 and self.drawpos[1] == 573:
            dx = (247-np.sign(mp[0])*mp[0])*-1
            dy = (573-np.sign(mp[1])*mp[1])*-1

        else:
            dx = (self.drawpos[0]-np.sign(mp[0])*mp[0])*-1
            dy = (self.drawpos[1]-np.sign(mp[1])*mp[1])*-1

        # checking if ball pos is at mouse pos
        if int(self.drawpos[0]) in range(int(mp[0]) - 2, int(mp[0]) + 3) and int(self.drawpos[1]) in range(int(mp[1]) - 2, int(mp[1]) + 3):
            self.drawpos[0] = int(mp[0])
            self.drawpos[1] = int(mp[1])
            dx = 0
            dy = 0

        # pulling bow towards mouse
        if dx == 0:
            self.drawpos[1] += 5*np.sign(dy)

        else:
            theta = np.arctan(abs(dy/dx))
            self.drawpos[0] += 5 * np.cos(theta)*np.sign(dx)
            self.drawpos[1] += 5 * np.sin(theta)*np.sign(dy)

        # updating draw position of bow
        self.draw_dx = 247-self.drawpos[0]
        self.draw_dy = 573-self.drawpos[1]


    def shoot(self, end_pos, dx, dy):
        """drawing arrow longer = take more time, faster arrow, slower fire rate, more distance and dmg
            shorter = less time, slower arrow & faster fire rate, less distance and dmg. Returns the arrow shot"""
        shot = Arrow(dx, dy, end_pos, self.surface)
        self.arrows += 1
        self.drawpos = [247, 573]
        self.draw_dy, self.draw_dx = 0, 0
        self.time_between_shots = 0
        return shot

    def draw(self):
        """updates the bow model"""
        self.surface.blit(self.model[0], (190, 550))

        drawdx = 247 + (247-self.drawpos[0])
        drawdy = 573 + (573-self.drawpos[1])

        pygame.draw.aaline(self.surface, (0, 0, 0), (193, 573), self.drawpos)
        pygame.draw.aaline(self.surface, (0, 0, 0), (303, 573), self.drawpos)
        pygame.draw.aaline(self.surface, (235, 64, 52), self.drawpos, (drawdx, drawdy))

        ball = pygame.Rect(self.drawpos[0] - 6, self.drawpos[1] - 10, 16, 16)
        pygame.draw.ellipse(self.surface, (27, 163, 3), ball)
        self.time_between_shots += 1


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


class WaveManager:
    """Represents a wave of enemies"""

    def __init__(self, enemies, bow):
        self.enemies = enemies
        self.arrows = []
        self.bow = bow
        self.failed_shots = 0
        self.closest_distance = [99999999999999999, 99999999999999999]

    def detectCollisions(self):
        """detects if arrows hit any enemies, and despawns arrows out of bounds, updates arrows and enemies"""

        # removes newest arrow if too many onscreen, +1 failed shot
        if len(self.arrows) > 24:
            self.arrows.pop(-1)
            self.bow.arrows -= 1
            self.failed_shots += 1

        # checking for collisions
        for arr in self.arrows:
            for e in self.enemies:
                # remove arrow if found in range of enemy, i.e enemy killed
                if int(arr.pos[0]) in range(int(e.pos[0] - 15), int(e.pos[0] + 15)) and int(arr.pos[1]) in range(int(e.pos[1] - 15), int(e.pos[1] + 15)):
                    self.arrows.remove(arr)
                    self.enemies.remove(e)

                # calculate distance between arrow and enemy, update closest distance if smaller
                else:
                    dx = e.pos[0] - arr.pos[0]
                    dy = e.pos[1] - arr.pos[1]

                    if abs(dx) < self.closest_distance[0] and abs(dy) < self.closest_distance[1]:
                        self.closest_distance = [abs(dx), abs(dy)]
                        print(self.closest_distance)
            # despawn offscreen arrows
            if not (int(arr.pos[0]) in range(-100, 560) and int(arr.pos[1]) in range(-100, 800)):
                self.arrows.remove(arr)

        for e in self.enemies:
            e.check_quads([a.pos for a in self.arrows])
        # update the survivors/arrow positions
        [a.move() for a in self.arrows]

    def draw(self):
        self.bow.draw()
        [e.draw() for e in self.enemies]
        [a.draw() for a in self.arrows]


    def add_enemy(self, enemy):
        """adds an enemy to the enemies list"""
        self.enemies.append(enemy)

    def add_arrow(self, arrow):
        """adds arrow to arrow list"""
        if arrow is not None:
            self.arrows.append(arrow)
