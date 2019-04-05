#!/usr/bin/env python3

from Ent import Ent
from Env import Env
import vector2d
import pygame
import numpy as np

Env_map = [[1,1,1,1,1,1,1,1,1,2,2,2,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,2,2,2,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1],
        [1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1],
        [1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,1],
        [1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,1],
        [1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1],
        [1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1],
        [1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1],
        [1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1],
        [1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1],
        [1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,1,1,1,1,1],
        [1,1,1,1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,1,1],
        [1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1],
        [1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,1,1,1,1,1],
        [1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,2,2,2,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1],]

class Map(object):
    def __init__(self):
        self.env = Env(Env_map)
        self.ent = Ent(self.generate_map())
        #Ent(self.generate_map())

    #Fonction obsolète
    '''
    def batch_map(self):
        tmp = []
        tmp.append(self.ent.matrix)
        tmp.append(self.env.matrix)
        return tmp
    '''
    
    def generate_map(self, type = "Ent"):
        if (type == "Ent"):
            return np.zeros((20, 20))
    #Fonction obsolète
'''
    def aff_map(self, window, grass, mountain, water):
        stp = 27.5
        global Cam
        Cam = vector2d.vector2d(0, 0)
        pygame.draw.rect(window, 0, (0, 0, 1200, 800))
        for y in range(len(self.env.matrix)):
            for x in range(len(self.env.matrix[y])):
                vect = vector2d.vector2d(Cam.x + 600 - stp * y + x * stp, Cam.y + y * stp / 2 + x * stp / 2)
                if (self.env.matrix[y][x] == 1):
                    window.blit(grass, (vect.x, vect.y))
                if (self.env.matrix[y][x] == 2):
                    window.blit(water, (vect.x, vect.y))
                if (self.ent.matrix[y][x] == 1):
                    window.blit(player, (vect.x, vect.y))

'''
    #Fonction obsolète
'''
    def start(self):
        Graph = 1
        run = 1
        global grass, mountain, player, water
        if Graph == 1:
            window = pygame.display.set_mode((1200, 800))
            grass = pygame.image.load("grass.png").convert_alpha()
            grass = pygame.transform.scale(grass, (55, 55))
            mountain = pygame.image.load("mountain.png").convert_alpha()
            mountain = pygame.transform.scale(mountain, (55, 55))
            water = pygame.image.load("water.png").convert_alpha()
            water = pygame.transform.scale(water, (55, 55))
            player = pygame.image.load("player.png").convert_alpha()
            player = pygame.transform.scale(water, (55, 55))
            while (run == 1):
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            run = 0
                        if event.key == pygame.K_LEFT:
                            Cam.x += 25
                        if event.key == pygame.K_RIGHT:
                            Cam.x -= 25
                        if event.key == pygame.K_UP:
                            Cam.y += 25
                        if event.key == pygame.K_DOWN:
                            Cam.y -= 25
                self.aff_map(window, grass, mountain, water)
                pygame.display.flip()'''