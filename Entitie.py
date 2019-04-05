#!/usr/bin/env python3

from vector2d import vector2d
import numpy as np

class Entitie(object):
    def __init__(self, id, x=0, y=4):
        self.id = id
        self.pos = vector2d(x, y)
        self.vision = 4
        self.life = 20
        self.food = 100
        self.evolv = 0
        self.type = 1 #Player == 1 || Poisson == 2 || Algue == 3
        self.alim = 0 #Herbivore == -1 || Carnivore == 1

    def getVision(self):
        return self.vision

    def setVision(self, newvision):
        self.vision = newvision

    def getPos(self):
        return self.pos

    def setPos(self, newpos):
        self.pos = newpos

    def getLife(self):
        return self.life

    def setLife(self, newlife):
        self.life = newlife

    def getReward(self, v):
        reward = self.life * 2 + self.food - 1
        return reward

    def get_vision(self, map_env, map_ent):
        #print("L'ENTITE:", self.pos.x, self.pos.y, map_ent[self.pos.y][self.pos.x])        
        v = np.zeros((self.vision * 2 + 1, self.vision * 2 + 1, 2))
        for i in range(self.vision * 2 + 1):
            for n in range(self.vision * 2 + 1):
                if (self.pos.x - self.vision + n < 0 or
                self.pos.x - self.vision + n > 19 or
                self.pos.y - self.vision + i < 0 or
                self.pos.y - self.vision + i > 19):
                    v[i][n][0] = -1
                    v[i][n][1] = -1
                else:
                    v[i][n][0] = map_env[self.pos.y - self.vision + i][self.pos.x - self.vision + n]
                    v[i][n][1] = map_ent[self.pos.y - self.vision + i][self.pos.x - self.vision + n]
        return v

    def move(self, env, vect):
        newpos = vector2d(self.pos.x + vect.x, self.pos.y + vect.y)

        if (newpos.x < 0):
            newpos.x = 19
        if (newpos.y < 0):
            newpos.y = 19
        newpos.x = newpos.x % 20
        newpos.y = newpos.y % 20
        if (env.get_e_i_m(self.pos) == 1):
            env.set_e_i_m(1, newpos)
            env.del_e_i_m(self.pos)
        self.pos = newpos

    #Matrix = Map
    def eat(self, map, vect):
        for i in map.ent.matrix[self.pos.x][self.pos.y]:
            if (map.ent.matrix[self.pos.x][self.pos.y] == 2 or map.ent.matrix[self.pos.x][self.pos.y] == 3):
                map.ent.del_e_i_m()
