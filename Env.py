#!/usr/bin/env python3

class Env(object):
    def __init__(self, matrix):
        self.matrix = matrix
    def set_e_i_m(self, id, pos):
        self.matrix[pos.y][pos.x] = id
    def del_e_i_m(self, pos):
        self.matrix[pos.y][pos.x] = 0
    def get_e_i_m(self, pos):
        return self.matrix[pos.y][pos.x]