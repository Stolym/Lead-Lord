#!/usr/bin/env python3

class Ent(object):
    def __init__(self, matrix):
        
        for y in matrix:
            for x in y:
                x = []
                x.append(0)
        self.matrix = matrix
        """
        tmp = []
        tmp_b  = np.zeros((20, 20))
        tmp.append(self.matrix)
        tmp.append(tmp_b)
        print(tmp)
        """
    def set_e_i_m(self, id, pos):
        self.matrix[pos.y][pos.x] = id
    def del_e_i_m(self,pos):
        self.matrix[pos.y][pos.x] = 0
        """
        for i in range(self.matrix[pos.x][pos.y]):
            if (self.matrix[pos.x][pos.y] == id):
                self.matrix[pos.x][pos.y] = 0
                return 0
        return 1
        """
    def get_e_i_m(self, pos):
        return self.matrix[pos.y][pos.x]