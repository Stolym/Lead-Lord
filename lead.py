#!/usr/bin/env python3

#import tensorflow as tf
from Entitie import Entitie
from vector2d import vector2d
#from GameEngine import GameEngine
from Ent import Ent
from Env import Env
from Map import Map
from Memory import Memory
from Model import Model
from GameEngine import GameEngine
import numpy as np
import tensorflow as tf
import pygame
import time

'''
Rewards:

Par tour de boucle :

    R = (((Life * 3 + (Food / 20)) / 40) - 1)
    Si Food > 0:
        Food -= 1
    Si Food <= 0:
        Life -= 1

Mange une algue:
    Food += 20
Se déplace dans un rocher:
    Life -= 1
Se déplace le ventre vide:
    Life -= 1
'''


'''
class brain(object):
    def __init__(self):
        self.v_input = tf.placeholder(tf.float32, [2, 4, 4])
        self.s_input = tf.placeholder(tf.float32, [1, 5])
        self.sum_input = tf.add_n([self.s_input, self.v_input])


    def init_modal(self):
        layer_conv_a = tf.layers.conv2d(
            self.sum_input,
            filters=36,
            padding=1,
            kernel_size=2,
            strides=1
            activation=tf.nn.leaky_relu
        )
        max_pooling = tf.layers.max_pooling2d(
            layer_conv_a,
            pool_size = 2
            strides = 1
        )
'''

def initialize_object():

    model = Model(20, 20)
    memory = Memory(10000)
    return ([model, memory])

def initialize_entities():
    entities = []
    entities.append(Entitie(1, 8, 8))
    entities.append(Entitie(2, 10, 15))
    entities.append(Entitie(3, 7, 10))
    return entities

def inititalize_enviroment():
    object_utility = initialize_object()
    map = Map()
    entities = initialize_entities()
    return object_utility, map, entities


if __name__ == "__main__":
    #1. Initialisation de l'environement
    object_utility, map, entities = inititalize_enviroment()
    game = GameEngine(object_utility, map, entities, 100000)
    game.run()
''' 
    map.env.set_e_i_m(4, vector2d(5, 5))
    for i in map.env.matrix:
        print(np.asarray(i))
    for i in map.ent.matrix:
        print(np.asarray(i))
    print(map.batch_map())
    print(str(map.env.matrix))
'''    