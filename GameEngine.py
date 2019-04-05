#!/usr/bin/env python3

from vector2d import vector2d
from Map import Map as m
from Entitie import Entitie
from Ent import Ent
from Env import Env
from Memory import Memory
from Model import Model
import numpy as np
import tensorflow as tf
import pygame
import time
from random import randint

size_memory = 1000


"""

FOOD = 100
LIFE = 20

"""


class GameEngine:
    def __init__(self, object, Map, entities, size_step = 1000):
        self.model = object[0]
        #self.memory = object[1]
        self.Map = Map
        self.entities = entities
        self.size_step = size_step
        self.memory = []
        self.all_rewards = []


    #[Graphique] Initialisation des images pour l'affichage.
    def init_graphical(self):
        self.window = pygame.display.set_mode((1200, 800))
        self.font = pygame.font.SysFont("monospace", 15)
        self.grass = pygame.image.load("grass.png").convert_alpha()
        self.grass = pygame.transform.scale(self.grass, (55, 55))
        self.mountain = pygame.image.load("mountain.png").convert_alpha()
        self.mountain = pygame.transform.scale(self.mountain, (55, 55))
        self.water = pygame.image.load("water.png").convert_alpha()
        self.water = pygame.transform.scale(self.water, (55, 55))
        self.player = pygame.image.load("player.png").convert_alpha()
        self.player = pygame.transform.scale(self.player, (35, 35))
        self.fish = pygame.image.load("fish.png").convert_alpha()
        self.fish = pygame.transform.scale(self.fish, (35, 35))
        self.seaweed = pygame.image.load("seaweed.png").convert_alpha()
        self.seaweed = pygame.transform.scale(self.seaweed, (35, 35))
    
    #[Graphique] Affichage des entités (joueur, poisson, algue)
    def aff_entites(self, y, x, vect):
        for ent in self.entities:
            if (ent.pos.x == x and ent.pos.y == y):
                if (ent.id == 1):
                    self.window.blit(self.player, (vect.x+8, vect.y-20))
                if (ent.id == 2):
                    self.window.blit(self.fish, (vect.x+8, vect.y-20))
                if (ent.id == 3):
                    self.window.blit(self.seaweed, (vect.x+8, vect.y-20))

    #[Graphique] Affichage des points de vie et de la nourriture du joueur
    def aff_label(self):
        food_l = self.font.render("Food :" + str(self.entities[0].food), 1, (255,255,0))
        life_l = self.font.render("Life :" + str(self.entities[0].life), 1, (255,255,0))
        self.window.blit(food_l, (0, 100))
        self.window.blit(life_l, (0, 80))

    #[Graphique] Affichage du décors de la map (Eau, Terre)
    def aff_map(self):
        stp = 27.5
        global Cam
        Cam = vector2d(0, 0)
        pygame.draw.rect(self.window, 0, (0, 0, 1200, 800))
        for y in range(len(self.Map.env.matrix)):
            for x in range(len(self.Map.env.matrix[y])):
                vect = vector2d(Cam.x + 600 - stp * y + x * stp, Cam.y + y * stp / 2 + x * stp / 2)
                if (self.Map.env.matrix[y][x] == 1):
                    self.window.blit(self.grass, (vect.x, vect.y))
                if (self.Map.env.matrix[y][x] == 2):
                    self.window.blit(self.water, (vect.x, vect.y))
                self.aff_entites(y, x, vect)
        self.aff_label()

    def update_entities(self):
        for ent in self.entities:
            self.Map.ent.set_e_i_m(ent.id, ent.pos)

    #[Game Engine] Comptage des entités d'un même type dans la grille d'entité
    def count_entites_id(self, id):
        count = 0
        for ent in self.entities:
            if (ent.id == id):
                count += 1
        return count

    #[Game Engine] Création de nourriture lorsqu'il n'y en a plus assez
    def spawn_food(self):
        while self.count_entites_id(2) < 10:
            pos_x = 0
            pos_y = 0
            while self.Map.env.matrix[pos_x][pos_y] != 2:
                pos_x = randint(0, 19)
                pos_y = randint(0, 19)
            self.entities.append(Entitie(2, pos_y, pos_x))
        while self.count_entites_id(3) < 10:
            pos_x = 0
            pos_y = 0
            while self.Map.env.matrix[pos_x][pos_y] != 2:
                pos_x = randint(0, 19)
                pos_y = randint(0, 19)
            self.entities.append(Entitie(3, pos_y, pos_x))

    #[Game Engine] Fonction vérifiant qu'une nourriture se trouve sur la même case que le joueur
    def update_food(self):
        for ent in self.entities:
            if (ent.id == 2 or ent.id == 3):
                #Si l'élément en cours de vérification est une nourriture (id 2 ou 3) et se trouve sur les mêmes positions que le joueur
                if (ent.pos.x == self.entities[0].pos.x and
                    ent.pos.y == self.entities[0].pos.y):
                    #On augmente sa récompense
                    self.current_reward += 50
                    #On vérifie augmente sa nourriture de 20 et ses points de vie de 1
                    self.entities[0].food = 100 if self.entities[0].food + 20 > 100  else self.entities[0].food + 10
                    self.entities[0].life = 20 if self.entities[0].life + 1 > 20  else self.entities[0].life + 1
                    #On enlève l'entitée
                    self.entities.remove(ent)

    def discount_rewards(self, rewards, fractor, normalization = 0):
        discount = np.zeros_like(rewards)
        G = 0.0
        for i in reversed(range(0, len(rewards))):
            G = G * fractor + rewards[i]
            discount[i] = G
        if (normalization == 1):
            mean = np.mean(discount)
            std = np.std(discount)
            discount = (discount - mean) / (std)
        return discount

    def engine_train(self):
            states, actions, rewards = zip(*self.memory)
            tmp = rewards
            rewards = self.discount_rewards(rewards, 0.97, 1)
            self.memory = list(zip(states, actions, rewards))
            self.model.train(self.memory, self.sess)

    #[Algorythme] Calcul de la récompense ajoutée selon la distance qu'à le joueur avec une nourriture
    def calcul_reward(self):
        best = 0

        for x in range(len(self.entities)):
            if (np.abs(self.entities[x].pos.x - self.entities[0].pos.x) + np.abs(self.entities[x].pos.y - self.entities[0].pos.y) <= 8 and x != 0):
                if (8 - (np.abs(self.entities[x].pos.x - self.entities[0].pos.x) + np.abs(self.entities[x].pos.y - self.entities[0].pos.y)) > best):
                    best = 8 - (np.abs(self.entities[x].pos.x - self.entities[0].pos.x) + np.abs(self.entities[x].pos.y - self.entities[0].pos.y)) 
        self.current_reward += best

    def update_state(self):
        #self.current_state = self.entities[0].get_vision(self.Map.env.matrix, self.Map.ent.matrix)
        self.calcul_reward()

    def delta_state(self):
        self.state_delta = self.entities[0].get_vision(self.Map.env.matrix, self.Map.ent.matrix)
        #exit(0)
        #2 matrix 9 9 2
        """
        self.state_delta = np.zeros(len(self.Map.env.matrix)*10*4)
        for x in range(len(self.Map.env.matrix)*10*2):
            if (i == 19):
                y += 1
                i = 0
            if (y == 20):
                y -= 1
            #print(x)
            self.state_delta[x] = self.Map.env.matrix[y][i]
            self.state_delta[x + 400] = self.Map.ent.matrix[y][i]            
            i += 1
        """

    def AI_mbl(self):
        self.update_state()
        #self.delta_state()
        action_a = 1 if self.current_action == 0 else 0
        action_b = 1 if self.current_action == 1 else 0
        action_c = 1 if self.current_action == 2 else 0
        action_d = 1 if self.current_action == 3 else 0
        #print(np.asarray([self.state_delta]))
        #print([action_a, action_b, action_c, action_d])
        #print(self.current_reward)
        tupples = ([self.state_delta], [action_a, action_b, action_c, action_d], self.current_reward)
        self.memory.append(tupples)

        self.total_reward += self.current_reward
        self.current_reward = 0
        #self.last_state = self.current_state  
            
    #[Game Engine] Vérification de la mort du joueur
    def death(self):
        if (self.Map.env.matrix[self.entities[0].pos.y][self.entities[0].pos.x] == 1):
            self.entities[0].life -= 1
            self.current_reward -= 10
        if (self.entities[0].food <= 0):
            self.entities[0].life -= 1
            self.current_reward -= 10
        else:
            self.entities[0].food -= 1
            self.current_reward -= 1

        #Si le joueur est mort
        if (self.entities[0].life <= 0):
            #Réitialisation
            self.entities[0].life = 20
            self.entities[0].food = 100
            self.entities[0].pos = vector2d(8, 8)
            if (self.best_score < self.total_reward):
                self.best_score = self.total_reward
            self.all_rewards.append(self.total_reward)
            #print("total_reward: " + str(self.total_reward) + " Best score : " + str(self.best_score))
            self.run = 0
            

    def pre_initialization(self):
        self.current_reward = 0
        self.total_reward = 0
        self.train = 1
        self.run = 1
        self.current_action = 0
        self.current_state = self.entities[0].get_vision(self.Map.env.matrix, self.Map.ent.matrix)
        self.last_state = self.entities[0].get_vision(self.Map.env.matrix, self.Map.ent.matrix)


    def _values_ctor(self):
        self.aff = 1
        self.best_score = 0
        #Create session tensorflow
        self.sess = tf.InteractiveSession()
        #initialize value in session
        tf.global_variables_initializer().run()

    def map_update(self):
        self.spawn_food()
        self.Map.ent.matrix = self.Map.generate_map()
        self.update_entities()
        self.update_food()
        #print(np.asarray(self.Map.ent.matrix))
        #print(np.asarray(self.Map.env.matrix))

    def AI_choose_action(self):
        #input matrix initialize
        self.delta_state()
        #print(np.asarray(self.state_delta))
        #print(self.state_delta)

        #get action by sess modal
        self.current_action = self.model.forward([self.state_delta], self.sess)
        #print(self.current_action)
        #print(self.current_action)

    def render(self):
        self.Map.ent.matrix = self.Map.generate_map()
        self.update_entities()
        self._values_ctor()
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, "./model/model_10000.ckpt")

#       l_clock = pygame.time.get_ticks()
        for step in range(self.size_step):
            self.pre_initialization()
            while (self.run == 1):
                self.AI_choose_action()
                self.map_update()
                self.death()
                self.AI_mbl()
                self.action(self.current_action)
                if (self.aff == 1):
                    self.aff_map()
                    pygame.display.flip()
                    pygame.time.delay(100)
            #print("Best score "+ str(self.best_score))
            #self.engine_train()
            """
            if (step % 10 == 0):
                print("train "+ str(step))
                print("Best score "+ str(self.best_score))
                print(np.mean(self.all_rewards))
                print(np.mean(self.model.all_loss))
                self.model.all_loss = []
                self.all_rewards = []
                self.memory = []    
            if (step % 1000 == 0):
                saver = tf.train.Saver(tf.global_variables())
                save_path = saver.save(self.sess, "./model/model_"+str(step)+".ckpt")
            """
    def action(self, action):
        if (self.entities[0].id == 1):
            if (action == 0):
                self.entities[0].move(self.Map.ent, vector2d(1, 0))
            if (action == 1):
                self.entities[0].move(self.Map.ent, vector2d(-1, 0))
            if (action == 2):
                self.entities[0].move(self.Map.ent, vector2d(0, 1))
            if (action == 3):
                self.entities[0].move(self.Map.ent, vector2d(0, -1))

    def run(self):
        pygame.init()
        self.init_graphical()
        self.render()
        pygame.quit()
        """self.current_reward -= 5
        while step < self.size_step:
            state = self.render()
            self.action(1)
            print(state)
            step += 1
        """