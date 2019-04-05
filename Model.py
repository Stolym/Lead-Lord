#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

hidden_len=512
#learning_rate=0.000001
learning_rate=1e-9
batch_size=100
gamma=0.99
decay_rate=0.99
value_scale = 1.0
entropy_scale = 0.00
gradient_clip = 40




"""
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3



"""


class Model:
    def __init__(self, map_x, map_y):
        self.map_x = map_x
        self.map_y = map_y
        self.all_loss = []
        self.build_model()
    
    def build_model(self):
        #Création d'une placeholder tensorflow input de taille 9 par 9, par 2
        self.input = tf.placeholder(tf.float32, [None, 9, 9, 2], name="input")
        #self.reshape_layer = tf.reshape(self.input, [-1, 162])
        #Création de placeholder action (transition) permetant l'entraînement
        self.actions = tf.placeholder(tf.float32, [None, 4], name="actions")
        #Placeholder Reward
        #self.m_reward = tf.placeholder(tf.float32, [None, 1], name="reward")
        #discount reward
        self.discount_reward = tf.placeholder(tf.float32, [None, 1], name="reward")

        self.conv_layer_a = tf.layers.conv2d(
            self.input,
            32,
            (2, 2),
            (2, 2),
            "same",
            activation= tf.nn.relu
        )
        #Get layers , filters, Kernel_size, strides, padding, activation
        self.conv_layer_b = tf.layers.conv2d(self.conv_layer_a, 64, (2, 2), (2, 2), "same", activation= tf.nn.relu)
        
        #reshape convolution layer to 1d
        self.flatten_layer = tf.layers.flatten(self.conv_layer_b)
        
        #Normal neuron layer
        self.dense_layer = tf.layers.dense(
            self.flatten_layer,
            units = hidden_len,
            activation = tf.nn.relu, 
            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer = tf.constant_initializer(0.1)
        )
        """
        self.dense_layer_b = tf.layers.dense(
            self.dense_layer,
            units = hidden_len,
            activation = tf.nn.relu,
        )
        """

#            bias_initializer = tf.constant_initializer(0.1),
#            kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),

        self.output_layer = tf.layers.dense(
            self.dense_layer,
            units = 4,
            activation = tf.nn.sigmoid, 
        )

        #select best ouput
        self.softmax = tf.nn.softmax(self.output_layer)
        
        #select best nomial action
        self.calc_action = tf.multinomial(self.output_layer, 1)
        
        """
        self._values = tf.layers.dense(self.dense_layer, 1)
        self.a_prob = tf.nn.softmax(self.output_layer)
        self.action_log = tf.nn.log_softmax(self.output_layer)

        
        self.v_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output_layer, labels=tf.argmax(self.actions, 1))
        self.prog_loss = tf.reduce_mean((self.discount_reward - self._values) * self.v_loss)
        self.values_loss = value_scale * tf.reduce_mean(tf.square(self.discount_reward - self._values))
        self.entropy_loss = -entropy_scale * tf.reduce_sum(self.a_prob * tf.exp(self.a_prob))
        self.loss = self.prog_loss + self.values_loss - self.entropy_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.grads = tf.gradients(self.loss, tf.trainable_variables())
        self.grads, _ = tf.clip_by_global_norm(self.grads, gradient_clip) # gradient clipping
        self.grads_and_vars = list(zip(self.grads, tf.trainable_variables()))
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)
        """
        #self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #    logits=self.output_layer,
        #    labels=self.actions
        #)

        #self.loss = tf.reduce_mean(self.neg_log_prob * self.reward)
        #self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_layer, labels=self.actions)
        #self.loss = tf.reduce_mean(self.neg_log_prob * self.reward)

        #calculation loss
        self.loss = tf.losses.log_loss(labels=self.actions, predictions=self.output_layer, weights=self.discount_reward)
        #train layers
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
    

    def forward(self, stade, sess):
        #stade = stade[:, np.newaxis]
        output = sess.run(self.calc_action, feed_dict={self.input: stade})
        #action = np.random.choice(range(len(output.ravel())), p=output.ravel())
        return (output[0][0])
    
    def train(self, memory, sess):
        states, actions, d_rewards = zip(*memory)
        states = np.vstack(states)
        actions = np.vstack(actions)
        d_rewards = np.vstack(d_rewards)
        train = sess.run(self.train_op, feed_dict={self.input: states, self.actions: actions, self.discount_reward: d_rewards})
        _loss = sess.run(self.loss, feed_dict={self.input: states, self.actions: actions , self.discount_reward: d_rewards})
        self.all_loss.append(_loss)
        #print(_loss)
        #print()
        #print("train ")
        #print(train)