# Script for training sinusoidal neural networks

import cma
import tensorflow as tf
from es import CMAES
import time 
import pickle
from multiprocessing import Pool
import multiprocessing
import functions

# Important: multiprocessing requires the code to be executed in parallel to be in __main__ 
# this is why it is required to split the code in a main script (this) and a function script (cma_functions.py)

import matplotlib.pyplot as plt

if __name__ == '__main__' :
  
  # starting parallel workers 
  multiprocessing.set_start_method('spawn', True)
  workers = Pool(funcs.num_workers)

  # training phase (Tensorflow graph activation, performing CMA-ES optimisation)
  with funcs.sess.as_default() :
    funcs.sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    cma = CMAES(funcs.NPARAMS,
    sigma_init=funcs.sigma,
    weight_decay=0,
   popsize=funcs.NPOPULATION) 
  cma_history = funcs.solve(cma,workers)

  # loading learned gait after training
  with open(funcs.save_dir, 'rb') as f:
    bestparams = pickle.load(f)
  with open(funcs.max_fit_dir, 'rb') as f:
    history = pickle.load(f)

  # testing learned gait
  funcs.env[0].reset()
  time.sleep(2)
  fitness = funcs.fitness_func(bestparams,True)

  # showing reward/time graph
  plt.plot(history)
  plt.show()

  # showing weights and biases as a C++ array
  print('{',end = '')
  for param in bestparams:
    print(param, ",", end = '')
  print('}')
  print("Testing reward:",fitness)
