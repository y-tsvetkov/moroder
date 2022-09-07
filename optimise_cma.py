# Script for training sinusoidal neural networks

import cma
import tensorflow as tf
from es import CMAES
import time 
import pickle
from multiprocessing import Pool
import multiprocessing
import optim_functions as optim

# Important: multiprocessing requires the code to be executed in parallel to be in __main__ 
# this is why it is required to split the code in a main script (this) and a function script (training_functions.py)

import matplotlib.pyplot as plt

if __name__ == '__main__' :
  
  # starting parallel workers 
  multiprocessing.set_start_method('spawn', True)
  workers = Pool(optim.num_workers)

  # training phase (Tensorflow graph activation, performing CMA-ES optimisation)
  with optim.sess.as_default() :
    optim.sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.optim.Saver()
    cma = CMAES(optim.NPARAMS,
    sigma_init=optim.sigma,
    weight_decay=0,
   popsize=optim.NPOPULATION) 
  cma_history = optim.solve(cma,workers)

  # loading learned gait after training
  with open(optim.save_dir, 'rb') as f:
    bestparams = pickle.load(f)
  with open(optim.max_fit_dir, 'rb') as f:
    history = pickle.load(f)

  # testing learned gait
  optim.env[0].reset()
  time.sleep(2)
  fitness = optim.fitness_func(bestparams,True)

  # showing reward/time graph
  plt.plot(history)
  plt.show()

  # showing weights and biases as a C++ array
  print('{',end = '')
  for param in bestparams:
    print(param, ",", end = '')
  print('}')
  print("Testing reward:",fitness)
