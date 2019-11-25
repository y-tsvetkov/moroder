#Contains the test loop 
import cma
import tensorflow as tf
from es import CMAES
import time 
import pickle
from multiprocessing import Pool
import multiprocessing

import funcs 

# NOTE: The multiprocessing library requires that everything in the script is ran within '__main__'. 
# This is the reason why all functions are exported to a separate file.

import matplotlib.pyplot as plt

if __name__ == '__main__' :
  
  #start the pool
  multiprocessing.set_start_method('spawn', True)
  workers = Pool(funcs.num_workers)

  #training phase
  with funcs.sess.as_default() :
    funcs.sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    cma = CMAES(funcs.NPARAMS,
    sigma_init=funcs.sigma,
    weight_decay=funcs.wd,
   popsize=funcs.NPOPULATION) 
    cma_history = funcs.solve(cma,workers)

  #load best parameters from training
  with open(funcs.save_dir, 'rb') as f:
    bestparams = pickle.load(f)
  with open(funcs.history_dir, 'rb') as f:
    history = pickle.load(f)

  #test the environment with the best parameters
  funcs.env[0].reset()
  time.sleep(2)
  funcs.fitness_func(bestparams, True)

  #plot the training curve and display the parameters
  plt.plot(history)
  plt.show()
  print('{',end = '')
  for param in bestparams:
    print(param, ",", end = '')
  print('}', end = '')