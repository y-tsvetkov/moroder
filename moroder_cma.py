# За провеждане на експерименти с оптимизация, този скрипт трябва да бъде активиран.

import cma
import tensorflow as tf
from es import CMAES
import time 
import pickle
from multiprocessing import Pool
import multiprocessing
import funcs 

# Важно: библиотеката за паралелизация изисква целия код да бъде в '__main__'. 
# Това е причината всички вътрешни функции да бъдат поставени в отделен файл.

import matplotlib.pyplot as plt

if __name__ == '__main__' :
  
  # стартиране на групата паралелни работници 
  multiprocessing.set_start_method('spawn', True)
  workers = Pool(funcs.num_workers)

  # фаза на трениране (активиране на Tensorflow graph, активиране на CMA-ES алгоритъм
  with funcs.sess.as_default() :
    funcs.sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    cma = CMAES(funcs.NPARAMS,
    sigma_init=funcs.sigma,
    weight_decay=0,
   popsize=funcs.NPOPULATION) 
  cma_history = funcs.solve(cma,workers)

  # За тест на крайната походка се зареждат параметрите и 
  # максималната награда при поколение (статистически данни)
  with open(funcs.save_dir, 'rb') as f:
    bestparams = pickle.load(f)
  with open(funcs.max_fit_dir, 'rb') as f:
    history = pickle.load(f)

  # тест на походката
  funcs.env[0].reset()
  time.sleep(2)
  fitness = funcs.fitness_func(bestparams,True)

  # показване на графиката най-голяма награда/време
  plt.plot(history)
  plt.show()

  # показване на параметри във функция като С++ масив
  print('{',end = '')
  for param in bestparams:
    print(param, ",", end = '')
  print('}')
  print("Награда при тест:",fitness)
