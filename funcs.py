# Съдържа функциите за създаване и трениране на невронна мрежа чрез еволюционни стратегии.

import numpy as np
import tensorflow as tf
import gym
import pybullet as p
import time 
from statistics import stdev
import pickle
from math import sin, cos, pi
import pybullet as p
import pybullet_data
import random

# Директории за параметри и статистически данни
# Съответно: параметри на тренираната мрежа, параметри на експеримента,
# данни за най-голямата награда глобално, най-голямата,
# средната и най-малката награда във всяко поколение
save_dir = 'weights_sine_controller_0.5tifac_8hl_tanh.pkl'
history_dir = 'history_sine_controller_0.5tifac_8hl_tanh.pkl'
max_fit_dir = 'maxfit_sine_controller_0.5tifac_8hl_tanh.pkl'
avg_fit_dir = 'avgfit_sine_controller_0.5tifac_8hl_tanh.pkl'
min_fit_dir = 'minfit_sine_controller_0.5tifac_8hl_tanh.pkl'

# определяне на броя изходни параметри в зависимост от типа експеримент
trot =  False
out_size = 6 if trot else 12


# неврони в скрит слой - препоръчително е те да са между 15 и 30
hlayer_size = 8 

# стандартно отклонение на първото поколение
sigma = 0.2

# опити на поколение (популация), максимален брой поколения
NPOPULATION = 250
MAX_ITERATION = 5000

# брой работници при трениране - препоръчително е този брой да бъде съобразен с броя на ядрата на машината и да не го надвишава
num_workers = 1 

# опити на всеки набор параметри - при наличие на по-висока сложност и произволност е добре те да са поне 2
rollouts_per_param = 2 
 
# при линейното нарастване на дължината на експеримента, наказанието за падане, 
# страничните сили и произволността на ориентацията роботът се учи по-добре
# Това са максималните стойности на всеки един от тези параметри
max_ep_limit = 3100
max_penalty = 100
max_perturb_bounds = [100, 70]
max_orient_bounds = [1,0.7]

# през колко поколения да се дават данни за това как се учи алгоритъма
generations_per_diagnostics = 1

# масиви за симулационни среди
client = []
env = []

# масиви за TensorFlow graphs
in_placeholder = []
w = []
b = []
hid = []
w0 = []
b0 = []
actout = []
w_placeholder = []
w0_placeholder= []
b_placeholder = []
b0_placeholder = []
assignop_w = []
assignop_w0 = []
assignop_b = []
assignop_b0 = []

# създаване на невронна мрежа и симулатори
for i in range(num_workers) :

    # Стартиране на симулатори
    # важно: при трениране типа на симулация трябва да се промени от p.GUI на p.DIRECT за по-голяма скорост
    client.append(p.connect(p.DIRECT))
    p.setPhysicsEngineParameter(enableConeFriction=0, physicsClientId = client[i])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    env.append(gym.make('MoroderEnvNOIT-v0', physics_client=client[i], ep_limit = 1000, 
    torque_factor = 0.0,angle_factor = 0.0, x_factor=0, time_factor = 0.5, 
    penalty_for_falling = 0.01, add_noise = True, perturb_bounds = [0,0], 
    conventional_nn = False, rect_field = False, height_field = True, field_range = 0.01, randomise_orient = True, include_orient = True, 
    randomise_torque = True,rew_factor = 2000, trot = trot, 
    randomise_ang_vel = True, time_only = False,  orient_bounds = [0,0]))
    
    # Създаване на TensorFlow graph:
    # Входен слой
    in_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (env[i].observation_params,None)))

    # Скрит слой
    w.append(tf.Variable(tf.zeros(shape = (hlayer_size,env[i].observation_params),dtype = tf.float64),dtype = tf.float64, shape = (hlayer_size,env[i].observation_params)))
    b.append(tf.Variable(tf.zeros(shape = (hlayer_size,1),dtype = tf.float64),dtype = tf.float64, shape = (hlayer_size,1)))
    hid.append(tf.math.sin(tf.add(tf.matmul(tf.cast(w[i],tf.float64),in_placeholder[i]),b[i])))

    # Изходен слой
    w0.append(tf.Variable(tf.zeros(shape = (out_size,hlayer_size),dtype = tf.float64),dtype = tf.float64, shape = (out_size,hlayer_size)))
    b0.append(tf.Variable(tf.zeros(shape = (out_size,1),dtype = tf.float64),dtype = tf.float64, shape = (out_size,1)))
    actout.append(tf.add(tf.matmul(w0[i],hid[i]),b0[i]))
    
    # Задаване на носители на данни и действия в graph-a
    w_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (hlayer_size,env[i].observation_params)))
    w0_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (out_size,hlayer_size)))
    b_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (hlayer_size,1)))
    b0_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (out_size,1)))
    assignop_w.append(tf.assign(w[i],w_placeholder[i]))
    assignop_w0.append(tf.assign(w0[i],w0_placeholder[i]))
    assignop_b.append(tf.assign(b[i],b_placeholder[i]))
    assignop_b0.append(tf.assign(b0[i],b0_placeholder[i]))

# получаване на брой елементи за оптимизация
w_elements = hlayer_size*env[0].observation_params
w0_elements = hlayer_size*out_size
NPARAMS = hlayer_size+w_elements+w0_elements +out_size
sess = tf.compat.v1.Session()

# Получаване на награда за един комплект параметри
def fitness_func(paramlist, render = False):

  # тъй като тази функция обичайно върви паралелно, по този начин стойностите 
  # автоматично се прехвърлят на симулаторите, които ще ги извършат
  id = random.randrange(num_workers) 

  # TensorFlow graph-ът получава параметрите
  w_newvals = np.reshape(np.round(paramlist[0:w_elements]*100)*0.01,(hlayer_size,env[id].observation_params)) #formerly int32
  b_newvals = np.reshape(paramlist[w_elements:w_elements+hlayer_size],(hlayer_size,1))
  w0_newvals = np.reshape(paramlist[w_elements+hlayer_size:w_elements+hlayer_size+w0_elements],(out_size,hlayer_size)) 
  b0_newvals = np.reshape(paramlist[w_elements+hlayer_size+w0_elements:],(out_size,1)) 
  sess.run(assignop_w[id], feed_dict = {w_placeholder[id]: w_newvals})
  sess.run(assignop_w0[id], feed_dict = {w0_placeholder[id]: w0_newvals})
  sess.run(assignop_b[id], feed_dict = {b_placeholder[id]: b_newvals})
  sess.run(assignop_b0[id], feed_dict = {b0_placeholder[id]: b0_newvals})
  steps = 0

  # масив от симулационни ходове
  rollouts = np.zeros(rollouts_per_param)

  # Този цикъл представлява същинската част - извършване на сим.
  # стъпки до край на сим. хода и изчисляване на наградата накрая.
  # При няколко хода наградата е средноаритметичното на наградите от всички ходове
  for i in range(len(rollouts)):
    obs = env[id].reset()
    obs = np.reshape(obs, (env[id].observation_params,1))
    while not env[id].isDone :
      act = sess.run(actout[id], feed_dict = {in_placeholder[id]: obs})
      act = np.reshape(act, (out_size,))
      tempobs,rollouts[i],_,_ = env[id].step(act)

      # при тестове на тренираните параметри чрез това изчакване симулацията отговаря на реалното време
      # (всяка стъпка отговаря на 1/240-на от секундата) 
      if render:
        time.sleep(1/240)

      obs = np.reshape(tempobs, (env[id].observation_params,1))
  return (sum(rollouts)/len(rollouts))

# тази функция изпълнява тренировъчната част за алгоритъм solver набор от работници worker_pool
def solve(solver,worker_pool):
  history = []
  maxfit = []
  avgfit = []
  minfit = []

   # това време на изчакване е нужно за правилното стартиране на работниците
  time.sleep(2)
  for j in range(MAX_ITERATION):

    # получават се параметрите за това поколение
    solutions = solver.ask()

    t = time.time()
    fitness_list = []

    # изчисляване паралелно на наградата според параметрите
    for solution in solutions:
      fitness_list.append(worker_pool.apply_async(fitness_func, (solution,False,)))
    
    # изпращане на наградите на алгоритъма
    fitness_list = [fit.get(timeout=None) for fit in fitness_list]
    solver.tell(fitness_list)
    
    # получаване на масив с резултати на поколението: 
    # първият елемент е масив с най-успешните елементи, вторият - отговарящата награда
    result = solver.result()

    # добавяне на данни към директориите
    history.append(result[1])
    maxfit.append(max(fitness_list))
    minfit.append(min(fitness_list))
    avgfit.append(sum(fitness_list)/len(fitness_list))

    # Връщане на диагностични данни на всеки generations_per_diagnostics поколения:
    if (j+1) % generations_per_diagnostics == 0:

      # Най-добрите параметри се тестват отново и се сравняват с най-добрите параметри на това поколение:      
      testfitness = fitness_func(result[0],render=False)
      testmaxfitness = fitness_func(solutions[fitness_list.index(max(fitness_list),False)])    
  
      print("Глобална награда на поколение", (j+1), result[1])
      print('макс. награда:', max(fitness_list),'мин. награда:', min(fitness_list),
      'средна награда:', sum(fitness_list)/len(fitness_list))
      print('Стандартно отклонение:', stdev(fitness_list))
      print('Време между отчитания:', time.time() - t)
      print('Макс. време на сим. ход:', env[0].ep_limit,'Наказание за падане:', env[0].penalty_for_falling, 'Макс. стр. сила:', env[0].perturb_high)

      # Увеличаване на симулационните параметри (ще достигнат максималната си стойност след 600 поколения)
      # Поради ключовото влияние на страничните сили, те достигат максималната си стойност след 300 поколения;
      for id in range(num_workers) :
         env[id].ep_limit = np.clip(env[id].ep_limit+7,env[id].ep_limit,max_ep_limit)
         env[id].penalty_for_falling = np.clip(env[id].penalty_for_falling+max_penalty/600,0,max_penalty)
         env[id].perturb_low = np.clip(env[id].perturb_low+max_perturb_bounds[1]/600, 0, max_perturb_bounds[1])
         env[id].perturb_high = np.clip(env[id].perturb_high+max_perturb_bounds[0]/600, 0, max_perturb_bounds[0])
         env[id].orient_low = np.clip(env[id].orient_low+max_orient_bounds[1]/300,0,max_orient_bounds[1])
         env[id].orient_high = np.clip(env[id].orient_high+max_orient_bounds[0]/300,0,max_orient_bounds[0])
      print('Тествана награда за глобални най-добри параметри:',testfitness)
      print('Тествана награда за най-добри параметри на поколение:', testmaxfitness)

      # Всички симулационни данни се запазват 
      with open(save_dir, 'wb') as f: 
        pickle.dump(result[0], f)
      with open(history_dir, 'wb') as f:  
        pickle.dump(history, f)
      with open(max_fit_dir,"wb") as f:
        pickle.dump(maxfit,f)
      with open(min_fit_dir,"wb") as f:
        pickle.dump(minfit,f)
      with open(avg_fit_dir,"wb") as f:
        pickle.dump(avgfit,f)
      print("Моделът е запазен!" , '\n')

  # На края на оптимизацията се извеждат параметрите на тренировката
  print("Намерен локален оптимум:\n", result[0])
  print("Награда в този локален оптимум:", result[1])
  print('Параметри:')
  print('Макс. въртящ момент и ъглова скорост на моторите:',env[0].maxVel,env[0].force_value,'\n',
  'Наказания за отклонение по ъгъл и въртящ момент:', env[0].angle_factor, env[0].torque_factor,'\n',
  'Големина на поп., поколения, нач. стандарт. откл.:',NPOPULATION,MAX_ITERATION,sigma, '\n',
  'Граници на странични сили:',[env[0].perturb_low,env[0].perturb_high], '\n',
   'Макс. дължина на ход:',env[0].ep_limit, '\n',
   'Брой неврони в скрит слой:', hlayer_size, '\n',
  'Резултатите са съхранени в:', history_dir, '\n',
  'Наказание за падане:', env[0].penalty_for_falling,'\n')

  return history
