# Contains required functions for training sinusoidal neural networks using CMA-ES.

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

# directories for saving parameters and diagnostical data (fitness history, maximum, minimum and average fitness)
save_dir = 'weights_biases.pkl'
history_dir = 'history.pkl'
max_fit_dir = 'maxfit.pkl'
avg_fit_dir = 'avgfit.pkl'
min_fit_dir = 'minfit.pkl'

# setting experiment parameters
trot =  False
out_size = 6 if trot else 12
hlayer_size = 8  
sigma = 0.2
NPOPULATION = 250
MAX_ITERATION = 5000
num_workers = 1 # number of parallel processes (workers) - 1 for GUI training or safer non-GUI training, number of CPU cores for fastest training
rollouts_per_param = 2 # tests per single parameter set of population - bigger than 2 recommended for tests with randomisation
 
# linearly increasing of particular parameters helps learning stability and robustness
max_ep_limit = 3100 # episode length in timesteps
max_penalty = 100 # penalty per falling
max_perturb_bounds = [100, 70] # perturbation force in newtons
max_orient_bounds = [1,0.7] # orientation randomisation range (deviation, whether it's left or right is randomised)

# per how many generations to save diagnostical data
generations_per_diagnostics = 1

# simulation environment arrays
client = []
env = []

# TensorFlow graph arrays
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

# creating simulations and TensorFlow graphs
for i in range(num_workers) :

    # starting simulators
    client.append(p.connect(p.DIRECT))
    p.setPhysicsEngineParameter(enableConeFriction=0, physicsClientId = client[i])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    env.append(gym.make(
      'MoroderEnv-v0', 
      physics_client=client[i], 
      ep_limit = 1000, 
      torque_factor = 0.0,
      angle_factor = 0.0, 
      x_factor=0, 
      time_factor = 0.5, 
      penalty_for_falling = 0.01, 
      add_noise = True, 
      perturb_bounds = [0,0], 
      conventional_nn = False, 
      rect_field = False, 
      height_field = True, 
      field_range = 0.01, 
      randomise_orient = True, 
      include_orient = True, 
      randomise_torque = True,
      rew_factor = 2000, 
      trot = trot, 
      randomise_ang_vel = True, 
      time_only = False,  
      orient_bounds = [0,0]
    ))
    
    # creating TensorFlow graph:
    # input layer
    in_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (env[i].observation_params,None)))

    # hidden layer
    w.append(tf.Variable(tf.zeros(shape = (hlayer_size,env[i].observation_params),dtype = tf.float64),dtype = tf.float64, shape = (hlayer_size,env[i].observation_params)))
    b.append(tf.Variable(tf.zeros(shape = (hlayer_size,1),dtype = tf.float64),dtype = tf.float64, shape = (hlayer_size,1)))
    hid.append(tf.math.sin(tf.add(tf.matmul(tf.cast(w[i],tf.float64),in_placeholder[i]),b[i])))

    # output layer
    w0.append(tf.Variable(tf.zeros(shape = (out_size,hlayer_size),dtype = tf.float64),dtype = tf.float64, shape = (out_size,hlayer_size)))
    b0.append(tf.Variable(tf.zeros(shape = (out_size,1),dtype = tf.float64),dtype = tf.float64, shape = (out_size,1)))
    actout.append(tf.add(tf.matmul(w0[i],hid[i]),b0[i]))
    
    # setting weight and bias placeholders which hold the values
    w_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (hlayer_size,env[i].observation_params)))
    w0_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (out_size,hlayer_size)))
    b_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (hlayer_size,1)))
    b0_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (out_size,1)))
    assignop_w.append(tf.assign(w[i],w_placeholder[i]))
    assignop_w0.append(tf.assign(w0[i],w0_placeholder[i]))
    assignop_b.append(tf.assign(b[i],b_placeholder[i]))
    assignop_b0.append(tf.assign(b0[i],b0_placeholder[i]))

# getting number of optimisation parameters
w_elements = hlayer_size * env[0].observation_params
w0_elements = hlayer_size * out_size
NPARAMS = hlayer_size + w_elements + w0_elements + out_size
sess = tf.compat.v1.Session()

# calculating fitness for a single parameter set of a population of generation
def fitness_func(paramlist, render = False):

  # since this function runs in parallel, this is required to assign free simulators to workers.
  # if two workers are assigned the same ID, the second one waits for the first one to be done
  id = random.randrange(num_workers) 

  # receiving parameters
  w_newvals = np.reshape(np.round(paramlist[0:w_elements] * 100) * 0.01, (hlayer_size, env[id].observation_params)) #formerly int32
  b_newvals = np.reshape(paramlist[w_elements:w_elements + hlayer_size], (hlayer_size, 1))
  w0_newvals = np.reshape(paramlist[w_elements+hlayer_size:w_elements + hlayer_size + w0_elements], (out_size,hlayer_size)) 
  b0_newvals = np.reshape(paramlist[w_elements+hlayer_size + w0_elements:], (out_size, 1)) 
  sess.run(assignop_w[id], feed_dict = {w_placeholder[id]: w_newvals})
  sess.run(assignop_w0[id], feed_dict = {w0_placeholder[id]: w0_newvals})
  sess.run(assignop_b[id], feed_dict = {b_placeholder[id]: b_newvals})
  sess.run(assignop_b0[id], feed_dict = {b0_placeholder[id]: b0_newvals})
  steps = 0

  rollouts = np.zeros(rollouts_per_param)

  # this cycle represents the core training loop -  running rollouts_per_param rollouts until they are done.
  # afterwards, the individual fitness values are averaged
  for i in range(len(rollouts)):
    obs = env[id].reset()
    obs = np.reshape(obs, (env[id].observation_params,1))
    while not env[id].isDone :
      act = sess.run(actout[id], feed_dict = {in_placeholder[id]: obs})
      act = np.reshape(act, (out_size,))
      tempobs,rollouts[i], _, _ = env[id].step(act)

      # applying a delay on each timestep when testing to make simulation run in real time
      if render:
        time.sleep(1 / 240) #PyBullet timestep

      obs = np.reshape(tempobs, (env[id].observation_params, 1))
  return (sum(rollouts) / len(rollouts))

# optimising a population according to a solver (in this case, CMA-ES)
def solve(solver,worker_pool):
  history = []
  maxfit = []
  avgfit = []
  minfit = []

   # sleep so workers can safely start (required by multiprocessing)
  time.sleep(2)
  for j in range(MAX_ITERATION):

    # receiving parameters for a generations
    solutions = solver.ask()

    t = time.time()
    fitness_list = []

    # evaluating fitness in parallel for each solution
    for solution in solutions:
      fitness_list.append(worker_pool.apply_async(fitness_func, (solution, False, )))
    
    fitness_list = [fit.get(timeout=None) for fit in fitness_list]
    solver.tell(fitness_list)
    
    # receiving generation result: 
    # first element is best parameters, second element is best fitness for these parameters
    result = solver.result()

    # adding data to directories
    history.append(result[1])
    maxfit.append(max(fitness_list))
    minfit.append(min(fitness_list))
    avgfit.append(sum(fitness_list)/len(fitness_list))
    
    if (j+1) % generations_per_diagnostics == 0:

      # testing of best parameters to verify that simulation is producing actually viable neural nets:      
      testfitness = fitness_func(result[0], render=False)
      testmaxfitness = fitness_func(solutions[fitness_list.index(max(fitness_list), False)])    
  
      print("Best reward so far", (j + 1), result[1])
      print('Best reward this generation:', max(fitness_list),'Min reward:', min(fitness_list),
      'Avg reward:', sum(fitness_list)/len(fitness_list))
      print('St. dev.:', stdev(fitness_list))
      print('Time taken:', time.time() - t)
      print('Current episode limit:', env[0].ep_limit,'Fall penalty:', env[0].penalty_for_falling, 'Max perturbation:', env[0].perturb_high)

      # increasing difficulty of simulation at each generations_per_diagnostics generations:
      for id in range(num_workers) :
         env[id].ep_limit = np.clip(env[id].ep_limit + 7, env[id].ep_limit,max_ep_limit)
         env[id].penalty_for_falling = np.clip(env[id].penalty_for_falling+max_penalty/600, 0, max_penalty)
         env[id].perturb_low = np.clip(env[id].perturb_low + max_perturb_bounds[1]/600, 0, max_perturb_bounds[1])
         env[id].perturb_high = np.clip(env[id].perturb_high + max_perturb_bounds[0]/600, 0, max_perturb_bounds[0])
         env[id].orient_low = np.clip(env[id].orient_low + max_orient_bounds[1]/300, 0, max_orient_bounds[1])
         env[id].orient_high = np.clip(env[id].orient_high + max_orient_bounds[0]/300, 0, max_orient_bounds[0])
      print('Reward for best parameters so far:',testfitness)
      print('Reward for best parameters this gen:', testmaxfitness)

      # saving simulation data 
      with open(save_dir, 'wb') as f: 
        pickle.dump(result[0], f)
      with open(history_dir, 'wb') as f:  
        pickle.dump(history, f)
      with open(max_fit_dir,"wb") as f:
        pickle.dump(maxfit, f)
      with open(min_fit_dir,"wb") as f:
        pickle.dump(minfit, f)
      with open(avg_fit_dir,"wb") as f:
        pickle.dump(avgfit, f)
      print("Model saved!" , '\n')
      
  print("Local optimum:\n", result[0])
  print("Reward for local optimum:", result[1])

  return history
