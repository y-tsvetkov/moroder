#Contains all functions that need to be run for training neural networks using ES.
import numpy as np
import cma
import tensorflow as tf
import gym
import pybullet as p
from es import CMAES
import time 
import statistics
import pickle
from math import sin, cos, pi
import pybullet as p
import pybullet_data
from multiprocessing import Pool
import multiprocessing
import random



#Directories for saving/loading parameters and training statistics
save_dir = '1test.pkl'
reading_dir = '1weights.pkl'
history_dir = '1history.pkl'
max_fit_dir = '1maxfit.pkl'
avg_fit_dir = '1avgfit.pkl'
min_fit_dir = '1minfit.pkl'

#user-defined trotting parameters
sine_variables = [ 0.2415, -0.50953562,  0.46668569,  1.48742849, -0.2355861 ]

hlayer_size = 12 
sigma = 0.20
wd = 0.000 #weight decay is 0, as activating it changes the best fitness tracking - bug in the CMA library

NPOPULATION = 200
MAX_ITERATION = 5000

num_workers = 8
rollouts_per_param = 4 #it is a good practice to test every set of parameters multiple times to ensure that the learned policies are robust
max_ep_limit = 3000
max_penalty = 10

#Simulation instance arrays
client = []
env = []

#Tensorflow graph arrays
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

#this loop creates all simulators and neural networks for running experiments
for i in range(num_workers) :
    client.append(p.connect(p.DIRECT))#or p.GUI for visualisation when using only 1 worker
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    env.append(gym.make('MoroderEnv-v0', physics_client=client[i], ep_limit = 1000, 
    torque_factor = 0.0,angle_factor = 0.0, time_factor = 0.03, 
    penalty_for_falling = 0.01, randomize_vel = False, add_noise = True, perturb_bounds = [0,0], 
    relu = False, heightField = False, randomiseOrient = False, includeOrient = False, randomiseTorque = True,rew_factor = 20000))
    
    #Tensorflow graph creation
    in_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (env[i].observation_params,None)))

    w.append(tf.Variable(tf.zeros(shape = (hlayer_size,env[i].observation_params),dtype = tf.float64),dtype = tf.float64, shape = (hlayer_size,env[i].observation_params)))
    b.append(tf.Variable(tf.zeros(shape = (hlayer_size,1),dtype = tf.float64),dtype = tf.float64, shape = (hlayer_size,1)))
    hid.append(tf.math.sin(tf.add(tf.matmul(tf.cast(w[i],tf.float64),in_placeholder[i]),b[i])))
  
    w0.append(tf.Variable(tf.zeros(shape = (env[i].numJoints,hlayer_size),dtype = tf.float64),dtype = tf.float64, shape = (env[i].numJoints,hlayer_size)))
    b0.append(tf.Variable(tf.zeros(shape = (env[i].numJoints,1),dtype = tf.float64),dtype = tf.float64, shape = (env[i].numJoints,1)))
    actout.append(tf.add(tf.matmul(w0[i],hid[i]),b0[i]))

    w_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (hlayer_size,env[i].observation_params)))
    w0_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (env[i].numJoints,hlayer_size)))
    b_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (hlayer_size,1)))
    b0_placeholder.append(tf.compat.v1.placeholder(dtype = tf.float64, shape = (env[i].numJoints,1)))

    assignop_w.append(tf.assign(w[i],w_placeholder[i]))
    assignop_w0.append(tf.assign(w0[i],w0_placeholder[i]))
    assignop_b.append(tf.assign(b[i],b_placeholder[i]))
    assignop_b0.append(tf.assign(b0[i],b0_placeholder[i]))
    
w_elements = hlayer_size*env[0].observation_params
w0_elements = hlayer_size*env[0].numJoints
NPARAMS = hlayer_size+w_elements+w0_elements +env[0].numJoints
sess = tf.compat.v1.Session()
def fitness_func(paramlist, render = False, include_user_signal = False):
  id = random.randrange(num_workers) #as fitness is evaluated in parallel, this way workers are sent to non-busy threads
  
  #The Tensorflow graph receives the parameters
  w_newvals =(np.reshape(np.round(paramlist[0:w_elements]*10)*0.1,(hlayer_size,env[id].observation_params))).astype(np.float64) #formerly int32
  b_newvals = np.reshape(paramlist[w_elements:w_elements+hlayer_size],(hlayer_size,1))
  w0_newvals =np.reshape(paramlist[w_elements+hlayer_size:w_elements+hlayer_size+w0_elements],(env[id].numJoints,hlayer_size)) 
  b0_newvals =np.reshape(paramlist[w_elements+hlayer_size+w0_elements:],(env[id].numJoints,1)) 

  sess.run(assignop_w[id], feed_dict = {w_placeholder[id]: w_newvals})
  sess.run(assignop_w0[id], feed_dict = {w0_placeholder[id]: w0_newvals})
  sess.run(assignop_b[id], feed_dict = {b_placeholder[id]: b_newvals})
  sess.run(assignop_b0[id], feed_dict = {b0_placeholder[id]: b0_newvals})

  steps = 0
  if include_user_signal : #This variable defines whether or not the user-defined signal should be included
    femur_ampl, femur_phase, tibia_ampl, tibia_phase,tibia_offset= sine_variables
    femur_signal1 = femur_ampl*sin(steps)-femur_phase
    tibia_signal1 = tibia_ampl*cos(steps+tibia_offset)+tibia_phase
    femur_signal2 = femur_ampl*sin(steps+pi/2)-femur_phase
    tibia_signal2 = tibia_ampl*cos(steps+pi/2+tibia_offset)+tibia_phase
  rollouts = np.zeros(rollouts_per_param)
  for i in range(len(rollouts)):
    obs = env[id].reset()
    obs = np.reshape(obs, (env[id].observation_params,1))
    while not env[id].isDone :
      act = sess.run(actout[id], feed_dict = {in_placeholder[id]: obs})
      act = np.reshape(act, (env[id].numJoints,))
      if include_user_signal :
        femur_signal1 = femur_ampl*sin(steps)-femur_ampl
        tibia_signal1 = tibia_ampl*cos(steps+0.5)+tibia_ampl
        femur_signal2 = femur_ampl*sin(steps+pi)-femur_ampl
        tibia_signal2 = tibia_ampl*cos(steps+pi+0.5)+tibia_ampl
        sine_part = [0,femur_signal1,tibia_signal1,0,femur_signal2,tibia_signal2,0,femur_signal2,tibia_signal2,0,femur_signal1,tibia_signal1]      
        act = np.add(act,sine_part)
      tempobs,rollouts[i],_,_ = env[id].step(act)
    
      if render:
        time.sleep(1/240)
        pass

      obs = np.reshape(tempobs, (env[id].observation_params,1))
      steps +=0.02 #These steps are for the user-defined signal, not for the neural network
  return (sum(rollouts)/len(rollouts))

#calculates number of nagative-valued elements in an array
def negnum(arr):
  counter = 0
  for i in arr:
    if i < 0:
      counter += 1
  return counter

def solve(solver,pooled):
  history = []
  maxfit = []
  avgfit = []
  minfit = []
  time.sleep(10)
  for j in range(MAX_ITERATION):

    solutions = solver.ask() #parameters are received from the algorithm for evaluation

    t = time.time()
    fitness_list = []
    for solution in solutions:
      fitness_list.append(pooled.apply_async(fitness_func, (solution,)))
    #print([fit.get(timeout = 1) for fit in temp_fit])
    #for i,fit in zip(range(solver.popsize),temp_fit):
    
    fitness_list = [fit.get(timeout=None) for fit in fitness_list]
    solver.tell(fitness_list)
    result = solver.result() # first element is the best solution, second element is the best fitness
    history.append(result[1])
    maxfit.append(max(fitness_list))
    minfit.append(min(fitness_list))
    avgfit.append(sum(fitness_list)/len(fitness_list))
    if (j+1) % 1 == 0: #Output diagnostics data every generation end
      print("fitness at iteration", (j+1), result[1])
      print('max fitness:', max(fitness_list),'min fitness:', min(fitness_list),
      'mean fitness:', sum(fitness_list)/len(fitness_list))
      print('Standard deviation of fitness:', statistics.stdev(fitness_list),
      'Percentage of negative tries:', 100*negnum(fitness_list)/NPOPULATION)
      print('Loop execution time:', time.time() - t)
      print('Episode limit:', env[0].ep_limit,'Penalty for falling:', env[0].penalty_for_falling)
      for id in range(num_workers) :
         env[id].ep_limit = np.clip(env[id].ep_limit+7,env[id].ep_limit,max_ep_limit)
         env[id].penalty_for_falling = np.clip(env[id].penalty_for_falling+0.01,0,max_penalty)

      #Test the best-performing agent again
      testfitness = fitness_func(result[0],False)
      print('Tested fitness:',testfitness)

      #Save all model data
      print("Model is saved!" , '\n') 
      with open(save_dir, 'wb') as f: 
        pickle.dump(result[0], f)
      with open(history_dir, 'wb') as f:  
        pickle.dump(history, f)
      with open(reading_dir, 'wb') as f: 
        pickle.dump([env[0].maxVel,env[0].force_value,env[0].angle_factor, 
        env[0].torque_factor,NPOPULATION,MAX_ITERATION,sigma,
        [env[0].perturb_low,env[0].perturb_high],env[0].targetVel,
        env[0].ep_limit,hlayer_size], f)
      with open(max_fit_dir,"wb") as f:
        pickle.dump(maxfit,f)
      with open(min_fit_dir,"wb") as f:
        pickle.dump(minfit,f)
      with open(avg_fit_dir,"wb") as f:
        pickle.dump(avgfit,f)
  #At end of optimisation
  print("local optimum discovered by solver:\n", result[0])
  print("fitness score at this local optimum:", result[1])
  print('Params:')
  print('Max torque and velocity:',env[0].maxVel,env[0].force_value,'\n',
  'Angle and torque penalty:', env[0].angle_factor, env[0].torque_factor,'\n',
  'Population size, generations, sigma, weight decay:',NPOPULATION,MAX_ITERATION,sigma, '\n',
  'Perturbations:',[env[0].perturb_low,env[0].perturb_high], '\n',
  'Target velocity (as fast as it can go if None) :',env[0].targetVel, '\n',
   'Episode limit:',env[0].ep_limit, '\n',
   'Architecture of net:', hlayer_size, '\n',
  'Readings are stored in:', reading_dir, '\n',
  'Results list is stored in:', history_dir, '\n',
  'Penalty for falling:', env[0].penalty_for_falling,'\n')

  return history