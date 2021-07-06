# This file contains the simulation environment adapted for the OpenAI Gym framework.
# It must be placed in the 'env' folder at the installation directory of the 'gym' package
# The simulation must be registered in the 'gym' package by adding the following code 
# to '__init__.py' located in 'envs' inside the main directory of 'gym': 

'''
register(
    id='MoroderEnv-v0',
    entry_point='gym.envs.my_collection.moroder_env:MoroderEnv',
    )
'''
import gym
from gym import spaces, utils
from gym.utils import seeding
import math
import numpy as np
import random
import pybullet as p
from pybullet_envs.bullet import bullet_client
import pybullet_data
import time

class MoroderEnv(gym.Env):

  # generating a field of rough terrain consisting of rectangles (parallelepipeds) of arbitrary heights.
  # The field has a width of field_width and has num_width_rects rectangles widthwise,
  # num_length_rects rectangles lengthwise, each of which has a length of rect_length, and the whole field
  # is start_offset meters away from the robot. 
  def generate_rect_field(self,num_width_rects,num_length_rects,field_width,rect_length,start_offset):
    center_width = field_width - field_width/num_width_rects    
    for i in range(num_length_rects):
      for j in range(num_width_rects):
        height = random.uniform(0.005,self.field_range)        
        box_x = -center_width/2+j*center_width/(num_width_rects-1)
        box_y = start_offset+rect_length/2+i*rect_length
        box_z = height/2
        id = p.createCollisionShape(shapeType = p.GEOM_BOX, halfExtents = [field_width/(2*num_width_rects),rect_length/2,height/2])
        
        # Zero-mass objects are static and unmovable by the robot
        mass = 0 
        p.createMultiBody(mass,id,basePosition = [box_x,box_y,box_z])
      
  # generating a height field - a field of vertices with a given height.
  # Since the walls between vertices are slanted, this field is easier to 
  # traverse than the rectangle field.
  def generate_height_field(self):
       heightPerturbationRange = self.field_range
       height_field_rows = 128
       height_field_cols = 128
       height_field_data = [0]*height_field_rows*height_field_cols 

       # generating height field data
       for col in range (int(height_field_cols/2)):
         for row in range (int(height_field_rows/2) ):
           height = random.uniform(0,heightPerturbationRange)
           height_field_data[2*row+2*col*height_field_rows]=height
           height_field_data[2*row+1+2*col*height_field_rows]=height
           height_field_data[2*row+(2*col+1)*height_field_rows]=height
           height_field_data[2*row+1+(2*col+1)*height_field_rows]=height

       # creating the shape in the simulation
       terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, 
       meshScale=[.05,.05,1], heightfieldTextureScaling=(height_field_rows-1)/2, 
       heightfieldData=height_field_data, numHeightfieldRows=height_field_rows, 
       numHeightfieldColumns=height_field_cols,physicsClientId=self.physicsClient)
       self.planeID  = p.createMultiBody(0, terrainShape,physicsClientId=self.physicsClient)

       # setting coefficient of friction to a realistic value for floors
       p.changeDynamics(self.planeID, -1, lateralFriction=0.5,physicsClientId=self.physicsClient)
  
  # simulation initialisation method that runs every time the simulation is rebuilt
  def initSim(self, robotVersion, terrain = "plane.urdf", conventional_nn = False):

    # this array holds the IDs of connections that are joints
     self.joints = [0,1,3,5,6,8,10,11,13,15,16,18] 

     self.numJoints = len(self.joints)
     p.setGravity(0,0,-9.81,physicsClientId=self.physicsClient)

     # creating floor:
     if self.height_field :
       self.generate_height_field() # height field
     elif self.rect_field:
       self.generate_rect_field(4,14,self.field_width,0.2,0.1) # rectangle field
       self.planeID = p.loadURDF("plane.urdf",physicsClientId=self.physicsClient) # flat plane below each field
     else:
       self.planeID = p.loadURDF("plane.urdf",physicsClientId=self.physicsClient) # flat plane

     # initialising robot for given position and orientation (latter is randomised during course correction training):
     robot_init_pos = [0,0,0.15] # позиция
     if self.randomise_orient == True :
       robot_init_orient = p.getQuaternionFromEuler([0,0,random.uniform(1,0.1744)*random.choice(self.perturb_direction)],
       physicsClientId=self.physicsClient)
     else :

       # forwards
       robot_init_orient = p.getQuaternionFromEuler([0,0,0]) 

     # loading robot
     self.robotID = p.loadURDF(robotVersion,robot_init_pos, robot_init_orient, 
     flags = p.URDF_USE_INERTIA_FROM_FILE,physicsClientId=self.physicsClient)

     # defining motor properties
     if self.randomise_torque :
       self.force_value = self.orig_force_value*random.uniform(0.9,1)
     if self.randomise_ang_vel :
       self.maxVel = self.orig_max_vel*random.uniform(0.9,1)

  # Reward calculation in accordance to the definition in the paper:
  def calculateReward(self):

    # calculating average speed over episode length
    avgLinVel = self.LinVelSum/self.ep_limit 

    # terms for deviation to the side on the X axis (x_factor), deviation in angles (angle_factor),
    # used torque over the course of the experiment (torque factor) have been experimented with to determine
    # their usability
    reward = np.sign(avgLinVel)*self.rew_factor*abs(avgLinVel**2/self.maxLinVel)
    -self.torqueSum*self.torque_factor-self.angleSum*self.angle_factor-self.x_sum*self.x_factor
    return reward


  def __init__(self, physics_client, ep_limit = 1000, 
  torque_factor = 0, angle_factor = 0, x_factor=0, perturb_bounds = [100,50], 
  robotVer = 'moroder.urdf', time_factor = 1, 
  penalty_for_falling = -100, add_noise = False, conventional_nn = False,
  height_field = False, rect_field = False, field_range = 0.02, randomise_orient = True, 
  include_orient = False, randomise_torque = True,rew_factor = 2000, trot = False,
  round_action = True, randomise_ang_vel = True,time_only = False, orient_bounds = [1,0.5]):
     
     # adding noise to virtual IMU
     self.add_noise = add_noise 
     if self.add_noise :
       self.ang_vel_noise = 0.005
       self.rad_noise = 0.05
     else:
       self.rad_noise = 0
       self.ang_vel_noise = 0

     # robot version to be used
     self.robotVer = robotVer 

     # maximum length of a single rollout (episode)
     self.ep_limit = ep_limit 

     # motor parameters (orig parameters hold the original value that is modified if randomise_torque/maxVel)
     self.orig_max_vel = 5.23 
     self.orig_force_value = 0.9
     self.maxVel = self.orig_max_vel
     self.force_value = self.orig_force_value

     # getting perturbation ranges in a non-order-dependent way
     self.perturb_high = np.amax(perturb_bounds) 
     self.perturb_low = np.amin(perturb_bounds)
     self.perturb_direction = [1,-1]

     # if robot falls before episode end, penalty_for_falling is substracted from reward
     self.penalty_for_falling = penalty_for_falling

     # factors defined in calculateReward
     self.torque_factor = torque_factor 
     self.x_factor = x_factor
     self.angle_factor = angle_factor 
     self.rew_factor = rew_factor
       
     # rough terrain parameters
     self.height_field = height_field 
     self.rect_field = rect_field
     if height_field and rect_field:
       raise Exception("Two different uneven fields can't be created.")
     self.field_range = field_range
     self.field_width = 0.4
 
     # variables for holding sums
     self.torqueSum = 0
     self.angleSum = 0
     self.x_sum = 0
     self.LinVelSum = 0
     self.maxLinVel = 0

     # whether or not to include robot heading as input ( for course correction training)
     self.include_orient = include_orient 
     
     # defining what to randomise
     self.randomise_orient = randomise_orient
     self.randomise_torque = randomise_torque 
     self.randomise_ang_vel = randomise_ang_vel
     if randomise_orient:
       self.orient_high = np.amax(orient_bounds) 
       self.orient_low = np.amin(orient_bounds) 

     # if trot = True legs along same diagonals will receive the same commands, creating a trotting pattern
     self.trot = trot

     # rounding down the accuracy of the action (output of the network) 
     # to mimic the deadband width of servo motors
     self.round_action = round_action
     self.servo_resolution = 1.5*math.pi/180

     # if time_only = True, the only input is time (for simple periodic signal training)
     self.time_only = time_only 

     # if NN uses ReLU, don't add time to inputs
     self.conventional_nn = conventional_nn 
     if conventional_nn:
       self.observation_params = 4
     else :
       self.observation_params = 5   
     if include_orient:
       self.observation_params += 1

     if time_only:
       if include_orient or conventional_nn :
         raise Exception("Mismatch between desired inputs and number of inputs possible by the chosen architecture. Please check initialisation parameters.")
    
     self.time_factor = time_factor

     # if parallel training, simulation client ID must be specified
     self.physicsClient = physics_client 

     self._steps = 0
     
     # creating observation and action space (required by OpenAI Gym)
     self.observation_space = spaces.Box(low = -5, high = 5, shape = (self.observation_params,), dtype=np.float16)
     self._observation = []
     self.action_space_low = np.array([-3.14/3, -3.14/3, 0,-3.14/3, -3.14/3, 0,-3.14/3, -3.14/3, 0,-3.14/3, -3.14/3, 0])
     self.action_space_high = np.array([3.14/3, 3.14/3, 3.14/2,3.14/3, 3.14/3, 3.14/2,3.14/3, 3.14/3, 3.14/2,3.14/3, 3.14/3, 3.14/2])
     self.action_space = spaces.Box(low = self.action_space_low, high = self.action_space_high, dtype=np.float16)

     # holds data for whether or not the simulation has ended
     self.isDone = False

     self.initSim(robotVersion = self.robotVer) 
     self.jointposes = np.zeros((self.numJoints,))
     self.jointTorques = np.zeros((self.numJoints,))  
     
  # receives the output of the NN, steps the simulation forwards, returns next inputs (observation) and whether or not the episode has ended
  def step(self, nn_outputs):

    # reshaping NN output if trotting is desired
    if self.trot:
      if len(nn_outputs)*2 != self.numJoints:
        raise Exception("The NN outputs are more than half the number of leg motors (trotting is enabled). Please check simulation parameters.")
      action = np.concatenate((nn_outputs[:3],nn_outputs[3:],nn_outputs[3:],nn_outputs[:3]))
    else:
      action = nn_outputs

    # constraining NN output to action space
    action = np.clip(action, self.action_space_low,self.action_space_high)

    if self.round_action:
     action = [np.round(cmd/self.servo_resolution)*self.servo_resolution for cmd in action]

    # driving servo motors with a generator loop to reduce compute time
    [p.setJointMotorControl2(self.robotID,j,p.POSITION_CONTROL, i, 
    force = self.force_value, maxVelocity = self.maxVel,physicsClientId=self.physicsClient) for i,j in zip(action,self.joints)]

    # stepping simulation
    p.stepSimulation(physicsClientId=self.physicsClient)

    # receiving new positions and torques, position, orientation, linear and angular velocity
    self.jointposes,self.jointTorques =  [p.getJointState(self.robotID, self.joints[i],physicsClientId=self.physicsClient)[0] for i in range(self.numJoints)],[p.getJointState(self.robotID, 
    self.joints[i],physicsClientId=self.physicsClient)[3] for i in range(self.numJoints)]
    self._steps = self._steps + 1
    pos, orientQuaternion = p.getBasePositionAndOrientation(self.robotID,physicsClientId=self.physicsClient)
    orientRad = p.getEulerFromQuaternion(orientQuaternion,physicsClientId=self.physicsClient)
    bodyLinVel,bodyAngVel = p.getBaseVelocity(self.robotID,physicsClientId=self.physicsClient)

    if bodyLinVel[1] > self.maxLinVel :
      self.maxLinVel = abs(bodyLinVel[1])
    self.LinVelSum += bodyLinVel[1]
    self.torqueSum +=sum(np.absolute(self.jointTorques))
    self.angleSum += sum(np.absolute(orientRad)) 
    self.x_sum += np.absolute(pos[0])

    # applying perturbation
    if self._steps%300 == 0 and self._steps >=300 :
      p.applyExternalForce(self.robotID, -1, 
      [random.uniform(self.perturb_low,self.perturb_high)*random.choice(self.perturb_direction),
      random.uniform(self.perturb_low*0.1,self.perturb_high*0.1)*random.choice(self.perturb_direction),
      0],[0,random.uniform(-0.1,0.1),0.1],p.LINK_FRAME,physicsClientId=self.physicsClient)

    # creating new NN inputs
    self._observation = np.concatenate((orientRad[0]+ random.uniform(-self.rad_noise,self.rad_noise), 
    orientRad[1]+ random.uniform(-self.rad_noise,self.rad_noise),
    bodyAngVel[0] + random.uniform(-self.ang_vel_noise, self.ang_vel_noise), 
    bodyAngVel[1] + random.uniform(-self.ang_vel_noise, self.ang_vel_noise), 
    self._steps*self.time_factor), axis=None)
    if self.time_only :
      self._observation = self._steps*self.time_factor
    if self.conventional_nn:
      self._observation = np.concatenate((orientRad[0]+ random.uniform(-self.rad_noise,self.rad_noise), 
      orientRad[1]+ random.uniform(-self.rad_noise,self.rad_noise), 
      bodyAngVel[0] + random.uniform(-self.ang_vel_noise, self.ang_vel_noise), 
      bodyAngVel[1] + random.uniform(-self.ang_vel_noise, self.ang_vel_noise) 
      ), axis=None)
    if self.include_orient:
      self._observation = np.insert(self._observation,2,orientRad[2]+ random.uniform(-self.rad_noise,self.rad_noise),axis = None)
    reward = 0

    # returning reward if this step is last
    if self._steps >= self.ep_limit:
      self.isDone = True
      reward = self.calculateReward()

    # if the robot deviates in orientation or distance from the forwards axis, apply penalty and return reward
    if abs(orientRad[0]) >= 0.5 or abs(orientRad[1]) >= 0.5 or abs(orientRad[2]) >= 1.57 or abs(pos[0])*self.randomise_orient >= self.field_width/2 :
      if self._steps >= 700 : # ако роботът не може да остане прав за минимум 700 стъпки, той не получава награда
        reward = self.calculateReward()
        reward -= self.penalty_for_falling
      else :
        reward = -self.penalty_for_falling
      self.isDone = True

    return self._observation, reward, self.isDone, {}

  # restarting simulation
  def reset(self):
    p.resetSimulation(physicsClientId=self.physicsClient)
    self.initSim(robotVersion = self.robotVer)
    self.isDone = False
    obs = np.zeros((self.observation_params,))
    self._steps = 0
    self._observation = obs
    self.maxLinVel = 0
    self.LinVelSum = 0
    self.torqueSum = 0 
    self.angleSum = 0
    self.x_sum = 0
    return np.array(self._observation)

  # random seed generation (required by OpenAI Gym)
  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

   
