#Contains

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import math
import numpy as np
import random

import pybullet as p
from pybullet_envs.bullet import bullet_client
import pybullet_data
from mpmath import sech
import time
RENDER_HEIGHT = 360
RENDER_WIDTH = 480

class MoroderEnv(gym.Env):
  metadata = {'render.modes': ['rgb_array']}

  #initialise simulation, load plane and robot
  def initSim(self, robotVersion, terrain = "plane.urdf", xForwards = False, relu = False):
     self.xForwards = xForwards
     if robotVersion == '/home/mr-mecha/Desktop/Moroder/moroder.urdf' :
       self.joints = [0,1,3,5,6,8,10,11,13,15,16,18] # not all URDF joints are robots - some of them are rigid linkages (only revolute joints are in this matrix)
     else : 
       self.joints = 0 

     self.numJoints = len(self.joints)

     p.setGravity(0,0,-9.807,physicsClientId=self.physicsClient)
     #create uneven plane
     if self.heightField :
       heightPerturbationRange = 0.05
       numHeightfieldRows = 256
       numHeightfieldColumns = 256
       heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns 
       for j in range (int(numHeightfieldColumns/2)):
         for i in range (int(numHeightfieldRows/2) ):

           height = random.uniform(0,heightPerturbationRange)
           heightfieldData[2*i+2*j*numHeightfieldRows]=height
           heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
           heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
           heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height
       terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, 
       meshScale=[.05,.05,1], heightfieldTextureScaling=(numHeightfieldRows-1)/2, 
       heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, 
       numHeightfieldColumns=numHeightfieldColumns,physicsClientId=self.physicsClient)
       self.planeID  = p.createMultiBody(0, terrainShape,physicsClientId=self.physicsClient)

     else:
       self.planeID = p.loadURDF("plane.urdf",physicsClientId=self.physicsClient) #create flat plane
     #load robot with a particular position and orientation
     cubeStartPos = [0,0,0.15]
     if xForwards :
       cubeStartOrientation = p.getQuaternionFromEuler([0,0,-math.pi/2],physicsClientId=self.physicsClient)
     elif self.randomiseOrient == True :
       cubeStartOrientation = p.getQuaternionFromEuler([0,0,random.uniform(0.4361,0.1744)*random.choice(self.perturb_direction)],physicsClientId=self.physicsClient)
     else :
       cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
     self.robotID = p.loadURDF(robotVersion,cubeStartPos, cubeStartOrientation, flags = p.URDF_USE_INERTIA_FROM_FILE,physicsClientId=self.physicsClient)
     if self.randomiseTorque :
       self.force_value = 0.9*random.uniform(0.8,1) #if torque is randomised, it is computed here again

  def calculateReward(self):
    avgLinVel = self.LinVelSum/self.ep_limit
    if self.maxLinVel == 0: #evading the case where the robot stands perfectly still
      self.maxLinVel = 100
    if self.targetVel == None:
      reward = np.sign(avgLinVel)*self.rew_factor*(avgLinVel**2/self.maxLinVel)-self.torqueSum*self.torque_factor-self.angleSum*self.angle_factor
    else:
      reward = self.rew_factor*sech(avgLinVel-self.targetVel)*avgLinVel/self.maxLinVel-self.torqueSum*self.torque_factor-self.angleSum*self.angle_factor
    self.maxLinVel = 0
    return reward


  def __init__(self, physics_client, ep_limit = 1000, randomize_vel = False, targetVel = None, 
  torque_factor = 0, angle_factor = 0, perturb_bounds = [100,50], 
  robotVer = '/home/mr-mecha/Desktop/Moroder/moroder.urdf', time_factor = 1, 
  penalty_for_falling = -100, add_noise = False, relu = False,
  heightField = False, randomiseOrient = True, 
  includeOrient = False, randomiseTorque = True,rew_factor = 2000):
     #camera parameters for rendering
     self._cam_dist = 1.0
     self._cam_pitch = -30
     self._cam_yaw = 0
     self.add_noise = add_noise

     if self.add_noise :
       self.ang_vel_noise = 0.005
       self.rad_noise = 0.05
     else:
       self.rad_noise = 0
       self.ang_vel_noise = 0

     self.robotVer = robotVer #version of the robot to be used
     self.ep_limit = ep_limit
     self.oldLinVel = 0
     self.maxVel = 5.23 #max servo angular speed in rad/s
     self.force_value = 0.9 #max servo torque in N*m
     self.penalty_for_falling = penalty_for_falling
     self.targetVel = targetVel #if != 0, the robot will be trained to achieve that speed
     self.perturb_high, self.perturb_low = perturb_bounds #perturbation limits (if set to 0, no perturbations)
     self.perturb_direction = [1,-1]
     self.randomize_vel = randomize_vel #if true, the robot will be trained to achieve a target speed
     self.torque_factor = torque_factor #how much penalty will be applied to the robot's reward for consumed energy in the form of torque
     self.torqueSum = 0
     self.angleSum = 0
     self.maxLinVel = 0
     self.LinVelSum = 0
     self.relu = relu 
     self.randomiseOrient = randomiseOrient #if true, the robot will be tilted in each test
     self.includeOrient = includeOrient #the robot will receive information about its heading if true
     self.randomiseTorque = randomiseTorque #each rollout the robot's torque will be randomised within a fixed range to make the sim-to-real gap smaller

     if relu: #if relu layers are being used in the NN that calculates the output
       self.observation_params = 4
     else :
       self.observation_params = 5
     if randomize_vel:
       self.observation_params += 1
     if includeOrient:
       self.observation_params += 1

     self.rew_factor = rew_factor
     self.angle_factor = angle_factor #they are roughly the same in importance
     self.time_factor = time_factor
     self.heightField = heightField #if true, the terrain will be randomised in each rollout
     self.physicsClient = physics_client #in parallel simulations, the worker running the simulation must be defined
     self.initSim(robotVersion = self.robotVer) #load the objects in the environment
     self.jointposes = np.zeros((self.numJoints,))
     self.jointTorques = np.zeros((self.numJoints,))  
     self.force_matrix = np.full((self.numJoints,), self.force_value)
     self._steps = 0
     ASlow = np.array([-3.14/4, -3.14/4, 0,-3.14/4, -3.14/4, 0,-3.14/4, -3.14/4, 0,-3.14/4, -3.14/4, 0])
     AShigh = np.array([3.14/4, 3.14/4, 3.14/4,3.14/4, 3.14/4, 3.14/4,3.14/4, 3.14/4, 3.14/4,3.14/4, 3.14/4, 3.14/4,])
     self.observation_space = spaces.Box(low = -5, high = 5, shape = (self.observation_params,), dtype=np.float16)
     self._observation = []
     self.action_space = spaces.Box(low = ASlow, high = AShigh, dtype=np.float16)
     self.isDone = False
     
  # takes an action (defined by a 12-vector with motor positions) and returns the 
  # observation, reward (if the simulation is in the last step of the rollout)
  # and whether or not the rollout has ended
  def step(self, action):
    #apply the action vector to the joints and step the simulation
    [p.setJointMotorControl2(self.robotID,j,p.POSITION_CONTROL, i, 
    force = self.force_value, maxVelocity = self.maxVel,physicsClientId=self.physicsClient) for i,j in zip(action,self.joints)]
    p.stepSimulation(physicsClientId=self.physicsClient)
    self.jointposes,self.jointTorques =  [p.getJointState(self.robotID, self.joints[i],physicsClientId=self.physicsClient)[0] for i in range(self.numJoints)],[p.getJointState(self.robotID, self.joints[i])[3] for i in range(self.numJoints)]
    self._steps = self._steps + 1
    
    #get position, orientation, angular and linear velocity of the robot
    if self.randomize_vel :
      targetVel = random.uniform(0.25,1.25)
    pos, orientQuaternion = p.getBasePositionAndOrientation(self.robotID,physicsClientId=self.physicsClient)
    orientRad = p.getEulerFromQuaternion(orientQuaternion,physicsClientId=self.physicsClient)
    bodyLinVel,bodyAngVel = p.getBaseVelocity(self.robotID,physicsClientId=self.physicsClient)

    if bodyLinVel[1] > self.maxLinVel :
      self.maxLinVel = abs(bodyLinVel[1])
    self.LinVelSum += bodyLinVel[1]
    self.torqueSum +=sum(np.absolute(self.jointTorques))
    self.angleSum += sum(np.absolute(orientRad)) 

    #apply external force every 300 steps in a random direction with a random magnitude
    if self._steps%300 == 0 :
      p.applyExternalForce(self.robotID, -1, 
      [random.randint(self.perturb_low,self.perturb_high)*random.choice(self.perturb_direction),
      random.randint(self.perturb_low*0.1,self.perturb_high*0.1)*random.choice(self.perturb_direction),
      0],[0,random.uniform(-0.1,0.1),0.1],p.LINK_FRAME,physicsClientId=self.physicsClient)

    #create observation vector depending on env parameters
    if self.relu:
      self._observation = np.concatenate((orientRad[0]+ random.uniform(-self.rad_noise,self.rad_noise), 
      orientRad[1]+ random.uniform(-self.rad_noise,self.rad_noise), 
      bodyAngVel[0] + random.uniform(-self.ang_vel_noise, self.ang_vel_noise), 
      bodyAngVel[1] + random.uniform(-self.ang_vel_noise, self.ang_vel_noise) 
      ), axis=None)
    elif self.includeOrient:
      self._observation = np.concatenate((orientRad[0]+ random.uniform(-self.rad_noise,self.rad_noise), 
      orientRad[1]+ random.uniform(-self.rad_noise,self.rad_noise),
      orientRad[2]+ random.uniform(-self.rad_noise,self.rad_noise), 
      bodyAngVel[0] + random.uniform(-self.ang_vel_noise, self.ang_vel_noise), 
      bodyAngVel[1] + random.uniform(-self.ang_vel_noise, self.ang_vel_noise), 
      self._steps*self.time_factor), axis=None)
    else :
      self._observation = np.concatenate((orientRad[0]+ random.uniform(-self.rad_noise,self.rad_noise), 
      orientRad[1]+ random.uniform(-self.rad_noise,self.rad_noise),
      bodyAngVel[0] + random.uniform(-self.ang_vel_noise, self.ang_vel_noise), 
      bodyAngVel[1] + random.uniform(-self.ang_vel_noise, self.ang_vel_noise), 
      self._steps*self.time_factor), axis=None)
    if self.randomize_vel :
      self._observation = np.append(self._observation,targetVel, axis=None)


    reward = 0
    #if the robot tilts too much in any direction, the rollout ends, compute the reward
    if abs(orientRad[0]) >= 0.5 or abs(orientRad[1]) >= 0.5 or abs(orientRad[2]) >= 0.5: 
      if self._steps >= 700 : #if the robot does not manage to stand for at least 700 steps, no reward is given
        reward = self.calculateReward()
        reward -= self.penalty_for_falling
      else :
        reward = -self.penalty_for_falling
      self.isDone = True

    if self._steps >= self.ep_limit:
      self.isDone = True
      reward = self.calculateReward()

    return self._observation, reward, self.isDone, {}


  #resets the simulation and all parameters kept track of
  def reset(self):
    p.resetSimulation(physicsClientId=self.physicsClient)
    self.initSim(robotVersion = self.robotVer)
    self.isDone = False
    obs = np.zeros((self.observation_params,))
    self._steps = 0
    self._observation = obs
    self.oldLinVel = 0
    self.maxLinVel = 0
    self.LinVelSum = 0
    self.torqueSum = 0 
    self.angleSum = 0
    return np.array(self._observation)

  #renders the environment using an RGB array computed from a view matrix
  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])

      #the view matrix will track the robot so it receives its position
    base_pos, _ = p.getBasePositionAndOrientation(self.robotID,physicsClientId=self.physicsClient)

   #computing the RGBA matrix
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2,physicsClientId=self.physicsClient)
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
        nearVal=0.1, farVal=100.0,physicsClientId=self.physicsClient)
    (_, _, px, _, _) = p.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix, renderer=p.ER_TINY_RENDERER,physicsClientId=self.physicsClient)

    #converting to RGB matrix
    '''rgb_array1 = np.array(px)
    rgb_array2 = np.reshape(rgb_array1,(360,480,4)
    rgb_array = rgb_array2[:, :, :3]'''
    return rgb_array

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

   
