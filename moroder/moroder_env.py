# Този файл съдържа симулацията, адаптирана за framework OpenAI Gym.
# Той трябва да бъде поставен в папка 'env' в мястото на инсталация на пакета 'gym'
# Също така симулацията трябва да бъде регистрирана, като се добави следния код към '__init__.py', намиращ се в 'envs':

'''
register(
    id='MoroderEnvNOIT-v0',
    entry_point='gym.envs.morodernoit:MoroderEnvNOIT',
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

class MoroderEnvNOIT(gym.Env):

  # Генериране на поле от неравности, състоящо се от паралелепипеди с произволни височини.
  # Полето има ширина field_width и има num_width_rects на брой паралелепипеди по ширина,
  # num_length_rects по дължина, всеки от които има дължина rect_length, като цялото поле 
  # е отдалечено на start_offset метри от робота.
  def generate_rect_field(self,num_width_rects,num_length_rects,field_width,rect_length,start_offset):
    center_width = field_width - field_width/num_width_rects    
    for i in range(num_length_rects):
      for j in range(num_width_rects):
        height = random.uniform(0.005,self.field_range)        
        box_x = -center_width/2+j*center_width/(num_width_rects-1)
        box_y = start_offset+rect_length/2+i*rect_length
        box_z = height/2
        id = p.createCollisionShape(shapeType = p.GEOM_BOX, halfExtents = [field_width/(2*num_width_rects),rect_length/2,height/2])
        
        # Обекти с маса 0 са статични според симулатора и не могат да бъдат преместени от робота
        mass = 0 
        p.createMultiBody(mass,id,basePosition = [box_x,box_y,box_z])
      
  # Генериране на височинно поле - това е поле от неравности, състоящо се от един обект, 
  # разделен на квадрати, чиято височина е произволно избрана. В сравнение с полето от 
  # паралелепипеди, това поле е по-лесно за преодоляване, защото между два квадрата с
  # с различни височини се образува наклонен квадрат, т.е. няма стъпала в полето
  def generate_height_field(self):
       heightPerturbationRange = self.field_range
       height_field_rows = 128
       height_field_cols = 128
       height_field_data = [0]*height_field_rows*height_field_cols 

       # генериране на данните за височинното поле
       for col in range (int(height_field_cols/2)):
         for row in range (int(height_field_rows/2) ):
           height = random.uniform(0,heightPerturbationRange)
           height_field_data[2*row+2*col*height_field_rows]=height
           height_field_data[2*row+1+2*col*height_field_rows]=height
           height_field_data[2*row+(2*col+1)*height_field_rows]=height
           height_field_data[2*row+1+(2*col+1)*height_field_rows]=height

       # създаване на формата в симулатора
       terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, 
       meshScale=[.05,.05,1], heightfieldTextureScaling=(height_field_rows-1)/2, 
       heightfieldData=height_field_data, numHeightfieldRows=height_field_rows, 
       numHeightfieldColumns=height_field_cols,physicsClientId=self.physicsClient)
       self.planeID  = p.createMultiBody(0, terrainShape,physicsClientId=self.physicsClient)

       # задаване на коефициента на триене, за да няма приплъзване
       p.changeDynamics(self.planeID, -1, lateralFriction=0.5,physicsClientId=self.physicsClient)
  
  # инициализиране на симулацията (рестартиране на симулатор, поставяне на робот, т.н.)
  def initSim(self, robotVersion, terrain = "plane.urdf", conventional_nn = False):

    # задаване на матрицата със стави, съответстващи на индексите (не всички стави са подвижни)
     self.joints = [0,1,3,5,6,8,10,11,13,15,16,18] 

     self.numJoints = len(self.joints)
     p.setGravity(0,0,-9.81,physicsClientId=self.physicsClient)

     # създаване на земята на робота, за които има няколко варианта:
     if self.height_field :
       self.generate_height_field() # височинно поле
     elif self.rect_field:
       self.generate_rect_field(4,14,self.field_width,0.2,0.1) # поле от паралелепипеди
       self.planeID = p.loadURDF("plane.urdf",physicsClientId=self.physicsClient) # плоска част на полето
     else:
       self.planeID = p.loadURDF("plane.urdf",physicsClientId=self.physicsClient) # плоска земя

     # зареждане на робота при дадени позиция и завъртане:
     robot_init_pos = [0,0,0.15] # позиция

     # добавяне на произволност на ориентацията на робота, когато той се тренира за завиване
     if self.randomise_orient == True :
       robot_init_orient = p.getQuaternionFromEuler([0,0,random.uniform(1,0.1744)*random.choice(self.perturb_direction)],
       physicsClientId=self.physicsClient)
     else :

       # право напред
       robot_init_orient = p.getQuaternionFromEuler([0,0,0]) 

     # зареждане на робота 
     self.robotID = p.loadURDF(robotVersion,robot_init_pos, robot_init_orient, 
     flags = p.URDF_USE_INERTIA_FROM_FILE,physicsClientId=self.physicsClient)

     # определяне на въртящия момент и максималната скорост на въртене 
     if self.randomise_torque :
       self.force_value = self.orig_force_value*random.uniform(0.9,1)
     if self.randomise_ang_vel :
       self.maxVel = self.orig_max_vel*random.uniform(0.9,1)

  # Метод за изчисление на наградата при робота - наградната функция е описана в секция "Наградна функция" на документацията
  def calculateReward(self):

    # изчисляване на средната скорост на робота за целия период на един сим. ход
    avgLinVel = self.LinVelSum/self.ep_limit 

    # важно: наличието на членове, наказващи употребата на твърде много въртящ момент (torque_factor),
    # завъртане на тялото (angle_factor) или отклонение настрани (x_factor) е благоприятно за ученето
    # също така наградата се умножава по константата rew_factor, защото големината й също има ефект върху ученето
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
     
     # параметри на шума в отчитанията на сензорите
     self.add_noise = add_noise 
     if self.add_noise :
       self.ang_vel_noise = 0.005
       self.rad_noise = 0.05
     else:
       self.rad_noise = 0
       self.ang_vel_noise = 0

     # версия на робота, която ще бъде използвана
     self.robotVer = robotVer 

     # максимална дължина на един сим. ход
     self.ep_limit = ep_limit 

     # параметри на моторите - максимална скорост на въртене (в радиани/секунда) и въртящ момент (в Nm)
     self.orig_max_vel = 5.23 
     self.orig_force_value = 0.9
     self.maxVel = self.orig_max_vel
     self.force_value = self.orig_force_value

     # получаване на страничните приложени сили от масива perturb_bounds - горна и долна граница на силите и посока
     # функциите amax и amin връщат най-големия и най-малкия елемент в един масив:
     # по този начин подредбата на елементите не е от значение
     self.perturb_high = np.amax(perturb_bounds) 
     self.perturb_low = np.amin(perturb_bounds)
     self.perturb_direction = [1,-1]

     # ако роботът падне преди края на симулацията, penalty_for_falling се изважда от наградата
     self.penalty_for_falling = penalty_for_falling

     # определяне на факторите за въртящ момент, отклонение настрани и накланяне на робота,
     # служещи за наказания в наградната функция, и фактора на положителната награда
     self.torque_factor = torque_factor 
     self.x_factor = x_factor
     self.angle_factor = angle_factor 
     self.rew_factor = rew_factor

     # параметри на неравен терен: дали да се генерира височинно или паралелепипедно поле (но не и двете едновременно)
     # и височината на избраното поле
     self.height_field = height_field 
     self.rect_field = rect_field
     if height_field and rect_field:
       raise Exception("Не може да се създадат два вида полета едновременно.")
     self.field_range = field_range

     # за паралелепипедното поле се избира и ширината му
     self.field_width = 0.4
 
     # променливи за сбора на въртящия момент, стойностите на ъглите на завъртане,
     # отклонението настрани и скоростта на робота
     self.torqueSum = 0
     self.angleSum = 0
     self.x_sum = 0
     self.LinVelSum = 0
     self.maxLinVel = 0

     # определя се дали роботът ще получи данни за своята посока (отчитания на компас)
     self.include_orient = include_orient 

     # определяне на това дали завъртането на робота (около оста, сочеща нагоре), въртящият момент и 
     # максималната скорост на моторите ще бъдат рандомизирани - това помага на прехвърлянето в истинския свят 
     self.randomise_orient = randomise_orient
     self.randomise_torque = randomise_torque 
     self.randomise_ang_vel = randomise_ang_vel

     # определя се дали посоката на робота ще бъде произволна за всеки сим. ход в рамките на orient_bounds
     if randomise_orient:
       self.orient_high = np.amax(orient_bounds) 
       self.orient_low = np.amin(orient_bounds) 

     # при trot = True  роботът ще получава същите команди за краката, 
     # разположени на един и същ диагонал, по този начин генерирайки тръсова походка
     self.trot = trot

     # определя дали ъглите, на които моторите се завъртат, ще бъдат закръглени според 
     # максималната резолюция на истинските мотори 
     self.round_action = round_action
     self.servo_resolution = 1.5*math.pi/180

     # ако time_only = True, единствените входни данни, 
     # които робота ще получава е времето (за създаване на периодични сигнали) 
     self.time_only = time_only 

     # ако невронната мрежа използва обикновени, ReLU неврони, времето не се добавя към входните данни на мрежата
     self.conventional_nn = conventional_nn 
     if conventional_nn:
       self.observation_params = 4
     else :
       self.observation_params = 5   
     if include_orient:
       self.observation_params += 1

     if time_only:
       if include_orient or conventional_nn :
         raise Exception("Броят на входните данни не може да бъде променен, когато time_only == True. Проверете дали include_orient или conventional_nn == True  и изберете кои входни данни ще бъдат получени.")
    

     # по този фактор се умножава времето, по-малък фактор помага за получаването на по-стабилна походка 
     # и понижава възможността за плъзгане (виж. секция "Локални минимуми" в документацията)
     self.time_factor = time_factor

     # при паралелни експерименти, симулатора трябва да бъде конкретизиран
     self.physicsClient = physics_client 

     # брой симулационни стъпки
     self._steps = 0
     
     # създаване на пространство за входни данни - изискване от OpenAI Gym framework-a
     self.observation_space = spaces.Box(low = -5, high = 5, shape = (self.observation_params,), dtype=np.float16)
     self._observation = []

     # създаване на пространство за изходни данни - изискване от OpenAI Gym framework-a
     self.action_space_low = np.array([-3.14/3, -3.14/3, 0,-3.14/3, -3.14/3, 0,-3.14/3, -3.14/3, 0,-3.14/3, -3.14/3, 0])
     self.action_space_high = np.array([3.14/3, 3.14/3, 3.14/2,3.14/3, 3.14/3, 3.14/2,3.14/3, 3.14/3, 3.14/2,3.14/3, 3.14/3, 3.14/2])
     self.action_space = spaces.Box(low = self.action_space_low, high = self.action_space_high, dtype=np.float16)

     # Тази променлива дава статуса на симулационият ход (дали е приключил)
     self.isDone = False

     # създаване на всички обекти
     self.initSim(robotVersion = self.robotVer) 

     # масиви за позициите и въртящите моменти на моторите
     self.jointposes = np.zeros((self.numJoints,))
     self.jointTorques = np.zeros((self.numJoints,))  
     
  # получава масив с изходни данни - 12 позиции на моторите (6 при тръс), генериран от невронната мрежа  
  # връща нови входни данни, наградата (ако сим. хода е приключила) 
  # и дали сим. хода е приключил
  def step(self, nn_outputs):

    # ако желаната походка е конкретизирана да бъде тръсова, трябва масивът да бъде преобразуван
    if self.trot:
      if len(nn_outputs)*2 != self.numJoints:
        raise Exception("За експерименти с походка тип тръс, полученият масив трябва да има брой на елементи, половината от броя на мотори.")
      action = np.concatenate((nn_outputs[:3],nn_outputs[3:],nn_outputs[3:],nn_outputs[:3]))
    else:
      action = nn_outputs

    # ограничаване на стойностите на ъгъла (не трябва да надвишават стойностите, посочени в пространството на изходните данни)
    action = np.clip(action, self.action_space_low,self.action_space_high)

    # ограничаване на прецизността на изходните данни
    if self.round_action:
     action = [np.round(cmd/self.servo_resolution)*self.servo_resolution for cmd in action]

    # прилагане на действието в симулацията (цикълът чрез итератори изисква по-малко време за завършване)
    [p.setJointMotorControl2(self.robotID,j,p.POSITION_CONTROL, i, 
    force = self.force_value, maxVelocity = self.maxVel,physicsClientId=self.physicsClient) for i,j in zip(action,self.joints)]

    # задействане на симулационната стъпка
    p.stepSimulation(physicsClientId=self.physicsClient)

    # получаване на позициите и въртящия момент на моторите
    self.jointposes,self.jointTorques =  [p.getJointState(self.robotID, self.joints[i],physicsClientId=self.physicsClient)[0] for i in range(self.numJoints)],[p.getJointState(self.robotID, 
    self.joints[i],physicsClientId=self.physicsClient)[3] for i in range(self.numJoints)]
    self._steps = self._steps + 1
    
    # получаване на позиция, завъртане, линейна и ъглова скорост
    pos, orientQuaternion = p.getBasePositionAndOrientation(self.robotID,physicsClientId=self.physicsClient)
    orientRad = p.getEulerFromQuaternion(orientQuaternion,physicsClientId=self.physicsClient)
    bodyLinVel,bodyAngVel = p.getBaseVelocity(self.robotID,physicsClientId=self.physicsClient)

    if bodyLinVel[1] > self.maxLinVel :
      self.maxLinVel = abs(bodyLinVel[1])
    self.LinVelSum += bodyLinVel[1]
    self.torqueSum +=sum(np.absolute(self.jointTorques))
    self.angleSum += sum(np.absolute(orientRad)) 
    self.x_sum += np.absolute(pos[0])

    # прилагане на външна сила настрани на всеки 300 стъпки за 5 стъпки
    if self._steps%300 == 0 and self._steps >=300 :
      p.applyExternalForce(self.robotID, -1, 
      [random.uniform(self.perturb_low,self.perturb_high)*random.choice(self.perturb_direction),
      random.uniform(self.perturb_low*0.1,self.perturb_high*0.1)*random.choice(self.perturb_direction),
      0],[0,random.uniform(-0.1,0.1),0.1],p.LINK_FRAME,physicsClientId=self.physicsClient)

    # създаване на нови входни данни за невронната мрежа, с частни случаи за описаните горе опции
    self._observation = np.concatenate((orientRad[0]+ random.uniform(-self.rad_noise,self.rad_noise), 
    orientRad[1]+ random.uniform(-self.rad_noise,self.rad_noise),
    bodyAngVel[0] + random.uniform(-self.ang_vel_noise, self.ang_vel_noise), 
    bodyAngVel[1] + random.uniform(-self.ang_vel_noise, self.ang_vel_noise), 
    self._steps*self.time_factor), axis=None)

    # получаване само на времето
    if self.time_only :
      self._observation = self._steps*self.time_factor

    # данни за класическа невронна мрежа
    if self.conventional_nn:
      self._observation = np.concatenate((orientRad[0]+ random.uniform(-self.rad_noise,self.rad_noise), 
      orientRad[1]+ random.uniform(-self.rad_noise,self.rad_noise), 
      bodyAngVel[0] + random.uniform(-self.ang_vel_noise, self.ang_vel_noise), 
      bodyAngVel[1] + random.uniform(-self.ang_vel_noise, self.ang_vel_noise) 
      ), axis=None)

    # добавяне на ориентацията (като компас)
    if self.include_orient:
      self._observation = np.insert(self._observation,2,orientRad[2]+ random.uniform(-self.rad_noise,self.rad_noise),axis = None)
    reward = 0

    # когато максималния брой стъпки бъде достигнат, симулацията се прекратява
    if self._steps >= self.ep_limit:
      self.isDone = True
      reward = self.calculateReward()

    # ако роботът се наклони настрани или напред-назад с повече от 0.5 стъпки или се извърти назад,
    # или при трениране за посока излезне от предвиденото поле,
    # ходът се прекратява
    if abs(orientRad[0]) >= 0.5 or abs(orientRad[1]) >= 0.5 or abs(orientRad[2]) >= 1.57 or abs(pos[0])*self.randomise_orient >= self.field_width/2 :
      if self._steps >= 700 : # ако роботът не може да остане прав за минимум 700 стъпки, той не получава награда
        reward = self.calculateReward()
        reward -= self.penalty_for_falling
      else :
        reward = -self.penalty_for_falling
      self.isDone = True

    return self._observation, reward, self.isDone, {}

  # рестартиране на симулацията и анулиране на всички данни
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

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

   
