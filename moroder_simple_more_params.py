import pybullet as p
import numpy as np
import pybullet_data
import tensorflow as tf
from es import CMAES
import pybullet_data
from time import sleep
import pickle 
from math import sin,cos,pi
joints = [1,3,6,8,11,13,16,18]
num_params = 7
max_ang_vel = 5.23
max_torque = 0.9
MAX_ITERATION = 100

def sine_signal(a,w,t,offset,phi,freq_limit = 2*pi/120):
    w = np.clip(w,-freq_limit,freq_limit)
    return a*sin(w*t+phi)+offset

def fitness(paramlist,robotID) :
    reward = 0
    for ij in range(7200):
        sleep( 1/240)
        r_femur = sine_signal(paramlist[0],paramlist[1],ij,paramlist[2],0)
        r_tibia = sine_signal(paramlist[3],paramlist[4],ij,paramlist[5],paramlist[6])
        l_femur = sine_signal(paramlist[0],paramlist[1],ij,paramlist[2],pi)
        l_tibia = sine_signal(paramlist[3],paramlist[4],ij,paramlist[5],pi + paramlist[6])
        #action = [r_femur,r_tibia,r_femur,r_tibia,l_femur,l_tibia,l_femur,l_tibia] #for first 4 saves
        action = [r_femur,r_tibia,l_femur,l_tibia,l_femur,l_tibia,r_femur,r_tibia]
        for jj in range(8):
            
            p.setJointMotorControl2(robotID,joints[jj],p.POSITION_CONTROL, action[jj], force = max_torque, maxVelocity = max_ang_vel) 
        p.stepSimulation()
        lin_vel,_ = p.getBaseVelocity(robotID)
        _, orientQuaternion = p.getBasePositionAndOrientation(robotID)
        orientRad = p.getEulerFromQuaternion(orientQuaternion)
        reward += lin_vel[1]/72
        if abs(orientRad[0]) >= 0.3 or abs(orientRad[1]) >= 0.3 or abs(orientRad[2]) >= 1.57 :
            p.resetSimulation()
            p.setPhysicsEngineParameter(enableConeFriction=0)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0,0,-9.81)
            p.loadURDF("plane.urdf")
            robotID = p.loadURDF("/home/mr-mecha/Desktop/Moroder/moroder.urdf",[0,0,0.15],flags = p.URDF_USE_INERTIA_FROM_FILE)
            return reward
    p.resetSimulation()
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    p.loadURDF("plane.urdf")
    robotID = p.loadURDF("/home/mr-mecha/Desktop/Moroder/moroder.urdf",[0,0,0.15],flags = p.URDF_USE_INERTIA_FROM_FILE)
    return reward

def test_solver(solver):
    history = []
    for j in range(MAX_ITERATION):
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)
        for i in range(solver.popsize):
            fitness_list[i] = fitness(solutions[i],robotID)
        solver.tell(fitness_list)
        result = solver.result() # first element is the best solution, second element is the best fitness
        history.append(result[1])
        if (j+1) % 1 == 0:
            print("fitness at iteration", (j+1), result[1])
            print("params:\n", (j+1), result[0])
    print("local optimum discovered by solver:\n", result[0])
    print("fitness score at this local optimum:", result[1])
    return history

# F = front, H = hind, R and L = right and left
#1=FR femur motor
#3=FR tibia motor

#6 = HR femur motor
#8 = HR tibia motor

#11 = FL femur motor
#13 = FL tibia motor

#16 = HL femur motor
#18 = HL tibia motor
if __name__ == "__main__" :
    client = p.connect(p.GUI)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    p.loadURDF("plane.urdf")
    robotID = p.loadURDF("/home/mr-mecha/Desktop/Moroder/moroder.urdf",[0,0,0.15],flags = p.URDF_USE_INERTIA_FROM_FILE)
    cma = CMAES(num_params,
    sigma_init=0.2,
    weight_decay=0,
    popsize=100) 
    #cma_history = test_solver(cma)
    #fitness([-0.25035065,  0.19407259, -0.1564238,  0.33457894,  0.11721902, -0.20364131, -0.06148436],robotID)
    fitness( [-0.15746397,  0.53776289, -0.51510807,  8.26656121, -1.90173302,  0.35729393, -1.80671983],robotID)