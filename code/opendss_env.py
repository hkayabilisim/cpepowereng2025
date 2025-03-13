from typing import Any
import gymnasium as gym
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Tuple, Box
import numpy as np
import opendssdirect as dss
import random
import os
from itertools import product
import matplotlib.pyplot as plt

class OpenDSS(gym.Env):

    MAX_STEPS = 30
    UPPER_VOLTAGE_MAG = 1.05
    LOWER_VOLTAGE_MAG = 0.95
    MAX_REG_TAP_VALUE =16
    MIN_REG_TAP_VALUE =-16
    NUM_REG = 7
    NUM_INVERTER = 3
    NUM_ELEMENT_REG_TAP = MAX_REG_TAP_VALUE-MIN_REG_TAP_VALUE+1
    MIN_X1_VALUE = 0.93
    MAX_X1_VALUE = 0.99
    MIN_X2_VALUE = 1.01
    MAX_X2_VALUE = 1.07
    NUM_ELEMENT_X1 = int(round(MAX_X1_VALUE-MIN_X1_VALUE,2)*100+1)
    NUM_ELEMENT_X2 = int(round(MAX_X2_VALUE-MIN_X2_VALUE,2)*100+1)
    CHOOSEN_HOUR = 12 #0.5-0.7 PV Irradiance

    def load_case(self):
        dss.Text.Command('Clear')
        try:
            scenario_path = "../data/IEEE123Master_Loads.dss"
            dss.Text.Command(f'Redirect {scenario_path}')
            dss.Text.Command(f'set mode=daily number=1 stepsize=1h')
            dss.Text.Command('set number = 1')
            dss.Text.Command('set step = 1h')
            dss.Text.Command('set maxcontroliter = 10000')
            dss.Text.Command('set maxiter = 10000')
            self._set_max_tap_change(np.zeros(dss.RegControls.Count()))
        except:
            raise Exception("Invalid test case!")

    def _set_max_tap_change(self, tap_num):
        dss.RegControls.First()
        for i in range(dss.RegControls.Count()):
            dss.Text.Command(f'RegControl.{dss.RegControls.Name()}.MaxTapChange={tap_num[i]}')
            dss.RegControls.Next()


    def get_tap_values(self):
        dss.RegControls.First()
        values = []
        for i in range(dss.RegControls.Count()):
            values.append(dss.RegControls.TapNumber())
            dss.RegControls.Next()
        return values
    
    def set_tap_values(self, values):
        dss.RegControls.First()
        for i in range(dss.RegControls.Count()):
            dss.Text.Command(f'RegControl.{dss.RegControls.Name()}.TapNum={values[i]}')
            dss.RegControls.Next()

    def set_pv_values(self,name, x1, x2):
        dss.Text.Command(f'XYCurve.{name}.XArray = [0.5 {x1} 1 1 1 {x2} 1.5]')
        dss.XYCurves.Name(f'{name}')


    def solve_case(self, hour):
        dss.Text.Command(f'set hour = {hour}')
        dss.Solution.Solve()

    def calculate_error(self,p1,p2,p3):
        p1_error = np.linalg.norm(p1-np.ones(len(p1)))
        p2_error = np.linalg.norm(p2-np.ones(len(p2)))
        p3_error = np.linalg.norm(p3-np.ones(len(p3)))
        return np.mean([p1_error, p2_error, p3_error])
    

    def get_phase_values(self):
        p1 = dss.Circuit.AllNodeVmagPUByPhase(1)
        p2 = dss.Circuit.AllNodeVmagPUByPhase(2)
        p3 = dss.Circuit.AllNodeVmagPUByPhase(3)
        return p1, p2, p3


    def calculate_distance_standard(self, p1, p2, p3):
        max_size = max(len(p1), len(p2), len(p3))
        total_violation = 0

        for idx in range(max_size):
            elements = [p1[idx] if idx < len(p1) else None,
                        p2[idx] if idx < len(p2) else None,
                        p3[idx] if idx < len(p3) else None]
        
            filtered_list = [x for x in elements if x is not None]
            above_max_penalty = max(filtered_list) - self.UPPER_VOLTAGE_MAG
            below_min_penalty = self.LOWER_VOLTAGE_MAG - min(filtered_list)
            total_violation += max(above_max_penalty, 0) + max(below_min_penalty, 0)

        return total_violation
    
    def read_Regulators(self):
        dss.RegControls.First()
        for _ in range(dss.RegControls.Count()):
            print(f'Name : {dss.RegControls.Name()} Tap : {dss.RegControls.TapNumber()} MaxChange: {dss.RegControls.MaxTapChange()}')
            dss.RegControls.Next()
    
    def read_XYCurves(self):
        dss.XYCurves.First()
        for _  in range(dss.XYCurves.Count()):
            print(f'Name : {dss.XYCurves.Name()} Value : {dss.XYCurves.XArray()}')
            dss.XYCurves.Next()
    
    def flatten(self,arr):
        result = []
        for i in arr:
            if isinstance(i, list):
                result.extend(self.flatten(i))
            else:
                result.append(i)
        return result

    def __init__(self, case="IEEE123PV"):
        
        self.action_space = MultiDiscrete([(self.NUM_ELEMENT_REG_TAP), (self.NUM_ELEMENT_REG_TAP),
                                            (self.NUM_ELEMENT_REG_TAP), (self.NUM_ELEMENT_REG_TAP),
                                            (self.NUM_ELEMENT_REG_TAP),(self.NUM_ELEMENT_REG_TAP), 
                                            (self.NUM_ELEMENT_REG_TAP), 
                                            (self.NUM_ELEMENT_X1), (self.NUM_ELEMENT_X2), # x1 and x2 should start 930 and 1010,respectively
                                            (self.NUM_ELEMENT_X1),(self.NUM_ELEMENT_X2),
                                            (self.NUM_ELEMENT_X1),(self.NUM_ELEMENT_X2)])
        
        self.observation_space = Box(low=np.array([self.MIN_REG_TAP_VALUE,self.MIN_REG_TAP_VALUE, self.MIN_REG_TAP_VALUE,
                                                   self.MIN_REG_TAP_VALUE, self.MIN_REG_TAP_VALUE, self.MIN_REG_TAP_VALUE,
                                                   self.MIN_REG_TAP_VALUE, self.MIN_X1_VALUE, self.MIN_X1_VALUE, self.MIN_X1_VALUE,
                                                   self.MIN_X2_VALUE, self.MIN_X2_VALUE, self.MIN_X2_VALUE, self.CHOOSEN_HOUR], dtype=np.float32),
                                     high=np.array([self.MAX_REG_TAP_VALUE, self.MAX_REG_TAP_VALUE, self.MAX_REG_TAP_VALUE,
                                                   self.MAX_REG_TAP_VALUE, self.MAX_REG_TAP_VALUE, self.MAX_REG_TAP_VALUE,
                                                   self.MAX_REG_TAP_VALUE, self.MAX_X1_VALUE, self.MAX_X1_VALUE, self.MAX_X1_VALUE,
                                                   self.MAX_X2_VALUE, self.MAX_X2_VALUE, self.MAX_X2_VALUE, self.CHOOSEN_HOUR], dtype=np.float32), dtype=np.float32)
        
        
        self.case = case
        self.load_case()
        self.xy_curve_names = [eleman for eleman in dss.XYCurves.AllNames() if 'voltvar_curve' in eleman]
        self.default_load_shape = [0.934, 0.87, 0.826, 0.805, 0.8, 0.805, 0.85, 
                                   0.931, 1.115, 1.232, 1.253, 1.262, 1.207, 1.219, 
                                   1.221, 1.211, 1.218, 1.219, 1.198, 1.204, 1.169, 1.122, 1.076, 1.019]

    def reset(self, *, seed=None, options=None):

        tap_values = list(np.random.randint(self.MIN_REG_TAP_VALUE,self.MAX_REG_TAP_VALUE+1,self.NUM_REG))
        self.set_tap_values(tap_values)

        pv_array = []
        pv_array_x1 = []
        pv_array_x2 = []
        for xy_curve_name in self.xy_curve_names:
            x1 = round(self.MIN_X1_VALUE + (self.MAX_X1_VALUE - self.MIN_X1_VALUE) * random.random(), 2)
            x2 = round(self.MIN_X2_VALUE + (self.MAX_X2_VALUE - self.MIN_X2_VALUE) * random.random(), 2)
            pv_array_x1.extend([x1])
            pv_array_x2.extend([x2])
            self.set_pv_values(xy_curve_name, x1, x2)
        pv_array.extend([pv_array_x1,pv_array_x2])
        
        dss.LoadShape.Name('loadshape_1')
        dss.LoadShape.PMult(self.default_load_shape)
        self.solve_case(self.CHOOSEN_HOUR)

        p1, p2, p3 = self.get_phase_values()
        error = self.calculate_error(p1,p2,p3)
        self.reward = -self.calculate_distance_standard(p1,p2,p3) - error

        self.info = {}
        self.state = self.flatten(tap_values+pv_array+[self.CHOOSEN_HOUR])
    

        self.truncated = False
        self.terminated = False
        self.count = 0
        self.count_state_in_range_episode = 0
        self.info = {
            "error": error,
            "p1_max": np.max(p1),
            "p1_min": np.min(p1),
            "p1_mean": np.mean(p1),
            "p2_max": np.max(p2),
            "p2_min": np.min(p2),
            "p2_mean": np.mean(p2),
            "p3_max": np.max(p3),
            "p3_min": np.min(p3),
            "p3_mean": np.mean(p3)
        }

        return self.state, self.info


    def step(self, action):
        try:
            assert self.action_space.contains(action)
        except AssertionError:
            print(f"Invalid action: {action}")
        self.count += 1
        temp_state = self.flatten(self.state.copy())
        
        temp_state[:self.NUM_REG] = np.array(action[:self.NUM_REG])-self.MAX_REG_TAP_VALUE
        for i in range(self.NUM_REG, self.NUM_INVERTER+self.NUM_REG):
            
            temp_state[i] = round(action[i]/100+self.MIN_X1_VALUE,2)

        for i in range(self.NUM_INVERTER+self.NUM_REG, self.NUM_INVERTER*2+self.NUM_REG):
        
            temp_state[i] = round(action[i]/100+self.MIN_X2_VALUE,2)
            
        temp_state[-1] = self.CHOOSEN_HOUR

        self.state = temp_state.copy()

        next_tap_values = temp_state[:self.NUM_REG]
        self.set_tap_values(next_tap_values)

        for i in range(self.NUM_INVERTER):
            self.set_pv_values(self.xy_curve_names[i], temp_state[self.NUM_REG+i], temp_state[self.NUM_REG+i+self.NUM_INVERTER])

        try:
            self.solve_case(self.CHOOSEN_HOUR)
        except:
            print("Error",temp_state)

        p1, p2, p3 = self.get_phase_values()
        
        error = self.calculate_error(p1,p2,p3)
        self.reward = -self.calculate_distance_standard(p1,p2,p3) - error

        
        self.info["error"] = error
        self.info["p1_max"] = np.max(p1)
        self.info["p1_min"] = np.min(p1)
        self.info["p1_mean"] = np.mean(p1)
        self.info["p2_max"] = np.max(p2)
        self.info["p2_min"] = np.min(p2)
        self.info["p2_mean"] = np.mean(p2)
        self.info["p3_max"] = np.max(p3)
        self.info["p3_min"] = np.min(p3)
        self.info["p3_mean"] = np.mean(p3)

        if self.count == self.MAX_STEPS:
            self.truncated = True
    
        return self.state, self.reward, self.terminated, self.truncated, self.info