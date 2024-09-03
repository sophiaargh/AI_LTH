
import random
import numpy as np

from models import *


#
# Add your Filtering / Smoothing approach(es) here
#
class HMMFilter:
    def __init__(self, probs, tm : TransitionModel, om, sm: StateModel): ##tm:transition model, om:observation model, sm:states model
        self.__tm = tm
        self.__om = om
        self.__sm = sm
        self.__f = probs
        
    # implementing first the forward filter from the lecture notes
    # but we only have one observation (sensorR) and not a list
    def filter(self, sensorR) :
        f = self.__f
        O = self.__om
        T = self.__tm
        f = O.get_o_reading(sensorR) @ T.get_T() @ f
        f /= np.sum(f)
        self.__f = f
        return f
        
        
    def backward_smoothing(self, sensor_array, probs_array):
        n = len(sensor_array) # 5 in our implementation (5 steps back)
        m = self.__sm.get_num_of_states()
        b_array = []
        s = np.zeros(m)
        b = np.ones(m) #/ m
        for i in range(n - 2, -1, -1): #backward loop of 5 steps to get all the values in b
            b = self.__tm.get_T_transp() @ self.__om.get_o_reading(sensor_array[n-1-i]) @ b
            b = np.clip(b, a_min = 1e-10, a_max = None) #ensures that there is no division by 0
            b /= np.sum(b)
            b_array.insert(0,b)

        #we just want the smoothed value at the first timestep (so using the probability from 5 moves behind)
        s = probs_array[n-1] * b_array[0]
        s = np.clip(s, a_min = 1e-10, a_max = None)
        s /= np.sum(s)
        return s
        
