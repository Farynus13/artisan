import time
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from typing import Final, List, Optional, Callable
#import for keras model, load model
from keras.models import load_model,Model

try:
    from PyQt6.QtCore import QSemaphore # @Reimport @UnresolvedImport @UnusedImport
except ImportError:
    from PyQt5.QtCore import QSemaphore # type: ignore # @Reimport @UnresolvedImport @UnusedImport

_log: Final[logging.Logger] = logging.getLogger(__name__)

# expects a function control that takes a value from [<outMin>,<outMax>] to control the heater as called on each update()
class MPC:
    __slots__ = ['mpcSemaphore','outMin','outMax','dutySteps','dutyMin','dutyMax','control','lastOutput','iterations_since_duty','force_duty',
                 'active','target','model','modelPath', 'yellowingTemp','FCtemp','DROPtemp','yellowingTime','FCtime','DROPtime',
                 'step','input_horizon','output_horizon','xScaler','yScaler', 'min_input_length','modelLoaded'] 

    def __init__(self, control:Callable[[float], None]=lambda _: None, modelPath:str = "",
                 yellowingTemp:float = 160, FCtemp:float = 198, DROPtemp:float = 211,yellowingTime:float = 300,
                FCtime:float = 555,DROPtime:float = 700) -> None:
        self.mpcSemaphore:QSemaphore = QSemaphore(1)
        self.active:bool = False
        self.modelLoaded:bool = False

        self.outMin:int = 0 # minimum output value
        self.outMax:int = 100 # maximum output value
        self.dutySteps:int = 1 # change [1-10] between previous and new MPC duty to trigger call of control function
        self.dutyMin:int = 0
        self.dutyMax:int = 100
        self.control:Callable[[float], None] = control

        self.lastOutput:Optional[float] = None
        self.force_duty:int = 5 # at least every n update cycles a new duty value is send, even if its duplicating a previous duty (within the duty step)

        #MPC - LSTM Variables
        #
        #
        #
        self.modelPath:Optional[str] = modelPath # used to reinitialize the model to use for prediction     
        self.model:Optional[Model] = None
        self.xScaler:Optional[MinMaxScaler] = None # used to scale input data for prediction
        self.yScaler:Optional[MinMaxScaler] = None # used to unscale output data after prediction as outputs are scaled to [0, 1] None
        #because this LSTM model is autoregressive (uses outputs as inputs in a prediction loop)

        self.yellowingTemp:float = yellowingTemp # goal yellowing temperature to reach in °C
        self.FCtemp:float = FCtemp
        self.DROPtemp:float = DROPtemp
        self.yellowingTime:float = yellowingTime # time to reach yellowing temperature in seconds
        self.FCtime:float = FCtime
        self.DROPtime:float = DROPtime
        #
        self.step:float = 5. # time step in seconds
        self.input_horizon:int = int(6*60/self.step) # number of input horizon steps (hardcoded to 6 minutes)
        self.output_horizon:int = int(6*60/self.step) # number of output horizon steps hardcoded to 6 minutes, but will change based on next milestone (DRY, FC, DROP)
        self.min_input_length:int = int(10/self.step)

        #-----------------------------------------------------------------------------------

    ### External API guarded by semaphore

    def on(self) -> None:
        try:
            self.mpcSemaphore.acquire(1)
            #TODO: some stuff when switching on MPC mode
            self.active = True
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def off(self) -> None:
        try:
            self.mpcSemaphore.acquire(1)
            self.active = False
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def isActive(self) -> bool:
            try:
                self.mpcSemaphore.acquire(1)
                return self.active
            finally:
                if self.mpcSemaphore.available() < 1:
                    self.mpcSemaphore.release(1)

    # update control value (the mpc loop is running even if MPC is inactive, just the control function is only called if active)
    def update(self,et,bt,tx,targetTime,targetValues,delay,charge_idx,target_offset,target_factor) -> None:
        try: 
            self.mpcSemaphore.acquire(1)
            if self.active and charge_idx != -1 and \
                targetTime is not None and targetValues is not None and \
                len(targetTime) == len(targetValues) > 0 and \
                len(et[charge_idx:]) / self.step >= self.min_input_length:
                targetValues = [((x - target_offset) / target_factor) for x in targetValues]

                # len(et[charge_idx:]) / self.step >= self.min_input_length: # check if enough data is available to make a prediction
                # predictionData = self.preprocessData(et,bt,targetTime,targetValues,delay,charge_idx) # preprocess data for prediction
                # output = self.findOptimalDuty(predictionData) # find optimal duty value to reach DRY / FC / DROP temperature in the right time
                output = targetValues[-1] + 1
                # clamp output to [outMin,outMax]
                if output > self.outMax:
                    output = self.outMax
                elif output < self.outMin:
                    output = self.outMin

                int_output = int(round(min(self.dutyMax,max(self.dutyMin,output))))
                self.control(int_output)
                self.iterations_since_duty = 0
                self.lastOutput = output # kept to initialize Iterm on reactivating the PID
            self.iterations_since_duty += 1

        except Exception as e: # pylint: disable=broad-except
            _log.exception(e)
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def preprocessData(self,et,bt,targetTime,targetValues,delay,charge_idx) -> List[List[float]]:
        et = et[charge_idx:]
        bt = bt[charge_idx:]
        burner = [0]*len(et)
        i = 0
        for timex,valuex in zip(targetTime,targetValues):
            while i < len(burner) and i*delay <= timex:
                burner[i] = valuex
                i += 1
        if i < len(burner):
            while i < len(burner):
                burner[i] = burner[i-1]
                i += 1

        roast = np.array([et,bt,burner]).T
        roast = self.xScaler.transform(roast) # scale input data for prediction to [0,1] return roast
        for j in range(int(roast.shape[0]/self.step-self.min_input_length-1)):
            x_temp = roast[:(j+self.min_input_length)*self.step][::self.step]
            if x_temp.shape[0] >= self.input_horizon:
                x_temp = x_temp[x_temp.shape[0]-self.input_horizon:]
            else:
                x_temp = np.pad(x_temp, ((self.input_horizon - x_temp.shape[0], 0), (0, 0)), mode='constant')

        return np.expand_dims(x_temp,0)
        # return None
    
    def findOptimalDuty(self,predictionData) -> float:
        if self.modelLoaded and predictionData is not None:
            #perform optimization over the space of duty values over time to reach the target temperature in the right time
            pass
            

    # bring the mpc to its initial state (to be called externally)
    def reset(self) -> None:
        try:
            self.mpcSemaphore.acquire(1)
            self.init()
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    # re-initalize the MPC on restarting it after a temporary off state
    def init(self) -> None:
        try:
            self.mpcSemaphore.acquire(1)
            #TODO: re-initialize some variables here to restart MPC loop

          
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def setTarget(self, target:float, init:bool = True) -> None:
        try:
            self.mpcSemaphore.acquire(1)
            self.target = target
            if init:
                self.init()
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def getTarget(self) -> float:
        try:
            self.mpcSemaphore.acquire(1)
            return self.target
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def setMPC(self, modelPath:str, yellowingTemp:float, FCtemp:float, DROPtemp:float, 
    yellowingTime:float, FCtime:float, DROPtime:float) -> None:
        try:
            self.mpcSemaphore.acquire(1)
            #TODO: set the MPC parameters here

            self.modelPath = modelPath  
            self.yellowingTemp = yellowingTemp
            self.FCtemp = FCtemp
            self.DROPtemp = DROPtemp
            self.yellowingTime = yellowingTime
            self.FCtime = FCtime
            self.DROPtime = DROPtime 
            self.loadModel()
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def setLimits(self, outMin:int, outMax:int) -> None:
        try:
            self.mpcSemaphore.acquire(1)
            self.outMin = outMin
            self.outMax = outMax
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def setDutySteps(self, steps:int) -> None:
        try:
            self.mpcSemaphore.acquire(1)
            self.dutySteps = steps
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def setDutyMin(self, m:int) -> None:
        try:
            self.mpcSemaphore.acquire(1)
            self.dutyMin = m
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def setDutyMax(self, m:int) -> None:
        try:
            self.mpcSemaphore.acquire(1)
            self.dutyMax = m
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def setControl(self, f:Callable[[float], None]) -> None:
        try:
            self.mpcSemaphore.acquire(1)
            self.control = f
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def getDuty(self) -> Optional[float]:
        try:
            self.mpcSemaphore.acquire(1)
            if self.lastOutput is not None:
                return int(round(min(self.dutyMax,max(self.dutyMin,self.lastOutput))))
            return None
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def setPath(self, path:str) -> None:
        try:
            self.mpcSemaphore.acquire(1)
            if path is not None and path != "":
                self.modelPath = path
                #TODO: more validation here to check if the path is correct
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1) 

    def getPath(self) -> str:
        try:
            self.mpcSemaphore.acquire(1)
            return self.modelPath
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def loadModel(self) -> None:
        if self.modelPath is not None and self.modelPath != "":
            xscaler_path:str = self.modelPath.replace('.keras','.joblib').replace('model','x_scaler')
            yscaler_path:str = self.modelPath.replace('.keras','.joblib').replace('model','y_scaler')
            try:
                self.model = load_model(self.modelPath)
                self.xScaler = joblib.load(xscaler_path)
                self.yScaler = joblib.load(yscaler_path) 
                self.modelLoaded = True
            except Exception as e:
                _log.exception(f'Failed to load model,{self.modelPath}. Error: {e}')      
                self.modelLoaded = False # failed to load model                  
        else:
            _log.info('Incorrect model path')                                      

