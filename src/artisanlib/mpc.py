import time
import numpy
import random
import scipy.signal # type:ignore[import-untyped]
import logging
from typing import Final, List, Optional, Callable

#import for keras model, load model
from keras.models import load_model,Model

from artisanlib.filters import LiveSosFilter

try:
    from PyQt6.QtCore import QSemaphore # @Reimport @UnresolvedImport @UnusedImport
except ImportError:
    from PyQt5.QtCore import QSemaphore # type: ignore # @Reimport @UnresolvedImport @UnusedImport

_log: Final[logging.Logger] = logging.getLogger(__name__)

# expects a function control that takes a value from [<outMin>,<outMax>] to control the heater as called on each update()
class MPC:
    __slots__ = ['mpcSemaphore','outMin','outMax','dutySteps','dutyMin','dutyMax','control','lastOutput','iterations_since_duty','force_duty',
                 'active','target','model','modelPath', 'yellowingTemp','FCtemp','DROPtemp','yellowingTime','FCtime','DROPtime']

    def __init__(self, control:Callable[[float], None]=lambda _: None, modelPath:str = "",
                 yellowingTemp:float = 160, FCtemp:float = 198, DROPtemp:float = 211,yellowingTime:float = 300,
                FCtime:float = 555,DROPtime:float = 700) -> None:
        self.mpcSemaphore:QSemaphore = QSemaphore(1)
        self.active:bool = False

        self.outMin:int = 0 # minimum output value
        self.outMax:int = 100 # maximum output value
        self.dutySteps:int = 1 # change [1-10] between previous and new MPC duty to trigger call of control function
        self.dutyMin:int = 0
        self.dutyMax:int = 100
        self.control:Callable[[float], None] = control

        self.lastOutput:Optional[float] = None

        #MPC - LSTM Variables
        #
        #
        #
        self.modelPath:Optional[str] = modelPath # used to reinitialize the model to use for prediction        
        self.model:Optional[Model] = None
        self.yellowingTemp:float = yellowingTemp # goal yellowing temperature to reach in °C
        self.FCtemp:float = FCtemp
        self.DROPtemp:float = DROPtemp
        self.yellowingTime:float = yellowingTime # time to reach yellowing temperature in seconds
        self.FCtime:float = FCtime
        self.DROPtime:float = DROPtime
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
    def update(self,et,bt,tx) -> None:
        try:
            self.mpcSemaphore.acquire(1)
           
            #TODO: create MPC control function here to control the heater
            #
            #          HERE IS THE MPC CONTROL FUNCTION
            #
            #mock for now with changing output value between as random value between outMax and outMin
            output = random.uniform(self.outMin, self.outMax)

            # clamp output to [outMin,outMax]
            if output > self.outMax:
                output = self.outMax
            elif output < self.outMin:
                output = self.outMin

            int_output = int(round(min(self.dutyMax,max(self.dutyMin,output))))
            if self.active:
                self.control(int_output)
                self.iterations_since_duty = 0
            self.iterations_since_duty += 1
        except Exception as e: # pylint: disable=broad-except
            _log.exception(e)
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

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