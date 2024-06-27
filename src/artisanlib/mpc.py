import time
import numpy
import scipy.signal # type:ignore[import-untyped]
import logging
from typing import Final, List, Optional, Callable

from artisanlib.filters import LiveSosFilter

try:
    from PyQt6.QtCore import QSemaphore # @Reimport @UnresolvedImport @UnusedImport
except ImportError:
    from PyQt5.QtCore import QSemaphore # type: ignore # @Reimport @UnresolvedImport @UnusedImport

_log: Final[logging.Logger] = logging.getLogger(__name__)

# expects a function control that takes a value from [<outMin>,<outMax>] to control the heater as called on each update()
class MPC:
    __slots__ = []

    def __init__(self, control:Callable[[float], None]=lambda _: None) -> None:
        self.mpcSemaphore:QSemaphore = QSemaphore(1)

        self.outMin:int = 0 # minimum output value
        self.outMax:int = 100 # maximum output value
        self.dutySteps:int = 1 # change [1-10] between previous and new MPC duty to trigger call of control function
        self.dutyMin:int = 0
        self.dutyMax:int = 100
        self.control:Callable[[float], None] = control

        self.lastOutput:Optional[float] = None # used to reinitialize the Iterm and to apply simple moving average on the derivative part in derivative_on_measurement mode

        #MPC - LSTM Variables
        #
        #
        #
        #
        #
        #
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
    def update(self, i:Optional[float]) -> None:
        try:
            if i == -1 or i is None:
                # reject error values
                return
            self.mpcSemaphore.acquire(1)
           
            #TODO: create MPC control function here to control the heater
            #
            #          HERE IS THE MPC CONTROL FUNCTION
            #
            output = 0 ### MPC control function here to control the heater

            # clamp output to [outMin,outMax]
            if output > self.outMax:
                output = self.outMax
            elif output < self.outMin:
                output = self.outMin

            int_output = int(round(min(self.dutyMax,max(self.dutyMin,output))))
            if self.lastOutput is None or self.iterations_since_duty >= self.force_duty or int_output >= self.lastOutput + self.dutySteps or int_output <= self.lastOutput - self.dutySteps:
                if self.active:
                    self.control(int_output)
                    self.iterations_since_duty = 0
                self.lastOutput = output # kept to initialize Iterm on reactivating the PID
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

    def setMPC(self) -> None:
        try:
            self.mpcSemaphore.acquire(1)
            #TODO: set the MPC parameters here
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