
import time as libtime
import numpy
import logging
from typing import Final, Union, List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from artisanlib.main import ApplicationWindow # pylint: disable=unused-import

from artisanlib.util import decs2string, fromCtoFstrict, fromFtoCstrict, hex2int, str2cmd, stringfromseconds, cmd2str, float2float

try:
    from PyQt6.QtCore import pyqtSlot # @UnusedImport @Reimport  @UnresolvedImport
    from PyQt6.QtWidgets import QApplication # @UnusedImport @Reimport  @UnresolvedImport
except ImportError:
    from PyQt5.QtCore import pyqtSlot # type: ignore # @UnusedImport @Reimport  @UnresolvedImport
    from PyQt5.QtWidgets import QApplication # type: ignore # @UnusedImport @Reimport  @UnresolvedImport

_log: Final[logging.Logger] = logging.getLogger(__name__)


###################################################################################
############################# MPC CLASS DEFINITION  ##############################
###################################################################################
 
class MPCcontrol:
    __slots__ = ['aw','mpcActive','mpcDRYTemp','mpcFCTemp','mpcDROPTemp','mpcDRYTime','mpcFCTime','mpcDROPTime',
                 'dutySteps','dutyMin','dutyMax','targetRangeLimit','targetMin','targetMax','target',
                    'slider_force_move','createEvents','mpcSource','time_mpcON','modelPath','mpcOnCHARGE']

    def __init__(self, aw: 'ApplicationWindow') -> None:
        self.aw:ApplicationWindow = aw
        self.mpcActive:bool = False
        self.mpcOnCHARGE:bool = False

        self.createEvents:bool = False

        self.mpcDRYTemp:float = 160
        self.mpcFCTemp:float = 198
        self.mpcDROPTemp:float = 211
        self.mpcDRYTime:float = 300
        self.mpcFCTime:float = 555
        self.mpcDROPTime:float = 700


        self.dutySteps:int = 1
        self.dutyMin:int = 0
        self.dutyMax:int = 100
        self.targetRangeLimit:bool = False 
        self.targetMin:int = 0
        self.targetMax:int = 100
        self.target:int = 0

        self.mpcSource:int = 1
        self.time_mpcON:float = 0

        self.slider_force_move:bool = True # if True move the slider independent of the slider position to fire slider action!

        self.modelPath:str = None # used to reinitialize the model to use for prediction
        
    def toggleMPC(self) -> None:
        if self.mpcActive:
            self.mpcOff()
        else:
            self.mpcOn()

    # initializes the PID mode on PID ON and switch of mode
    def mpcModeInit(self) -> None:
        if self.aw.qmc.flagon:
            if self.aw.qmc.flagstart or len(self.aw.qmc.on_timex)<1:
                self.time_mpcON = 0
            else:
                self.time_mpcON = self.aw.qmc.on_timex[-1]
                
 # v is from [-min,max]
    def setEnergy(self, v:float) -> None:
        try:
            # if invertControl we invert min/max to max/min
            vx = float(v)
            slidernr = self.target - 1
            # we need to map the duty [0%,100%] to the [slidermin,slidermax] range
            slider_min = self.aw.eventslidermin[slidernr]
            slider_max = self.aw.eventslidermax[slidernr]
            # assumption: if self.targetRangeLimit then slider_min < self.targetMin < self.targetMax < slider_max
            heat_min = (max(self.targetMin, slider_min) if self.targetRangeLimit else slider_min)
            heat_max = (min(self.targetMax, slider_max) if self.targetRangeLimit else slider_max)
            heat = int(round(float(numpy.interp(vx,[0,100],[heat_min,heat_max]))))
            heat = self.aw.applySliderStepSize(slidernr, heat) # quantify by slider step size
            self.aw.addEventSignal.emit(heat,slidernr,self.createEvents,True,self.slider_force_move)
            self.aw.qmc.slider_force_move = True
        except Exception as e: # pylint: disable=broad-except
            _log.exception(e)

    # the internal software PID should be configured on ON, but not be activated yet to warm it up
    def confSoftwareMPC(self) -> None:
        if self.externalPIDControl() not in [1, 2, 4] and not(self.aw.qmc.device == 19 and self.aw.qmc.PIDbuttonflag) and self.aw.qmc.Controlbuttonflag:
            # software PID
            self.aw.qmc.mpc.setMPC(self.mpcSource,self.modelPath) 
            self.aw.qmc.mpc.setLimits(self.targetMin,self.targetMax) 
            self.aw.qmc.mpc.setDutySteps(self.dutySteps)
            self.aw.qmc.mpc.setDutyMin(self.dutyMin)
            self.aw.qmc.mpc.setDutyMax(self.dutyMax)
            self.aw.qmc.mpc.setControl(self.setEnergy)

    def mpcOn(self) -> None:
        if self.aw.qmc.flagon:
            if self.aw.pidcontrol.pidActive:
                self.aw.pidcontrol.pidOff()
                self.aw.pidcontrol.pidActive = False
            if not self.mpcActive:
                self.aw.sendmessage(QApplication.translate('StatusBar','MPC ON'))
            self.mpcModeInit()

            self.slider_force_move = True

            if self.aw.qmc.Controlbuttonflag:
                self.aw.qmc.mpc.setMPC(self.modelPath,self.mpcDRYTemp,self.mpcFCTemp,self.mpcDROPTemp,self.mpcDRYTime,self.mpcFCTime,self.mpcDROPTime)
                self.aw.qmc.mpc.setLimits(self.targetMin,self.targetMax)
                self.aw.qmc.mpc.setDutySteps(self.dutySteps)
                self.aw.qmc.mpc.setDutyMin(self.dutyMin)
                self.aw.qmc.mpc.setDutyMax(self.dutyMax)
                self.aw.qmc.mpc.setControl(self.setEnergy)
                self.mpcActive = True
                self.aw.qmc.mpc.on()
                self.aw.buttonCONTROL.setStyleSheet(self.aw.pushbuttonstyles['PIDactive'])

    def mpcOff(self) -> None:
        if self.mpcActive:
            self.aw.sendmessage(QApplication.translate('StatusBar','MPC OFF'))
        self.aw.setTimerColor('timer')
        if self.aw.qmc.flagon and not self.aw.qmc.flagstart:
            self.aw.qmc.setLCDtime(0)
        if self.aw.qmc.Controlbuttonflag:
            # software PID
            self.aw.qmc.mpc.setControl(lambda _: None)
            self.mpcActive = False
            self.aw.qmc.mpc.off()
            if not self.aw.HottopControlActive:
                self.aw.buttonCONTROL.setStyleSheet(self.aw.pushbuttonstyles['PID'])

    def setDutySteps(self, dutySteps:int) -> None:
        if self.aw.qmc.Controlbuttonflag and not self.externalPIDControl():
            self.aw.qmc.mpc.setDutySteps(dutySteps)

    # just store the mpc configuration TODO: for now pass, later when we start implementing mpc we will have things to set here
    def setMPC(self, DRYTemp:float, FCTemp:float, DROPTemp:float, DRYTime:float, FCTime:float, DROPTime:float, 
               modelPath:str, source:Optional[int] = None) -> None:
        self.mpcDRYTemp = DRYTemp
        self.mpcFCTemp = FCTemp
        self.mpcDROPTemp = DROPTemp
        self.mpcDRYTime = DRYTime
        self.mpcFCTime = FCTime
        self.mpcDROPTime = DROPTime
        self.modelPath = modelPath
        if source is not None:
            self.mpcSource = source

    # send conf to connected MPC
    def confMPC(self, DRYTemp:float, FCTemp:float, DROPTemp:float, DRYTime:float, FCTime:float, DROPTime:float, 
               modelPath:str, source:Optional[int] = None) -> None:
        if self.aw.qmc.Controlbuttonflag: # in all other cases if the "Control" flag is ticked
            self.aw.qmc.mpc.setMPC( DRYTemp, FCTemp, DROPTemp, DRYTime, FCTime, DROPTime, modelPath)    
            self.mpcDRYTemp = DRYTemp
            self.mpcFCTemp = FCTemp
            self.mpcDROPTemp = DROPTemp
            self.mpcDRYTime = DRYTime
            self.mpcFCTime = FCTime
            self.mpcDROPTime = DROPTime
            self.modelPath = modelPath

            self.aw.qmc.mpc.setLimits(0,100) 
            if source is not None and source>0:
                self.mpcSource = source
            self.aw.sendmessage(QApplication.translate('Message','mpc updated'))

    def setModelPath(self, path:str) -> None:
        if path is not None:
            self.modelPath = path


        
