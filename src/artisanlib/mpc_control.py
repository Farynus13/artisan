
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
    __slots__ = ['aw','mpcActive','mpcYellowing','mpcFC','mpcDROP','dutySteps','dutyMin','dutyMax','targetRangeLimit','targetMin','targetMax','target',
'slider_force_move']

    def __init__(self, aw: 'ApplicationWindow') -> None:
        self.aw:ApplicationWindow = aw
        self.mpcActive:bool = False

        self.mpcYellowing:float = 160
        self.mpcFC:float = 198
        self.mpcDROP:float = 212

        self.dutySteps:int = 1
        self.dutyMin:int = 0
        self.dutyMax:int = 100
        self.targetRangeLimit:bool = False 
        self.targetMin:int = 0
        self.targetMax:int = 100
        self.target:int = 0

        self.slider_force_move:bool = True # if True move the slider independent of the slider position to fire slider action!


        
