import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from typing import Final, List, Optional, Callable
#import for keras model, load model
from keras.models import load_model,Model
import onnxruntime as rt
from bayes_opt import BayesianOptimization


from threading import Thread

try:
    from PyQt6.QtCore import QSemaphore # @Reimport @UnresolvedImport @UnusedImport
except ImportError:
    from PyQt5.QtCore import QSemaphore # type: ignore # @Reimport @UnresolvedImport @UnusedImport

_log: Final[logging.Logger] = logging.getLogger(__name__)

# expects a function control that takes a value from [<outMin>,<outMax>] to control the heater as called on each update()
class MPC:
    __slots__ = ['mpcSemaphore','outMin','outMax','dutySteps','dutyMin','dutyMax','control','lastOutput','iterations_since_duty','force_duty',
                 'active','target','model','modelPath', 'yellowingTemp','FCtemp','DROPtemp','yellowingTime','FCtime','DROPtime',
                 'step','input_horizon','output_horizon','xScaler','yScaler', 'min_input_length','modelLoaded','optimizer','optimization_thread',
                    'optimization_thread_running','future_optimal_burner_settings','burner_options','control_length','control_changes_left',
                    'current_stage','target_temp','target_time'] 

    def __init__(self, control:Callable[[float], None]=lambda _: None, modelPath:str = "",
                 yellowingTemp:float = 160, FCtemp:float = 198, DROPtemp:float = 211,yellowingTime:float = 300,
                FCtime:float = 555,DROPtime:float = 700) -> None:
        self.mpcSemaphore:QSemaphore = QSemaphore(1)
        self.active:bool = False
        self.modelLoaded:bool = False

        self.outMin:int = 0 # minimum output value
        self.outMax:int = 100 # maximum output value
        self.dutySteps:int = 5
        self.dutyMin:int = 5
        self.dutyMax:int = 45
        self.control:Callable[[float], None] = control
        self.iterations_since_duty:int = 999
        self.lastOutput:Optional[float] = None
        self.force_duty:int = 5 # at least every n update cycles a new duty value is send, even if its duplicating a previous duty (within the duty step)

        #MPC - LSTM Variables
        #
        #
        #
        self.modelPath:Optional[str] = modelPath # used to reinitialize the model to use for prediction     
        self.model = None
        self.xScaler:Optional[MinMaxScaler] = None # used to scale input data for prediction
        self.yScaler:Optional[MinMaxScaler] = None # used to unscale output data after prediction as outputs are scaled to [0, 1] None
        #because this LSTM model is autoregressive (uses outputs as inputs in a prediction loop)

        self.yellowingTemp:float = yellowingTemp # goal yellowing temperature to reach in °C
        self.FCtemp:float = FCtemp
        self.DROPtemp:float = DROPtemp
        self.yellowingTime:float = yellowingTime # time to reach yellowing temperature in seconds
        self.FCtime:float = FCtime
        self.DROPtime:float = DROPtime
        self.current_stage:str = "DRY"
        self.target_temp:float = 0. # target temperature to reach
        self.target_time:float = 0. # target time to reach
        
        self.step:float = 10. # time step in seconds
        self.input_horizon:int = int(2*60/self.step) # number of input horizon steps (hardcoded to 6 minutes)
        self.output_horizon:int = int(2*60/self.step) # number of output horizon steps hardcoded to 6 minutes, but will change based on next milestone (DRY, FC, DROP)
        self.min_input_length:int = int(20/self.step)
        
        self.optimizer:Optional[BayesianOptimizer] = None # optimizer to find optimal burner settings
        self.optimization_thread:Optional[Thread] = None # thread to run the optimization loop
        self.optimization_thread_running:bool = False # flag to check if the optimization thread is running
        self.future_optimal_burner_settings:Optional[List[float]] = None # optimal burner settings to be used in the next update cycle
        self.control_length:int = 3 # number of control values we use out of the optimal burner settings
        self.control_changes_left:int = 0 # number of control changes left to reach the optimal burner settings
        #-----------------------------------------------------------------------------------

    ### External API guarded by semaphore

    def on(self) -> None:
        try:
            self.mpcSemaphore.acquire(1)
            self.active = True
            self.burner_options = list(range(self.dutyMin,self.dutyMax,self.dutySteps))
            if self.modelLoaded:
                # self.optimizer = GeneticAlgorithm(model=self.model,xScaler=self.xScaler,yScaler=self.yScaler,pop_size=4, 
                #                   num_timesteps=self.output_horizon, burner_options=self.burner_options, desired_accuracy=1.0)
                self.optimizer = BayesianOptimizer(model=self.model,xScaler=self.xScaler,yScaler=self.yScaler,
                                                   num_timesteps=self.output_horizon, burner_options=self.burner_options)
            self.iterations_since_duty = 999
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
    def current_target_time(self) -> float:
        if self.current_stage == "DRY":
            return self.yellowingTime
        elif self.current_stage == "FC":
            return self.FCtime
        elif self.current_stage == "DROP":
            return self.DROPtime
        
        return 0. # return 0 if the current stage is not set
    
    def current_target_temp(self) -> float:
        if self.current_stage == "DRY":
            return self.yellowingTemp
        elif self.current_stage == "FC":
            return self.FCtemp
        elif self.current_stage == "DROP":
            return self.DROPtemp
        
        return 0.

    # update control value (the mpc loop is running even if MPC is inactive, just the control function is only called if active)
    def update(self,et,bt,timex,eventsValue,eventsIdx,delay,charge_idx) -> None:
        try: 
            self.mpcSemaphore.acquire(1)
            if self.active and charge_idx != -1 and \
                len(bt[charge_idx:]) / self.step >= self.min_input_length and \
                    self.iterations_since_duty >= self.step: #and timex[-1] - timex[charge_idx] >= 60: 
                
                #if we reach target temp we change target, but only after turning point
                if bt[-1] >= self.target_temp:
                    if self.current_stage == "DRY":
                        self.current_stage = "FC"
                        self.target_temp = self.FCtemp
                        self.target_time = self.FCtime
                    elif self.current_stage == "FC":
                        self.current_stage = "DROP"
                        self.target_temp = self.DROPtemp
                        self.target_time = self.DROPtime
                    elif self.current_stage == "DROP":
                        self.active = False

                self.output_horizon = (self.target_time - timex[-1] - timex[charge_idx]) // self.step
                
                # get result of previous optimization
                output = None
                if self.future_optimal_burner_settings is not None and self.control_changes_left > 0:
                    output = self.future_optimal_burner_settings[self.control_length-self.control_changes_left]
                    self.control_changes_left -= 1

                #start new optimization
                if self.control_changes_left == 0:
                    predictionData = self.preprocessData(et,bt,eventsValue,eventsIdx,delay,charge_idx) # preprocess data for prediction
                    if self.optimization_thread_running:
                        self.stop_optimization() # stop expired optimization thread before starting a new one
                        self.iterations_since_duty = 0 # reset iterations since duty
                    self.start_optimization(predictionData)

                if output is not None: # if optimization is successful, use the result

                    # clamp output to [outMin,outMax]
                    if output > self.outMax:
                        output = self.outMax
                    elif output < self.outMin:
                        output = self.outMin

                    int_output = int(round(min(self.dutyMax,max(self.dutyMin,output))))
                    self.control(int_output)
                    self.iterations_since_duty = 0
            self.iterations_since_duty += 1

        except Exception as e: # pylint: disable=broad-except
            _log.exception(e)
        finally:
            if self.mpcSemaphore.available() < 1:
                self.mpcSemaphore.release(1)

    def start_optimization(self,predictionData) -> None:
        self.optimization_thread = Thread(target=self.findOptimalDuty, args=(predictionData,))
        self.optimization_thread.start()
        self.optimization_thread_running = True

    def stop_optimization(self) -> None:
        self.optimizer.stop()
        while self.optimization_thread.is_alive():
            pass 
        self.optimization_thread_running = False

    def preprocessData(self,et,bt,eventsValue,eventsIdx,delay,charge_idx) -> List[List[float]]:
        try:
            et = et[charge_idx:]
            bt = bt[charge_idx:]
            i = 0
            while eventsIdx[i] < charge_idx:
                i += 1
            eventsIdx = eventsIdx[i:]
            eventsValue = eventsValue[i:] 
            eventsIdx = [x - charge_idx for x in eventsIdx]

            burner = [0]*len(et)
            i = 0
            last_v = eventsValue[0]
            for idx,v in zip(eventsIdx,eventsValue):
                while i < len(burner) and i <= idx:
                    burner[i] = last_v
                    i += 1
                last_v = v

            while i < len(burner):
                burner[i] = burner[i-1]
                i += 1
            
            roast = np.array([bt,et,burner]).T
             # scale input data for prediction to [0,1] return roast
            roast = self.xScaler.transform(roast)
            roast = roast[::int(self.step/delay)]
            if roast.shape[0] >= self.input_horizon:
                roast = roast[roast.shape[0]-self.input_horizon:]
            else:
                roast = np.pad(roast, ((self.input_horizon - roast.shape[0], 0), (0, 0)), mode='constant')

            return np.expand_dims(roast,0)
        except Exception as e:
            _log.exception(f"Could not preprocess data for prediction. Error: {e}") 

        return None
    
    def findOptimalDuty(self,predictionData) -> None:
        
        self.future_optimal_burner_settings = None
        
        if self.modelLoaded and predictionData is not None:
            # #perform optimization over the space of duty values over time to reach the target temperature in the right time
            self.future_optimal_burner_settings = self.optimizer.run(predictionData, self.target_temp)
            if self.future_optimal_burner_settings is not None:
                self.control_changes_left = self.control_length

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
            self.burner_options = list(range(self.dutyMin,self.dutyMax,self.dutySteps))
            if self.modelLoaded:
                # self.optimizer = GeneticAlgorithm(model=self.model,xScaler=self.xScaler,yScaler=self.yScaler,pop_size=100, 
                #                   num_timesteps=self.output_horizon, burner_options=self.burner_options, desired_accuracy=1.0)
                self.optimizer = BayesianOptimizer(model=self.model,xScaler=self.xScaler,yScaler=self.yScaler,
                                                    num_timesteps=self.output_horizon, burner_options=self.burner_options)
            self.iterations_since_duty = 999
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
            xscaler_path:str = self.modelPath.replace('.onnx','.joblib').replace('model','x_scaler')
            yscaler_path:str = self.modelPath.replace('.onnx','.joblib').replace('model','y_scaler')
            try:
                # self.model = load_model(self.modelPath)
                self.model = rt.InferenceSession(self.modelPath) # load the model to use for prediction

                self.xScaler = joblib.load(xscaler_path)
                self.yScaler = joblib.load(yscaler_path) 
                self.modelLoaded = True
            except Exception as e:
                _log.exception(f'Failed to load model,{self.modelPath}. Error: {e}')      
                self.modelLoaded = False # failed to load model                  
        else:
            _log.info('Incorrect model path')                                      

class BayesianOptimizer:
    def __init__(self,model,xScaler,yScaler, num_timesteps, burner_options):
            self.model = model
            self.model_input_name = self.model.get_inputs()[0].name


            self.xScaler = xScaler
            self.yScaler = yScaler
            self.num_timesteps = num_timesteps
            self.burner_options = burner_options
            self.scaled_burner_options = np.zeros((len(self.burner_options), 3))
            self.scaled_burner_options[:, 2] = (np.array(self.burner_options)/10 + 1)
            self.scaled_burner_options = self.xScaler.transform(self.scaled_burner_options)
            self.scaled_burner_options = self.scaled_burner_options[:,2]

            self.stop_requested = False # flag to stop the optimization loop

    def stop(self) -> None:
            self.stop_requested = True

    def predict_system(self, current_state, burner_settings):
        current_batch = current_state
        predictions = []
        for i in range(self.num_timesteps):
            # current_pred = self.model.predict(current_batch)[0]
            #convert current_batch to float32
            current_batch  = current_batch.astype(np.float32)
            current_pred = self.model.run(None, {self.model_input_name: current_batch})[0][0]
            predictions.append(current_pred)
            current_batch = np.roll(current_batch, -1, axis=1)
            current_batch[0,-1,:]=np.append(current_pred, [burner_settings[i]])

        return predictions

    def objective_function(self, predicted_state, target_bt, individual,last_burner) -> float:
        control_sequence = np.zeros(self.num_timesteps+1) 
        control_sequence[0] = last_burner
        control_sequence[1:] = individual
        unscaled_predictions = self.yScaler.inverse_transform(predicted_state) 
        bt = unscaled_predictions[-1,0] # last time step of predicted state is the target bt
        # if expanded_individual is changing from i to i+1 then change_penalty factor increases by 1
        N_changes = sum([1 for i in range(1, len(control_sequence)) if control_sequence[i] != control_sequence[i-1]])/len(control_sequence)
        S_changes = sum([abs(control_sequence[i] - control_sequence[i-1]) for i in range(1, len(control_sequence))])/(len(control_sequence)*(self.scaled_burner_options[-1]-self.scaled_burner_options[0]))
        E_total = abs((target_bt - bt-70)/213)

        w1 = 2.0  # Increased weight for N_changes
        w2 = 1.0   # Weight for S_changes
        w3 = 1.0   # Weight for E_total

        return w1*N_changes + w2*S_changes + w3*E_total
    
    def wrapper_evaluate_roast(self,state_dict) -> float:
 #        pbounds = {f'c{i}' : (self.burner_options[0],self.burner_options[-1]) for i in range(self.num_timesteps)}
        #we need to recieve arguments to evaluate_roast as in pbounds

        def evaluate_roast(**kwargs) -> float:
            burner_settings = [kwargs[f'c{i}'] for i in range(self.num_timesteps)]
            current_state = state_dict["CURRENT_STATE"]    
            target_bt = state_dict["TARGET_BT"]   
            unscaled_last_state = self.xScaler.inverse_transform(current_state[-1])
            last_burner =(unscaled_last_state[0,2]-1)*10 
            predicted_state = self.predict_system(current_state, burner_settings)

            return self.objective_function(predicted_state, target_bt,burner_settings,last_burner)
        
        return evaluate_roast

    
    def run(self, current_state, target_bt):
        state_dict = {
            "CURRENT_STATE" : current_state, 
            "TARGET_BT" : target_bt
        }
        wrapper_function = self.wrapper_evaluate_roast(state_dict)

        pbounds = {f'c{i}' : (self.scaled_burner_options[0],self.scaled_burner_options[-1]) for i in range(self.num_timesteps)}
        optimizer = BayesianOptimization(f=wrapper_function, pbounds=pbounds, verbose=2, random_state=1)
        optimizer.maximize(init_points=10, n_iter=15)
        best_setting = optimizer.max['params']
        best_setting = [best_setting[f'c{i}'] for i in range(self.num_timesteps)]

        return self.unscale_setting(best_setting)
    
    def unscale_setting(self, burner):
        unscaled_burner = np.zeros((len(burner), 3))
        unscaled_burner[:, 2] = np.array(burner)
        unscaled_burner = self.xScaler.inverse_transform(unscaled_burner)[:,2]
        return (np.array(unscaled_burner)-1)*10

class GeneticAlgorithm:
    def __init__(self,model,xScaler,yScaler, pop_size, num_timesteps, burner_options, desired_accuracy, num_generations=25, 
                 num_parents=25, mutation_rate=0.01):
        self.model = model
        self.model_input_name = self.model.get_inputs()[0].name


        self.xScaler = xScaler
        self.yScaler = yScaler
        self.pop_size = pop_size
        self.num_timesteps = num_timesteps
        self.burner_options = burner_options
        self.scaled_burner_options = np.zeros((len(self.burner_options), 3))
        self.scaled_burner_options[:, 2] = (np.array(self.burner_options)/10 + 1)
        self.scaled_burner_options = self.xScaler.transform(self.scaled_burner_options)
        self.scaled_burner_options = self.scaled_burner_options[:,2]
        self.desired_accuracy = desired_accuracy
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.mutation_rate = mutation_rate
        self.population = None
        self.stop_requested = False # flag to stop the optimization loop

    def stop(self) -> None:
        self.stop_requested = True

    def init(self) -> None:
        self.population = self.initialize_population()
        self.stop_requested = False

    def initialize_population(self):
        return [[random.choice(self.scaled_burner_options) for _ in range(self.num_timesteps)] for _ in range(self.pop_size)]

    # def evaluate_population(self, current_state, target_bt):
    #     return [self.objective_function(self.predict_system(current_state, individual), target_bt) for individual in self.population]
    
    def evaluate_population(self, current_state, target_bt): 
        results = []
        unscaled_last_state = self.xScaler.inverse_transform(current_state[-1])
        last_burner =(unscaled_last_state[0,2]-1)*10 

        for individual in self.population:
            result = self.evaluate_individual(individual, current_state, target_bt,last_burner)
            results.append(result)
            if self.stop_requested:
                return None   
        return results
    
    def evaluate_individual(self, individual, current_state, target_bt,last_burner):
        predicted_state = self.predict_system(current_state, individual)
        if self.stop_requested:
            return None   
        return self.objective_function(predicted_state, target_bt,individual,last_burner)
    
    def predict_system(self, current_state, burner_settings):
        current_batch = current_state
        predictions = []
        for i in range(self.num_timesteps):
            if self.stop_requested:
                return None   
            # current_pred = self.model.predict(current_batch)[0]
            #convert current_batch to float32
            current_batch  = current_batch.astype(np.float32)
            current_pred = self.model.run(None, {self.model_input_name: current_batch})[0][0]
            predictions.append(current_pred)
            current_batch = np.roll(current_batch, -1, axis=1)
            current_batch[0,-1,:]=np.append(current_pred, [burner_settings[i]])

        return predictions

    def objective_function(self, predicted_state, target_bt, individual,last_burner) -> float:
        control_sequence = np.zeros(self.num_timesteps+1) 
        control_sequence[0] = last_burner
        control_sequence[1:] = individual
        unscaled_predictions = self.yScaler.inverse_transform(predicted_state) 
        bt = unscaled_predictions[-1,0] # last time step of predicted state is the target bt
        # if expanded_individual is changing from i to i+1 then change_penalty factor increases by 1
        N_changes = sum([1 for i in range(1, len(control_sequence)) if control_sequence[i] != control_sequence[i-1]])/len(control_sequence)
        S_changes = sum([abs(control_sequence[i] - control_sequence[i-1]) for i in range(1, len(control_sequence))])/(len(control_sequence)*(self.scaled_burner_options[-1]-self.scaled_burner_options[0]))
        E_total = abs((target_bt - bt-70)/213)

        w1 = 2.0  # Increased weight for N_changes
        w2 = 1.0   # Weight for S_changes
        w3 = 1.0   # Weight for E_total

        return w1*N_changes + w2*S_changes + w3*E_total
    
    
    def select_parents(self, fitness_scores):
        return random.choices(self.population, weights=[1/f for f in fitness_scores], k=self.num_parents)


    def crossover(self, parents, offspring_size):
        offspring = []
        for _ in range(offspring_size):
            parent1, parent2 = random.sample(parents, 2)
            crossover_point = random.randint(1, len(parent1) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        for individual in offspring:
            if random.random() < self.mutation_rate:
                mutate_point = random.randint(0, len(individual) - 1)
                individual[mutate_point] = random.choice(self.burner_options)
        return offspring

    def run(self, current_state, target_bt):
        for generation in range(self.num_generations):
            fitness_scores = self.evaluate_population(current_state, target_bt)
            if self.stop_requested:
                return None
            best_fitness = min(fitness_scores)

            # Check the stopping condition
            if best_fitness <= self.desired_accuracy:
                break

            parents = self.select_parents(fitness_scores)
            offspring_size = self.pop_size - len(parents)
            offspring = self.crossover(parents, offspring_size)
            offspring = self.mutate(offspring)
            self.population = parents + offspring


        best_individual_index = fitness_scores.index(best_fitness)
        best_individual = self.population[best_individual_index]

        return self.unscale_individual(best_individual)
    
    def unscale_individual(self, individual):
        unscaled_individual = np.zeros((len(individual), 3))
        unscaled_individual[:, 2] = np.array(individual)
        unscaled_individual = self.xScaler.inverse_transform(unscaled_individual)[:,2]
        return (np.array(unscaled_individual)-1)*10