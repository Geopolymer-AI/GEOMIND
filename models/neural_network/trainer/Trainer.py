#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:52:07 2024

@author: roussp14
"""

from keras import layers as Layers
from keras import Input, Model
from keras.optimizers import RMSprop
from ..NeuralNetworkModel import NeuralNetworkModel
from ..formulator import Formulator
from ..simulator import Simulator
from ..feasibility_controller.molar import getMoleRatios
import numpy as Numpy
from pydash import map_

TRAINING_EPOCHS = 310
BATCH_SIZE = 16

class Trainer(NeuralNetworkModel):
    def __init__(self, simulator, formulator, nb_properties, nb_precursors, should_load = False, is_vae = True, is_student = False, loss_weights = { "precursors" : 10, "sums" : 2, "siM_sol": 1, "siAl": 1, "solidLiquid": 1, "properties": 5 }):
        if not isinstance(loss_weights, dict):
            raise TypeError( "loss_weights must be a dict" )
            
        if not isinstance(simulator, Simulator):
            raise TypeError( "simulator must be a Simulator" )
        
        if not isinstance(formulator, Formulator):
            raise TypeError( "formulator must be a Formulator" )
        
        self._name = "Trainer"
        self._loss_weights = loss_weights
        self._simulator = simulator
        self._formulator = formulator
        super().__init__(nb_properties, nb_precursors, should_load)
    
    def _createModel(self):
        simulator = self._simulator.getModel()
        formulator = self._formulator.getModel()
        properties = Input( shape = (self.getNbProperties()) )
        
        simulator.trainable = False
        
        #Get precursors from target properties
        formulated_precursors, sums, siM_sol, siAl, solidLiquid = formulator( properties )
        #Name outputs
        formulated_precursors = Layers.Lambda( lambda x : x, name = "precursors" )( formulated_precursors )
        sums = Layers.Lambda( lambda x : x, name = "sums" )( sums )
        siM_sol = Layers.Lambda( lambda x: x, name = "siM_sol" )( siM_sol )
        siAl = Layers.Lambda( lambda x: x, name = "siAl" )( siAl )
        solidLiquid = Layers.Lambda( lambda x: x, name = "solidLiquid" )( solidLiquid )
        
        #Get predicted properties from predicted precursors
        viscosity, volume_weight, compression, density  = simulator( formulated_precursors )
        #Concatenate properties
        simulated_properties = Layers.Concatenate()( [viscosity, volume_weight, compression, density] )
        simulated_properties = Layers.Lambda( lambda x: x, name = "properties" )(  simulated_properties )
        
        trainer = Model( properties, [formulated_precursors, sums, siM_sol, siAl, solidLiquid, simulated_properties] )

        return trainer

    def _compile(self):
        super()._compile()
        self._model.compile( optimizer = RMSprop( learning_rate = 0.0005 ), loss = "mae", loss_weights = self.getLossWeights() )
        
    def getLossWeights(self):
        return self._loss_weights
    
    def trainSimulator(self):
        pass

    def predict(self, X, mean = None, std = None):
        precursors = self._formulator.predict(X)
        properties = self._simulator.predict(precursors[0], mean, std)
        
        return precursors[0], properties
    
    def evaluate(self, X, y):
        sums = Numpy.ones(len(y))
        siM_sol, siAl, solidLiquid = Numpy.swapaxes(map_( y, lambda x : getMoleRatios( x ) ), 0, 1)
        y = [y, sums, siM_sol, siAl, solidLiquid, X]
        return super().evaluate( X, y )

    def fit(self, X, y, epochs = TRAINING_EPOCHS, batch_size = BATCH_SIZE, verbose = 1, validation_data = None, should_use_early_stopping = False ):
        if validation_data is not None:
            if not (isinstance(validation_data, tuple) and len(validation_data) == 2):
                raise TypeError( "validation_data must be a tuple length of 2" )
        
        sums = Numpy.ones(len(y))
        siM_sol, siAl, solidLiquid = Numpy.swapaxes(map_( y, lambda x : getMoleRatios( x ) ), 0, 1)
        y = [y, sums, siM_sol, siAl, solidLiquid, X]

        if validation_data is not None:
            val_X = validation_data[0]
            val_y = validation_data[1]
            sums = Numpy.ones(len(val_y))
            siM_sol, siAl, solidLiquid = Numpy.swapaxes(map_( val_y, lambda x : getMoleRatios( x ) ), 0, 1)
            val_y = [val_y, sums, siM_sol, siAl, solidLiquid, val_X]
            validation_data = (val_X, val_y)

        return super().fit(X, y, epochs, batch_size, verbose, validation_data, should_use_early_stopping)