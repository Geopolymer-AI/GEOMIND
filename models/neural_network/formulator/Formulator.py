#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:55:00 2024

@author: roussp14
"""

from keras import layers as Layers
from keras import Input, Model
from keras.optimizers import RMSprop
from ..NeuralNetworkModel import NeuralNetworkModel
from ..feasibility_controller.molar import getMoleRatios
import tensorflow as Tensorflow
import numpy as Numpy
from pydash import map_

TRAINING_EPOCHS = 310
BATCH_SIZE = 16

class Formulator(NeuralNetworkModel):
    def __init__(self, nb_properties, nb_precursors, latent_dim = 8, should_load = False, is_vae = True, is_student = False, name = "Formulator"):
        if not isinstance(is_vae, bool):
            raise TypeError( "is_vae must be a bool" )
        
        if not isinstance(is_student, bool):
            raise TypeError( "is_student must be a bool" )
            
        self._is_vae = is_vae
        self._is_student = is_student
        super().__init__(nb_properties, nb_precursors, latent_dim, should_load, name)
    
    def _createModel(self):
        properties = Input( shape = (self.getNbProperties()) )
        latent_dim = self.getLatentDim()
        
        x = properties
        
        def createModelOutput( x ):
            precursors = Layers.Dense( self.getNbPrecursors(), activation = "sigmoid", name = "precursors" )( x )
            sums = Layers.Lambda( lambda x : Tensorflow.math.reduce_sum( x, axis = -1 ), name = "sums" )( precursors )
            mole_ratios = Layers.Lambda( lambda x: getMoleRatios( x ) )( precursors )
            siM_sol = Layers.Lambda( lambda x: Tensorflow.gather( x, 0 ), name = "siM_sol" )( mole_ratios )
            siAl = Layers.Lambda( lambda x: Tensorflow.gather( x, 1 ), name = "siAl" )( mole_ratios )
            solidLiquid = Layers.Lambda( lambda x: Tensorflow.gather( x, 2 ), name = "solidLiquid" )( mole_ratios )
            
            formulator = Model( properties, [precursors, sums, siM_sol, siAl, solidLiquid] )
            
            return formulator
        
        if self.isVAE() == False:
            x = NeuralNetworkModel.createMLP( x, output_dim = 32, dropout = 0 )
            formulator = createModelOutput( x )
            
            return formulator
        
        #Encoder
        x = Layers.Dense( 128, activation = Layers.PReLU() )( x )
        x = Layers.Dense( 128, activation = Layers.PReLU() )( x )
        x = Layers.Dense( 64, activation = Layers.PReLU() )( x )
        x = Layers.Dense( 64, activation = Layers.PReLU() )( x )
        
        #Latent space
        z_mean = Layers.Dense( latent_dim )( x )
        z_log_var = Layers.Dense( latent_dim )( x )
        if self.isStudent():
            z = Layers.Lambda( NeuralNetworkModel.studentSampling )( [z_mean, z_log_var, latent_dim, 3.] )
        else:
            z = Layers.Lambda( NeuralNetworkModel.sampling )( [z_mean, z_log_var, latent_dim] )
        
        #Decoder
        x = Layers.Dense( 64, activation = Layers.PReLU() )( z )
        x = Layers.Dense( 64, activation = Layers.PReLU() )( x )
        x = Layers.Dense( 128, activation = "relu" )( x )
        x = Layers.Dense( 128, activation = "relu" )( x )

        formulator = createModelOutput( x )
        
        return formulator

    def _compile(self):
        super()._compile()
        self._model.compile( optimizer = RMSprop( learning_rate = 0.0005 ), loss = "mae" )
    
    def isVAE(self):
        return self._is_vae
    
    def isStudent(self):
        return self._is_student
    
    def evaluate(self, X, y):
        sums = Numpy.ones(len(y))
        siM_sol, siAl, solidLiquid = Numpy.swapaxes(map_( y, lambda x : getMoleRatios( x ) ), 0, 1)
        y = [y, sums, siM_sol, siAl, solidLiquid]
        return super().evaluate( X, y )

    def fit(self, X, y, epochs = TRAINING_EPOCHS, batch_size = BATCH_SIZE, verbose = 1, validation_data = None, should_use_early_stopping = False ):
        if validation_data is not None:
            if not (isinstance(validation_data, tuple) and len(validation_data) == 2):
                raise TypeError( "validation_data must be a tuple length of 2" )
        
        sums = Numpy.ones(len(y))
        siM_sol, siAl, solidLiquid = Numpy.swapaxes(map_( y, lambda x : getMoleRatios( x ) ), 0, 1)
        y = [y, sums, siM_sol, siAl, solidLiquid]

        if validation_data is not None:
            val_y = validation_data[1]
            sums = Numpy.ones(len(val_y))
            siM_sol, siAl, solidLiquid = Numpy.swapaxes(map_( val_y, lambda x : getMoleRatios( x ) ), 0, 1)
            val_y = [val_y, sums, siM_sol, siAl, solidLiquid]
            validation_data = (validation_data[0], val_y)

        return super().fit(X, y, epochs, batch_size, verbose, validation_data, should_use_early_stopping)