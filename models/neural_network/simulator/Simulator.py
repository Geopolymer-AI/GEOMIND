#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:31:08 2024

@author: roussp14

This file defines the Simulator class.

"""

from keras import layers as Layers
from keras import Input, Model
from tensorflow.keras.optimizers import RMSprop
from ..NeuralNetworkModel import NeuralNetworkModel
from ..feasibility_controller.molar import getMoleRatios
import tensorflow as Tensorflow
import numpy as Numpy
from GPyOpt.methods import BayesianOptimization

TRAINING_EPOCHS = 250
BATCH_SIZE = 16

class Simulator(NeuralNetworkModel):
    def __init__(self, nb_properties, nb_precursors, unfeasible_output_values, latent_dim = 8, should_load = False, should_compute_molar_ratios = True, name = "Simulator"):
        if not isinstance(should_compute_molar_ratios, bool):
            raise TypeError( "should_compute_molar_ratios must be a bool" )
        
        if not (isinstance(unfeasible_output_values, list) and len(unfeasible_output_values) == nb_properties):
            raise TypeError( "unfeasible_output_values must be a list length of nb_properties" )
            
        self._unfeasible_output_values = unfeasible_output_values
        self._should_compute_molar_ratios = should_compute_molar_ratios
        super().__init__(nb_properties, nb_precursors, latent_dim, should_load, name)
    
    def _createModel(self):
        precursors = Input( shape = (self.getNbPrecursors()) )
        latent_dim = self.getLatentDim()
    
        ratios, ratio_m1, ratio_m2, ratio_m3, ratio_m4, ratio_m5 = Tensorflow.split( precursors, [6, 1, 1, 1, 1, 1], -1 )
        std_properties_min = Tensorflow.convert_to_tensor(self._unfeasible_output_values, dtype = Tensorflow.float32)
        
        #Calculate molar ratios from precursors
        mole_ratios = Layers.Lambda( lambda x: getMoleRatios( x ) )( precursors )
        siM_sol = Layers.Lambda( lambda x: Tensorflow.gather( x, 0 ) )( mole_ratios )
        siAl = Layers.Lambda( lambda x: Tensorflow.gather( x, 1 ) )( mole_ratios )
        solidLiquid = Layers.Lambda( lambda x: Tensorflow.gather( x, 2 ) )( mole_ratios )
        
        x = Layers.Concatenate()([precursors, siM_sol, siAl, solidLiquid])
    
        #Encoder
        x = Layers.Dense( 128, activation = Layers.PReLU() )( x )
        x = Layers.Dense( 128, activation = Layers.PReLU() )( x )
        x = Layers.Dense( 64, activation = Layers.PReLU() )( x )
        x = Layers.Dense( 64, activation = Layers.PReLU() )( x )
        
        #Latent space
        z_mean = Layers.Dense( latent_dim )( x )
        z_log_var = Layers.Dense( latent_dim )( x )
        z = Layers.Lambda( NeuralNetworkModel.sampling )( [z_mean, z_log_var, latent_dim] )
        
        #Decoder
        x = Layers.Dense( 64, activation = Layers.PReLU() )( z )
        x = Layers.Dense( 64, activation = Layers.PReLU() )( x )
        x = Layers.Dense( 128, activation = "relu" )( x )
        x = Layers.Dense( 128, activation = "relu" )( x )
    
        viscosity = Layers.Dense( 3 )( x )
        volume_weight = Layers.Dense( 1 )( x )
        compression = Layers.Dense( 1 )( x )
        density = Layers.Dense( 1 )( x )
        
        if self.shouldComputeMolarRatios() == False:
            viscosity = Layers.Lambda( lambda x: x, name = "viscosity" )( viscosity )
            volume_weight = Layers.Lambda( lambda x: x, name = "volume_weight" )( volume_weight )
            compression = Layers.Lambda( lambda x: x, name = "compression" )( compression )
            density = Layers.Lambda( lambda x: x, name = "density" )( density )
            
            simulator = Model( precursors, [viscosity, volume_weight, compression, density] )

            return simulator
        
        properties = Layers.Concatenate()( [viscosity, volume_weight, compression, density] )
        
        #Control the molar ratios limits
        properties = Tensorflow.where( Tensorflow.logical_or( siM_sol >= 1.1, siM_sol <= 0.2 ), std_properties_min, properties )
        properties = Tensorflow.where( Tensorflow.logical_or( siAl >= 3, siAl <= 1.3 ), std_properties_min, properties )
        
        properties_m1_m5 = Tensorflow.where( Tensorflow.logical_or( solidLiquid >= 2, solidLiquid <= 0.44 ), std_properties_min, properties )
        properties_m2_m4 = Tensorflow.where( Tensorflow.logical_or( solidLiquid >= 0.78, solidLiquid <= 0.44 ), std_properties_min, properties )
        properties_m3 = Tensorflow.where( Tensorflow.logical_or( solidLiquid >= 0.85, solidLiquid <= 0.44 ), std_properties_min, properties )
        
        properties = Tensorflow.where( Tensorflow.logical_or( ratio_m1 > 0.1, ratio_m5 > 0.2 ), properties_m1_m5, Tensorflow.where( ratio_m3 > 0.2, properties_m3, properties_m2_m4 ) )
        viscosity, volume_weight, compression, density = Tensorflow.split( properties, [3, 1, 1, 1], -1 )
        
        viscosity = Layers.Lambda( lambda x: x, name = "viscosity" )( viscosity )
        volume_weight = Layers.Lambda( lambda x: x, name = "volume_weight" )( volume_weight )
        compression = Layers.Lambda( lambda x: x, name = "compression" )( compression )
        density = Layers.Lambda( lambda x: x, name = "density" )( density )
        
        simulator = Model( precursors, [viscosity, volume_weight, compression, density] )

        return simulator

    def _compile(self):
        super()._compile()
        self._model.compile( optimizer = RMSprop( learning_rate = 0.0005 ), loss = { "viscosity": "mae", "volume_weight": "mae", "compression": "mae", "density": "mae" } )
    
    def shouldComputeMolarRatios(self):
        return self._should_compute_molar_ratios
    
    def predict(self, X, mean = None, std = None ):
        if not (std is None or ((isinstance(std, list) or isinstance(std, Numpy.ndarray)) and len(std) == self.getNbProperties())):
            raise TypeError( "std must be None or a list length of nb_properties" )
        
        if not (mean is None or ((isinstance(mean, list) or isinstance(mean, Numpy.ndarray)) and len(mean) == self.getNbProperties())):
            raise TypeError( "mean must be None or a list length of nb_properties" )
            
        v, mv, c, d = super().predict(X)
        properties = Numpy.concatenate([v, mv, c, d], axis = -1)
        if not (std is None or mean is None):
            properties = properties * std + mean
        properties = Numpy.where( properties < 0, -1, properties )
        
        return properties
    
    #This function predicts precursors with Bayesian Optimization from one sample of target properties
    def predictPrecursors(self, sample_y, mean = None, std = None, max_iter = 200, should_plot_convergence = True):
        if not isinstance(max_iter, int):
            raise TypeError( "max_iter must be an int" )
        
        if not isinstance(should_plot_convergence, bool):
            raise TypeError( "should_plot_convergence must be a bool" )
            
        def costFunction( model, parameters, target_output ):
            output = model.predict( Numpy.array([parameters]) )
            output = Numpy.concatenate( output, axis = -1 )
            
            ratios_cost = Numpy.abs( 1 - Numpy.sum(parameters) )
            error = Numpy.mean( Numpy.abs( target_output - output ) )
            
            return ratios_cost + error

        def optimize( parameters ):
            parameters = parameters[0]
            cost = costFunction( self._model, parameters, sample_y )
            
            print(cost)
            
            return cost
        
        precursors = [{'name': 's1', 'type': 'continuous', 'domain': (0, 0.8)},
                      {'name': 's3', 'type': 'continuous', 'domain': (0, 0.8)},
                      {'name': 'sna', 'type': 'continuous', 'domain': (0, 0.8)},
                      {'name': 's3p', 'type': 'continuous', 'domain': (0, 0.8)},
                      {'name': 'koh', 'type': 'continuous', 'domain': (0, 0.15)},
                      {'name': 'naoh', 'type': 'continuous', 'domain': (0, 0.15)},
                      {'name': 'm1', 'type': 'continuous', 'domain': (0, 0.8)},
                      {'name': 'm2', 'type': 'continuous', 'domain': (0, 0.8)},
                      {'name': 'm3', 'type': 'continuous', 'domain': (0, 0.8)},
                      {'name': 'm4', 'type': 'continuous', 'domain': (0, 0.8)},
                      {'name': 'm5', 'type': 'continuous', 'domain': (0, 0.8)}]
            
        optimizer = BayesianOptimization( optimize, domain = precursors )
        optimizer.run_optimization( max_iter = max_iter )
        if should_plot_convergence:
            optimizer.plot_convergence()

        best_precursors = optimizer.x_opt
        best_properties = self.predict(Numpy.array([best_precursors]), mean, std)
        best_cost = costFunction(self._model, best_precursors, sample_y)
        
        return best_precursors, best_properties, best_cost
    
    def evaluate(self, X, y):     
        return super().evaluate( X, Tensorflow.split(y, [3,1,1,1], -1) )

    def fit(self, X, y, epochs = TRAINING_EPOCHS, batch_size = BATCH_SIZE, verbose = 1, validation_data = None, should_use_early_stopping = False ):
        if validation_data is not None:
            if not (isinstance(validation_data, tuple) and len(validation_data) == 2):
                raise TypeError( "validation_data must be a tuple length of 2" )
        
        y = Tensorflow.split(y, [3,1,1,1], -1)

        if validation_data is not None:
            validation_data = (validation_data[0], Tensorflow.split(validation_data[1], [3,1,1,1], -1))

        return super().fit(X, y, epochs, batch_size, verbose, validation_data, should_use_early_stopping)