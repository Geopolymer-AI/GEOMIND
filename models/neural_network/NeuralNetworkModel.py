#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:42:20 2024

@author: roussp14

This file defines the NeuralNetworkModel abstract class.

"""

from abc import ABC as AbstractClass
from abc import abstractmethod
from keras.models import load_model
from keras import backend as Backend
from keras import layers as Layers
import tensorflow as Tensorflow
from tensorflow_probability.python.distributions import StudentT
import numpy as Numpy
import os
from tensorflow.keras.callbacks import EarlyStopping

class NeuralNetworkModel(AbstractClass):
    @abstractmethod
    def __init__(self, nb_properties, nb_precursors, latent_dim = 8, should_load = False, name = "AbstractModel"):
        if not isinstance(nb_properties, int):
            raise TypeError( "nb_properties must be an int" )
            
        if not isinstance(nb_precursors, int):
            raise TypeError( "nb_precursors must be an int" )
            
        if not isinstance(latent_dim, int):
            raise TypeError( "latent_dim must be an int" )
        
        if not isinstance(should_load, bool):
            raise TypeError( "should_load must be a bool" )
        
        if not isinstance(name, str):
            raise TypeError( "name must be a str" )
        
        self._name = name
        self._latent_dim = latent_dim
        self._nb_properties = nb_properties
        self._nb_precursors = nb_precursors
        self._training_history = None
        self._model = self._loadModel() if should_load else self._createModel()
        self._compile()
    
    def _loadModel(self):
        return load_model(f"saved_models/{self._name}")
    
    @abstractmethod
    def _compile(self):
        if self._model is None:
            raise ValueError( "Create the model before compile it. Model attribute is currently None")
        
    def getNbProperties(self):
        return self._nb_properties
    
    def getNbPrecursors(self):
        return self._nb_precursors
    
    def getLatentDim(self):
        return self._latent_dim
    
    def getModel(self):
        return self._model
    
    def getTrainingHistory(self):
        return self._training_history
    
    def summarize(self):
        self._model.summary( print_fn = lambda x: print(x) )
    
    def save(self):
        if not os.path.exists("saved_models"):
           os.makedirs("saved_models")
           
        self._model.save( f"saved_models/{self._name}" )
    
    @classmethod
    def sampling(cls, args):
        z_mean, z_log_var, latent_dim = args
        epsilon = Tensorflow.random.normal( shape = ( Backend.shape(z_mean)[0], latent_dim ), mean = 0, stddev = 1. )
        return z_mean + Backend.exp(z_log_var) * epsilon
    
    @classmethod
    def studentSampling(cls, args):
        z_loc, z_scale, latent_dim, df = args
        distribution = StudentT( df = df, loc = 0, scale = 1 )
        epsilon = distribution.sample(  ( Backend.shape(z_loc)[0], latent_dim ) )
        return z_loc + z_scale * epsilon
    
    @classmethod
    def createMLP(cls, x, output_dim = 32, dropout = 0.25 ):
        x = Layers.Dense( output_dim * 2, activation = Layers.PReLU() )( x )
        x = Layers.Dense( output_dim * 2, activation = Layers.PReLU() )( x )
        x = Layers.BatchNormalization()( x )
        if dropout > 0 :
            x = Layers.Dropout( dropout )( x )
        x = Layers.Dense( output_dim, activation = Layers.PReLU() )( x )
        x = Layers.Dense( output_dim, activation = Layers.PReLU() )( x )
        x = Layers.BatchNormalization()( x )
        if dropout > 0 :
            x = Layers.Dropout( dropout )( x )
        x = Layers.Dense( output_dim, activation = "relu" )( x )
        x = Layers.Dense( output_dim, activation = "relu" )( x )
        
        return x

    @classmethod
    def createVAE(cls, x, latent_dim = 4, dropout = 0.25, is_student = False, student_df = 3.):
        x = Layers.Dense( latent_dim * 8, activation = Layers.PReLU() )( x )
        x = Layers.BatchNormalization()( x )
        if dropout > 0 :
            x = Layers.Dropout( dropout )( x )
        x = Layers.Dense( latent_dim * 8, activation = Layers.PReLU() )( x )
        x = Layers.BatchNormalization()( x )
        
        x = Layers.Dense( latent_dim * 4, activation = Layers.PReLU() )( x )
        
        z_mean = Layers.Dense( latent_dim )( x )
        z_log_var = Layers.Dense( latent_dim )( x )
        if is_student:
            z = Layers.Lambda( cls.studentSampling )( [z_mean, z_log_var, latent_dim, student_df] )
        else:
            z = Layers.Lambda( cls.sampling )( [z_mean, z_log_var, latent_dim] )
        
        x = Layers.Dense( Numpy.prod( latent_dim * 4 ), activation = "relu" )( z )
        
        x = Layers.Dense( latent_dim * 8, activation = "relu" )( x )
        x = Layers.BatchNormalization()( x )
        if dropout > 0 :
            x = Layers.Dropout( dropout )( x )
        x = Layers.Dense( latent_dim * 8, activation = "relu" )( x )
        
        return x
        
    @abstractmethod
    def _createModel(self):
        pass

    def predict(self, X):
        return self._model.predict(X)
    
    def evaluate(self, X, y):
        return self._model.evaluate(X, y)

    def fit(self, X, y, epochs, batch_size, verbose = 1, validation_data = None, should_use_early_stopping = False ):
        if validation_data is not None:
            if not (isinstance(validation_data, tuple) and len(validation_data) == 2):
                raise TypeError( "validation_data must be a tuple length of 2" )

        if should_use_early_stopping:
            earlyStopping = EarlyStopping(monitor="val_loss", start_from_epoch = round(epochs*0.9), patience = round(epochs*0.1), mode = "min", restore_best_weights = True)
            
            if validation_data is None:
                self._training_history = self._model.fit(X, y, epochs = epochs * 2, batch_size = batch_size, verbose = verbose, callbacks=[earlyStopping])
            else:
                self._training_history = self._model.fit(X, y, epochs = epochs * 2, batch_size = batch_size, verbose = verbose, callbacks=[earlyStopping], validation_data = validation_data)
        
            return self._training_history

        if validation_data is None:
            self._training_history = self._model.fit(X, y, epochs = epochs, batch_size = batch_size, verbose = verbose)
        else:
            self._training_history = self._model.fit(X, y, epochs = epochs, batch_size = batch_size, verbose = verbose, validation_data = validation_data)
            
        return self._training_history