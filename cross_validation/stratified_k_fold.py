#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:03:43 2024

@author: roussp14
"""

import numpy as Numpy
from sklearn.model_selection import StratifiedKFold
from models import Simulator, Formulator, Trainer
import os
import matplotlib.pyplot as Pyplot

#This function is used in order to get training epochs, final MAE and validation predictions of cross-validation of Simulator and Formulator models
#This function also plot and save loss curves for each fold
def stratifiedKFoldCrossValidation( properties, precursors, unfeasible_mixture_properties, mean_properties, std_properties, k_fold = 8 ):
    if not os.path.exists("cross_validation/loss_curves"):
       os.makedirs("cross_validation/loss_curves")
            
    nb_properties = len(properties[0])
    nb_precursors = len(precursors[0])
    nb_samples = len(properties) // k_fold
    print(f'Number of samples by fold: {nb_samples}')
    
    simulator_k_errors = []
    simulator_k_scores = []
    trainer_k_scores = []
    simulator_epochs = []
    formulator_epochs = []
    predicted_properties = []
    val_properties = []
    
    #Cross-validation splitted by metakaolins
    y_sol, y_met = Numpy.split(precursors, [6], axis=-1)
    y_met = Numpy.flip(y_met, axis = -1)
    met_classes = Numpy.argmax(y_met, axis=-1)
    
    skf = StratifiedKFold( n_splits = k_fold )
    
    for i, (train_index, val_index) in enumerate(skf.split(precursors, met_classes)):
        print('Processing fold #', i)
        
        train_X = properties[train_index]
        train_y = precursors[train_index]
        val_X = properties[val_index]
        val_y = precursors[val_index]
        
        #Create models
        formulator = Formulator( nb_properties, nb_precursors, latent_dim = 32 )
        simulator = Simulator( nb_properties, nb_precursors, unfeasible_mixture_properties, latent_dim = 32 )
        trainer = Trainer( simulator, formulator, nb_properties, nb_precursors )
        
        #Fit Simulator
        history = simulator.fit( train_y, train_X, verbose = 0, should_use_early_stopping = True, validation_data = (val_y, val_X))
        
        #Plot loss curves
        simulator_loss = history.history["loss"]
        simulator_val_loss = history.history["val_loss"]
        epochs = range(1, len(simulator_loss) + 1)
    
        Pyplot.clf()
        Pyplot.plot( epochs, simulator_loss, "tab:blue", label = "Entraînement", linewidth = 0.8, linestyle = "dashdot" )
        Pyplot.plot( epochs, simulator_val_loss, "tab:red", label = "Validation", linewidth = 0.8, linestyle = "solid" )
        Pyplot.title( "Perte pendant l'entraînement et la validation du Simulateur" )
        Pyplot.xlabel( "Nombre d'époques" )
        Pyplot.ylabel( "MAE" )
        Pyplot.legend()
        Pyplot.savefig( f"cross_validation/loss_curves/perte_simulator_fold_{i+1}.png", dpi = 250 )  
        
        #Fit Formulator
        history = trainer.fit( train_X, train_y, verbose = 0, should_use_early_stopping = True, validation_data = (val_X, val_y))

        #Plot loss curves
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        precursors_loss = history.history["precursors_loss"]
        precursors_val_loss = history.history["val_precursors_loss"]
        properties_loss = history.history["properties_loss"]
        properties_val_loss = history.history["val_properties_loss"]
    
        epochs = range(1, len(loss) + 1)
    
        Pyplot.clf()
        Pyplot.ylim((0,0.8))
        Pyplot.xlim((8,len(epochs)))
        Pyplot.plot( epochs, Numpy.concatenate(( [precursors_loss[0]]* 9, Numpy.convolve(precursors_loss, Numpy.ones(10), 'valid') / 10 )), label = "Entraînement precurseurs", linewidth = 0.8, linestyle = "dashdot" )
        Pyplot.plot( epochs, Numpy.concatenate(( [precursors_val_loss[0]]* 9, Numpy.convolve(precursors_val_loss, Numpy.ones(10), 'valid') / 10 )), label = "Validation precurseurs", linewidth = 0.8, linestyle = "solid" )
        Pyplot.plot( epochs, Numpy.concatenate(( [properties_loss[0]]* 9, Numpy.convolve(properties_loss, Numpy.ones(10), 'valid') / 10 )), label = "Entraînement propriétés", linewidth = 0.8, linestyle = "dashdot" )
        Pyplot.plot( epochs, Numpy.concatenate(( [properties_val_loss[0]]* 9, Numpy.convolve(properties_val_loss, Numpy.ones(10), 'valid') / 10 )), label = "Validation propriétés", linewidth = 0.8, linestyle = "solid" )
        Pyplot.title( "Perte pendant l'entraînement et la validation du Formulateur" )
        Pyplot.xlabel( "Nombre d'époques" )
        Pyplot.ylabel( "MAE" )
        Pyplot.legend()
        Pyplot.savefig( f"cross_validation/loss_curves/perte_trainer_fold_{i+1}.png", dpi = 250 )
    
        #Get validation scores
        simulator_epochs.append(len(simulator_loss))  
        formulator_epochs.append(len(loss))   
        simulator_k_scores.append( simulator.evaluate( val_y, val_X ) )
        trainer_k_scores.append( trainer.evaluate( val_X, val_y ) )

        #Predict validation set
        predicted_X = simulator.predict( val_y, mean_properties, std_properties )
        val_X = val_X * std_properties + mean_properties
        val_X = Numpy.where( val_X < 0, -1, val_X )
        
        predicted_properties.append( predicted_X )
        val_properties.append( val_X )
        simulator_k_errors.append( Numpy.around( Numpy.mean(Numpy.abs(val_X - predicted_X), axis = 0), decimals = 3 ) )
        
    return { "simulator_epochs": simulator_epochs, "formulator_epochs": formulator_epochs, "simulator_k_scores": simulator_k_scores, "trainer_k_scores": trainer_k_scores, "predicted_properties": predicted_properties, "val_properties": val_properties, "simulator_k_errors": simulator_k_errors}

#This function is used in order to get loss histories of cross-validation of Simulator and Formulator alternative models, trained either with and without GEOMIND improvements
def stratifiedKFoldCrossValidationLosses( properties, precursors, unfeasible_mixture_properties, mean_properties, std_properties, k_fold = 8 ):
    nb_properties = len(properties[0])
    nb_precursors = len(precursors[0])
    nb_samples = len(properties) // k_fold
    print(f'Number of samples by fold: {nb_samples}')
    
    precursors_loss = []
    precursors_val_loss = []
    mlp_loss = []
    mlp_val_loss = []
    student_loss = []
    student_val_loss = []
    gaussian_loss = []
    gaussian_val_loss = []
    
    properties_loss = []
    properties_val_loss = []
    properties2_loss = []
    properties2_val_loss = []
    
    mlp_properties_loss = []
    mlp_properties_val_loss = []
    student_properties_loss = []
    student_properties_val_loss = []
    gaussian_properties_loss = []
    gaussian_properties_val_loss = []
    
    #Cross-validation splitted by metakaolins
    y_sol, y_met = Numpy.split(precursors, [6], axis=-1)
    y_met = Numpy.flip(y_met, axis = -1)
    met_classes = Numpy.argmax(y_met, axis=-1)
    
    skf = StratifiedKFold( n_splits = k_fold )
    
    for i, (train_index, val_index) in enumerate(skf.split(precursors, met_classes)):
        print('Processing fold #', i)
        
        train_X = properties[train_index]
        train_y = precursors[train_index]
        val_X = properties[val_index]
        val_y = precursors[val_index]

        #Simulator model
        simulator = Simulator( nb_properties, nb_precursors, unfeasible_mixture_properties, latent_dim = 32 )
    
        simulator_history = simulator.fit( train_y, train_X, verbose = 0, validation_data = (val_y, val_X))
        
        properties_loss.append(simulator_history.history["loss"])
        properties_val_loss.append(simulator_history.history["val_loss"])
        
        #Simulator model without molar ratios calculations
        simulator2 = Simulator( nb_properties, nb_precursors, unfeasible_mixture_properties, latent_dim = 32, should_compute_molar_ratios = False )
    
        simulator2_history = simulator2.fit( train_y, train_X, verbose = 0, validation_data = (val_y, val_X))
        
        properties2_loss.append(simulator2_history.history["loss"])
        properties2_val_loss.append(simulator2_history.history["val_loss"])
        
        #Formulator model Gaussian VAE
        formulator = Formulator( nb_properties, nb_precursors, latent_dim = 32 )
        trainer = Trainer( simulator, formulator, nb_properties, nb_precursors )
        
        trainer_history = trainer.fit( train_X, train_y, verbose = 0, validation_data = (val_X, val_y))
        
        gaussian_loss.append(trainer_history.history["precursors_loss"])
        gaussian_val_loss.append(trainer_history.history["val_precursors_loss"])
        gaussian_properties_loss.append(trainer_history.history["properties_loss"])
        gaussian_properties_val_loss.append(trainer_history.history["val_properties_loss"])
        
        #Formulator model MLP
        formulator = Formulator( nb_properties, nb_precursors, latent_dim = 32, is_vae = False )
        trainer = Trainer( simulator, formulator, nb_properties, nb_precursors )
        
        trainer_history = trainer.fit( train_X, train_y, verbose = 0, validation_data = (val_X, val_y))
        
        mlp_loss.append(trainer_history.history["precursors_loss"])
        mlp_val_loss.append(trainer_history.history["val_precursors_loss"])
        mlp_properties_loss.append(trainer_history.history["properties_loss"])
        mlp_properties_val_loss.append(trainer_history.history["val_properties_loss"])
        
        #Formulator model Student's t VAE
        formulator = Formulator( nb_properties, nb_precursors, latent_dim = 32, is_student = True )
        trainer = Trainer( simulator, formulator, nb_properties, nb_precursors )
        
        trainer_history = trainer.fit( train_X, train_y, verbose = 0, validation_data = (val_X, val_y))
        
        student_loss.append(trainer_history.history["precursors_loss"])
        student_val_loss.append(trainer_history.history["val_precursors_loss"])
        student_properties_loss.append(trainer_history.history["properties_loss"])
        student_properties_val_loss.append(trainer_history.history["val_properties_loss"])
        
        #Formulator model trained by itself
        formulator = Formulator( nb_properties, nb_precursors, latent_dim = 32 )
        
        formulator_history = formulator.fit( train_X, train_y, verbose = 0, validation_data = (val_X, val_y))
        
        precursors_loss.append(formulator_history.history["precursors_loss"])
        precursors_val_loss.append(formulator_history.history["val_precursors_loss"])
    
    return { 
        "precursors_loss": precursors_loss,
        "precursors_val_loss": precursors_val_loss,
        "mlp_loss": mlp_loss,
        "mlp_val_loss": mlp_val_loss,
        "student_loss": student_loss,
        "student_val_loss": student_val_loss,
        "gaussian_loss": gaussian_loss,
        "gaussian_val_loss": gaussian_val_loss,
        "properties_loss": properties_loss,
        "properties_val_loss": properties_val_loss,
        "properties2_loss": properties2_loss,
        "properties2_val_loss": properties2_val_loss,
        "mlp_properties_loss": mlp_properties_loss,
        "mlp_properties_val_loss": mlp_properties_val_loss,
        "student_properties_loss": student_properties_loss,
        "student_properties_val_loss": student_properties_val_loss,
        "gaussian_properties_loss": gaussian_properties_loss,
        "gaussian_properties_val_loss": gaussian_properties_val_loss
        }
