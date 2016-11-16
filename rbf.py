#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:37:04 2015

@author: pagutierrez
"""

"""
TODO: Incluir todos los import necesarios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn
import math

from sklearn.cluster import KMeans
from sklearn.cross_validation import StratifiedShuffleSplit
from scipy.spatial.distance import cdist #para cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix
from numpy import array

def entrenar_rbf(fichero_train, fichero_test, num_rbf, clasificacion, eta):
    """ Función principal
        Recibe los siguientes parámetros:
            - fichero_train: nombre del fichero de entrenamiento.
            - fichero_test: nombre del fichero de test.
            - num_rbf: número de neuronas de tipo RBF.
            - clasificacion: True si el problema es de clasificacion.
            - eta: valor del parámetro de regularización para la Regresión 
              Logística.
        Devuelve:
            - train_mse: Error de tipo Mean Squared Error en entrenamiento. 
              En el caso de clasificación, calcularemos el MSE de las 
              probabilidades predichas frente a las objetivo.
            - test_mse: Error de tipo Mean Squared Error en test. 
              En el caso de clasificación, calcularemos el MSE de las 
              probabilidades predichas frente a las objetivo.
            - train_ccr: Error de clasificación en entrenamiento. 
              En el caso de regresión, devolvemos un cero.
            - test_ccr: Error de clasificación en test. 
              En el caso de regresión, devolvemos un cero.
    """
    train_inputs, train_outputs, test_inputs, test_outputs = lectura_datos(fichero_train, 
                                                                           fichero_test)
                                                                           
    num_rbf = int(np.round(train_inputs.shape[0]*(neuronas/100.0)))  
      
    if (num_rbf < 1):
        num_rbf = 1
        
    kmedias, distancias, centros = clustering(clasificacion, train_inputs, 
                                              train_outputs, num_rbf)
    
    radios = calcular_radios(centros, num_rbf)
    
    matriz_r = calcular_matriz_r(distancias, radios)
    
    if not clasificacion:
        coeficientes = invertir_matriz_regresion(matriz_r, train_outputs)
    else:
        logreg = logreg_clasificacion(matriz_r, train_outputs, eta)

    
    distancias_test = kmedias.transform(test_inputs)  
    matriz_r_test = calcular_matriz_r(distancias_test, radios)
    
    if not clasificacion:

        predicted_test_a = np.dot(matriz_r_test, coeficientes)
        predicted_train_a = np.dot(matriz_r, coeficientes)
        predicted_test= np.round(predicted_test_a)
        predicted_train= np.round(predicted_train_a) 
        predicted_test += 0.
        predicted_train += 0.
        test_mse = mean_squared_error(test_outputs, predicted_test_a)
        train_mse = mean_squared_error(train_outputs, predicted_train_a)
        
    else:
        
        predicted_test = logreg.predict(matriz_r_test)
        predicted_train = logreg.predict(matriz_r)
        test_mse = mean_squared_error(test_outputs, predicted_test)
        train_mse = mean_squared_error(train_outputs, predicted_train)
        confusion_m = confusion_matrix(predicted_test, test_outputs)
        
        fallos = predicted_test == test_outputs
        #print fallos
        print np.where(fallos == False)[0]
        print confusion_m
    
    test_ccr = np.sum(predicted_test == test_outputs) * (test_outputs.shape[0]**-1)*100  
    train_ccr = np.sum(predicted_train == train_outputs) * (train_outputs.shape[0]**-1)*100
    
    
    return train_mse, test_mse, train_ccr, test_ccr

def lectura_datos(fichero_train, fichero_test):
    """ Realiza la lectura de datos.
        Recibe los siguientes parámetros:
            - fichero_train: nombre del fichero de entrenamiento.
            - fichero_test: nombre del fichero de test.
        Devuelve:
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - test_inputs: matriz con las variables de entrada de 
              test.
            - test_outputs: matriz con las variables de salida de 
              test.
    """
    
    train_df = pd.read_csv(fichero_train, header=None)
    test_df = pd.read_csv(fichero_test, header=None)
    
    train_inputs = train_df.values[:,0:-1]
    test_inputs = test_df.values[:,0:-1]
    
    train_outputs = train_df.values[:,-1]
    test_outputs = test_df.values[:,-1]
    

    return train_inputs, train_outputs, test_inputs, test_outputs

def inicializar_centroides_clas(train_inputs, train_outputs, num_rbf):
    """ Inicializa los centroides para el caso de clasificación.
        Debe elegir, aprox., num_rbf/num_clases patrones por cada clase.
        Recibe los siguientes parámetros:
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - centroides: matriz con todos los centroides iniciales
                          (num_rbf x num_entradas).
    """
    
    sss = StratifiedShuffleSplit(train_outputs, n_iter=1, train_size=num_rbf, test_size=None)

    
    for train_index, test_index in sss:
        centroides = train_inputs[train_index,:]
    
    indice = 0
   
    while centroides.shape[0] < num_rbf:        
        centroides = np.r_[centroides, [train_inputs[test_index[indice]]]] 
        indice += 1
        
    while centroides.shape[0] > num_rbf:
        centroides = centroides[np.random.choice(centroides.shape[0], num_rbf,0),:]
    
    
    return centroides

def clustering(clasificacion, train_inputs, train_outputs, num_rbf):
    """ Realiza el proceso de clustering. En el caso de la clasificación, se
        deben escoger los centroides usando inicializar_centroides_clas()
        En el caso de la regresión, se escogen aleatoriamente.
        Recibe los siguientes parámetros:
            - clasificacion: True si el problema es de clasificacion.
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - kmedias: objeto de tipo sklearn.cluster.KMeans ya entrenado.
            - distancias: matriz (num_patrones x num_rbf) con la distancia 
              desde cada patrón hasta cada rbf.
            - centros: matriz (num_rbf x num_entradas) con los centroides 
              obtenidos tras el proceso de clustering.
    """

    #coger centroides random y llamar una sola vez a KMeans?
    if clasificacion: 
        centros = inicializar_centroides_clas(train_inputs, train_outputs, num_rbf)
    else:
        centros = train_inputs[np.random.choice(train_inputs.shape[0],num_rbf, replace=False),:]
        
    kmedias = KMeans(n_clusters=num_rbf, init=centros, n_init=1, max_iter=500)
    distancias = kmedias.fit_transform(train_inputs)
    centros = kmedias.cluster_centers_   
    return kmedias, distancias, centros

def calcular_radios(centros, num_rbf):
    """ Calcula el valor de los radios tras el clustering.
        Recibe los siguientes parámetros:
            - centros: conjunto de centroides.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - radios: vector (num_rbf) con el radio de cada RBF.
    """

    radios = (1.0/(2.0*(num_rbf-1.0)))*sum(cdist(centros, centros, 'euclidean'))
    return radios

def calcular_matriz_r(distancias, radios):
    """ Devuelve el valor de activación de cada neurona para cada patrón 
        (matriz R en la presentación)
        Recibe los siguientes parámetros:
            - distancias: matriz (num_patrones x num_rbf) con la distancia 
              desde cada patrón hasta cada rbf.
            - radios: array (num_rbf) con el radio de cada RBF.
        Devuelve:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al principio, en la primera columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
    """
    
    
    radios_m = np.tile(radios, (distancias.shape[0],1))
    matriz_r = np.exp((-distancias**2)/(2.0*(radios_m**2)))
    
    matriz_r = np.c_[matriz_r, np.ones(distancias.shape[0])]
    return matriz_r

def invertir_matriz_regresion(matriz_r, train_outputs):
    """ Devuelve el vector de coeficientes obtenidos para el caso de la 
        regresión (matriz beta en las diapositivas)
        Recibe los siguientes parámetros:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al principio, en la primera columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
        Devuelve:
            - coeficientes: vector (num_rbf+1) con el valor del sesgo y del 
              coeficiente de salida para cada rbf.
    """

    coeficientes = np.dot(np.linalg.pinv(matriz_r),train_outputs)
   
    return coeficientes

def logreg_clasificacion(matriz_r, train_outputs, eta):
    """ Devuelve el objeto de tipo regresión logística obtenido a partir de la
        matriz R.
        Recibe los siguientes parámetros:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al principio, en la primera columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - eta: valor del parámetro de regularización para la Regresión 
              Logística.
        Devuelve:
            - logreg: objeto de tipo sklearn.linear_model.LogisticRegression ya
              entrenado.
    """

    #solver = 'lbfgs' para conjuntos de datos multiclase - sólo puede usar penalty='l2'
    #solver = 'liblinear' para conjuntos pequeños de datos
    c=1.0/eta
    logreg = LogisticRegression(fit_intercept=False, penalty='l2', C=c, solver='lbfgs')
    logreg.fit(matriz_r, train_outputs)
    
    return logreg

if __name__ == "__main__":
    train_mses = np.empty(5)
    train_ccrs = np.empty(5)
    test_mses = np.empty(5)
    test_ccrs = np.empty(5)
    
    #of = open('neuronas.csv', 'w')
    #of = open('regularizacion.csv', 'w')
    of = open('clas-reg.csv', 'w')
    #of.write("Problema, Eta, Media MSE, Desviacion Típica MSE, Media CCR, Desviación Típica CCR, Media MSE, Desviacion Típica MSE, Media CCR, Desviación Típica CCR\n")
    of.write("Problema, Media MSE, Desviacion Típica MSE, Media CCR, Desviación Típica CCR, Media MSE, Desviacion Típica MSE, Media CCR, Desviación Típica CCR\n")
    of.close()
    #for pb in ['iris', 'digits']:
    for pb in ['digits']:
        if pb == 'iris':
            neuronas = 10
            eta = 1
        else:
            neuronas = 25
            eta = 3
            
        print "Problema: "+pb
        #for neuronas in [5, 10, 25, 50]:
        #for eta in range(0,11,1):
        #for a in [1, 2]: #para la comp con reg
        for a in [1]: #para no tener que quitar index
            print "Neuronas: %d" % neuronas
            for s in range(10,60,10):
                print "-----------"
                print "Semilla: %d" % s
                print "-----------"
                np.random.seed(s)
                if a == 1:
                    train_mses[s/10-1], test_mses[s/10-1], train_ccrs[s/10-1], test_ccrs[s/10-1] = \
                entrenar_rbf('./csv/train_'+pb+'.csv',
                                 './csv/test_'+pb+'.csv', neuronas, True, 10**(-eta))
                else:
                    train_mses[s/10-1], test_mses[s/10-1], train_ccrs[s/10-1], test_ccrs[s/10-1] = \
                entrenar_rbf('./csv/train_'+pb+'.csv',
                                 './csv/test_'+pb+'.csv', neuronas, False, 10**(-eta))
                print "MSE de entrenamiento: %f" % (train_mses[s/10-1])
                print "MSE de test: %f" % (test_mses[s/10-1])
                print "CCR de entrenamiento: %.2f%%" % (train_ccrs[s/10-1])
                print "CCR de test: %.2f%%" % (test_ccrs[s/10-1])
        
            print "*********************"        
            print "Resumen de resultados"
            print "*********************"    
            
            print "MSE de entrenamiento: %f +- %f" % (train_mses.mean(), np.std(train_mses))
            print "MSE de test: %f +- %f" % (test_mses.mean(), np.std(test_mses))
            print "CCR de entrenamiento: %.2f%% +- %.2f%%" % (train_ccrs.mean(), np.std(train_ccrs))
            print "CCR de test: %.2f%% +- %.2f%%" % (test_ccrs.mean(), np.std(test_ccrs))
            #of = open('neuronas.csv', 'w')
            #of = open('regularizacion.csv', 'a')
            of = open('clas-reg.csv', 'a')
            #of.write(pb +","+str(10**-eta)+","+str(train_mses.mean())+"," +str(np.std(train_mses))+"," +str(train_ccrs.mean())+"," +str(np.std(train_ccrs))+"," +str(test_mses.mean())+"," +str(np.std(test_mses))+"," +str(test_ccrs.mean())+","+str(np.std(test_ccrs))+"\n")
            of.write(pb +","+str(train_mses.mean())+"," +str(np.std(train_mses))+"," +str(train_ccrs.mean())+"," +str(np.std(train_ccrs))+"," +str(test_mses.mean())+"," +str(np.std(test_mses))+"," +str(test_ccrs.mean())+","+str(np.std(test_ccrs))+"\n")
            of.close()