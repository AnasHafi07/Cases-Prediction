# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:16:54 2022

@author: ANAS
"""

import pickle

class ModelSaving():
    
    def __init__(self):
        pass

    def save_model(self, PATH , MODEL ):
        
        with open(PATH,'wb') as file:
            pickle.dump(MODEL,file)