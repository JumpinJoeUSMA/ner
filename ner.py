################################################################################################
# IBM Confidential Materials
# Licensed Materials - Property of IBM
# IBM Situational Awareness
# (C) Copyright IBM Corp. 2021 All Rights Reserved.
# US Government Users Restricted Rights
#  - Use, duplication or disclosure restricted by
#    GSA ADP Schedule Contract with IBM Corp.
################################################################################################
import pandas as pd
import csv
import os
import numpy as np
from datasets import load_metric
from sklearn.metrics import classification_report
import inspect
import configparser
import easydict

# def read_config(c_file):
#     config = configparser.ConfigParser()
#     config.read(c_file)
#     opt = easydict.EasyDict()
#     if 'params' in config.sections():
#         opt.input_folder = config.get('params','input_folder')
#         opt.output_folder = config.get('params','output_folder')
#         opt.max_words = config.getint('params','max_words')
#     return opt

# class NER:
#     def __init__(self, config_file):
#         self.opt = read_config(config_file)
#         self.input_folder = self.opt.input_folder
#         self.output_folder = self.opt.output_folder
#         self.max_words = self.opt.max_words
#     def train:
    
#     def predict:

# Maybe we should create a predict object... it has:
# 1) the location to the model
# 2) data to predict - need to figure out the format required...
#           as of now, it requires a datasets.arrow_dataset.Dataset



def cleanPreds(predictions, labels, label_list):
    """
    Helper function to take HF predict output (tensor) and convert to a 2-D tensor of model predicted
    values and true labels. Each row corresponds to a "chunk"... each column to a word. 
    Removes artificial mask labels (-100 code from HF). 
    Args: 
        predictions (tensor): model predictions from trainer.predict(dataset) call
        labels (tensor): model labels from trainer.predict(dataset) call
        label_list (list): list of labels corresponding to integer HF representation
    Returns: 
        predictionTensor (2-D tensor): formatted output useful for other purposes
        labelsTensor (2-D tensor): formatted output useful for other purposes
    """   
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    predictionTensor = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    labelsTensor = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return(predictionTensor, labelsTensor)

def removeBI(predictionTensor):
    """
    Helper function to flatten 2-D tensor output from cleanPreds into a 1D vector
    and remove B- and I- indicators. Output is useful for creating confusion matrices.  
    Args: 
        predictionTensor (2-D tensor): output from cleanPreds
    Returns: 
        predictionVec (1D vector): 
    """   
    predictionVec = np.hstack(predictionTensor)
    predictionVec = [sub.replace('B-','') for sub in predictionVec]
    predictionVec = [sub.replace('I-','') for sub in predictionVec]
    return(predictionVec)

def combinePreds(firstOrder, secondOrder):
    """
    Method to combine label predictions from a firstOrder prediction and secondOrder 
    prediction. If firstOrder prediction does not contain a label for a word ("O" designation), 
    the method checks to see if the secondOrder prediction contains a label. If so, the method
    updates the "O" with secondOrder label. 
    Args: 
        firstOrder (2-D tensor): output from cleanPreds representing "best" prediction
        secondOrder (2-D tensor): output from cleanPreds representing "lesser" prediction
    Returns:
        combPred (2-D tensor): updated "best" prediction for NER labels. 
    """
    combPred = []
    for chunk1, chunk2 in list(zip(firstOrder, secondOrder)):
        adjChunk = []
        for label1, label2 in list(zip(chunk1, chunk2)):
            if label1 == "O":
                adjChunk.append(label2)
            else:
                adjChunk.append(label1)
        combPred.append(adjChunk)
    return(combPred)

def visPreds():
    # use displacy to visualize
    return()

def predictionCat():
    # get dataframe that labels correct or incorrect labeling
    return()



def classReport(trueLabels, *args):
    """
    Helper function to compute and display classification reports that compare 
    trueLabels with at least one other model output. User may specify as many models
    as desired. 
    Args: 
        trueLabels (1-D vector): output following removeBI call
        *args (1-D vectors): output following removeBI call. 
            Additional models separated by comma
    Returns: 
        NONE
    """
    for ar in args:
        print(classification_report(trueLabels,ar))

def heatmap():
    return()

def metrics_HF(predictionTensor, labelsTensor):
    """
    Helper function to compute common evaluation metrics (F1, Accuracy, Precision, Recall)
    for aggregate and individual label categories. 
    Args: 
        predictionTensor (2-D tensor): Prediction output from cleanPreds
        labelsTensor (2-D tensor): Label output from cleanPreds
    Returns: 
        metricResults (dict): Dictionary of evaluation metrics for aggregate and individual
            label categories
    """
    metric = load_metric("seqeval")
    return(metric.compute(predictions = predictionTensor, references = labelsTensor))
   
# if __name__ =="__main__":
#     dictFile = 'dictionaryExxon.csv'
#     tokenFile = 'out_30tokens.csv'
#     samplePCT = 0.5
#     dict_label(dictFile, tokenFile,samplePCT)