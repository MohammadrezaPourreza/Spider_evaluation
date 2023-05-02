# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:06:26 2022

@author: adeel
"""

import pandas as pd
import numpy as np
import glob
import re
import os
import json

with open(os.getcwd()+'/dev.json') as f:
    data = json.load(f)
dfjson = pd.concat((pd.json_normalize(d) for d in data), axis=0)

file_list = glob.glob(os.getcwd()+'/results/*')
for file in file_list:
    df = pd.read_csv(file)
    #df = df[0:100]
    df['GoldProcessed'] = df['GOLD SQL'].str.replace("['", '', regex=False)
    df['GoldProcessed'] = df['GoldProcessed'].str.replace("']", '', regex=False)
    df['GoldProcessed'] = df['GoldProcessed'].str.replace('["', '', regex=False)
    df['GoldProcessed'] = df['GoldProcessed'].str.replace('"]', '', regex=False)
    df['GoldRawText'] = df['GoldProcessed'].str.replace(" ", "")
    df['GoldRawText'] = df['GoldRawText'].str.lower()
    dfb = df.merge(dfjson, left_on='NLQ', right_on='question', how='left', indicator=True)
    dfb = dfb.fillna(method='ffill') 
    df_gold = dfb[['GoldProcessed', 'db_id']]
    df_predicted = df['PREDICTED SQL']
    file=file.replace('.csv','')
    file =  file.split("/")
    #file = re.findall(r"results/(.*)", file)
    file = file[-1]
    np.savetxt(os.getcwd()+'/processed_results/Gold_'+str(file)+'.txt', df_gold.values, fmt='%s', delimiter = '\t')
    np.savetxt(os.getcwd()+'/processed_results/Predicted_'+str(file)+'.txt', df_predicted.values, fmt='%s')
