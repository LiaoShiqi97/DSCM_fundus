# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:38:19 2021

@author: Friso Gerben Heslinga
"""

# drop observations with null values or has low quality fundus images, and split dataset
import pandas as pd
import numpy as np
import os

main_path = r'C:\Users\shiqi\PycharmProjects\DSCM_fundus\deepscm\deepscm\datasets\original_split_dataset'
features_dir = os.path.join(main_path, 'T2D_features_combined_20210921.xlsx')
train_dir = os.path.join(main_path, 'train_features.xlsx')
test_dir = os.path.join(main_path, 'test_features.xlsx')
val_dir = os.path.join(main_path, 'val_features.xlsx')
test_newT2D = os.path.join(main_path, 'test2_features.xlsx')
image_quality_dir = os.path.join(main_path, 'low_quality.xlsx')
nr_exclusion_quality = 12000

# load data
all_features = pd.read_excel(features_dir , engine='openpyxl', dtype=str)
quality_data = pd.read_excel(image_quality_dir , engine='openpyxl', dtype={'Filename': str, 'file_dir': str, 'quality': np.float64})

# Nr_unique_indiv = len(all_features.drop_duplicates(subset=['RandomID']))

# Exclude images because of poor image quality, nr_exclusion_quality!!!!
quality_sorted = quality_data.sort_values(by=['quality'], ignore_index=True)
high_quality_names = quality_sorted.loc[12000:,'Filename'].reset_index(drop=True)
all_features_selection = all_features[all_features['Filename'].isin(high_quality_names)]
# Nr_unique_indiv = len(all_features_selection.drop_duplicates(subset=['RandomID']))

# Exclude images becasuse of missing eye or fixation information
all_features_selection = all_features_selection.dropna(subset = ['eye','fixation_point']) 

# Obtain unique and exclude cases for which waist information is missing. 
features_unique = all_features_selection.drop_duplicates(subset=['RandomID'])
features_unique_cleaned = features_unique.dropna(subset = ['waist']) 

#Also exclude type 1 and other diabetes
features_unique_cleaned = features_unique_cleaned[features_unique_cleaned['N_GTS_WHO'].isin(['0','1','2','3'])]
# Nr_unique_indiv = len(features_unique_cleaned.drop_duplicates(subset=['RandomID']))

# Create subset with newly identified T2D cases
test_set2      = features_unique_cleaned[features_unique_cleaned['N_T2DM_new_diagn'] == '2'].reset_index(drop=True)
remainder_set  = features_unique_cleaned[features_unique_cleaned['N_T2DM_new_diagn'] != '2'].reset_index(drop=True)

## matching for test_set2
test_set2_new  = test_set2.astype({'Age': np.float32, 'waist': np.float32})
rest = remainder_set.astype({'Age': np.float32, 'waist': np.float32})

age_dif    = 1
waist_dif  = 2
age_dif2   = 2
waist_dif2 = 4
age_dif3   = 6
waist_dif3 = 12

matching_distance = np.zeros((len(test_set2),2))

for i in range(len(test_set2)):
    
    source_age   = test_set2_new.loc[i,'Age']
    source_sex   = test_set2_new.loc[i,'SEX']
    source_waist = test_set2_new.loc[i,'waist']
    
    matches = rest[(rest.SEX == source_sex) &  
                   (rest.Age >= (source_age - age_dif)) & (rest.Age <= (source_age + age_dif)) &
                   (rest.waist >= (source_waist - waist_dif)) & (rest.waist <= (source_waist + waist_dif)) &
                   (rest.N_GTS_WHO == '0')]
    
    if len(matches) == 0:
        print('wider search')
        matches = rest[(rest.SEX == source_sex) &  
               (rest.Age >= (source_age - age_dif2)) & (rest.Age <= (source_age + age_dif2)) &
               (rest.waist >= (source_waist - waist_dif2)) & (rest.waist <= (source_waist + waist_dif2)) &
               (rest.N_GTS_WHO == '0')]
        
        if len(matches) == 0:
            print('even wider search')
            matches = rest[(rest.SEX == source_sex) &  
               (rest.Age >= (source_age - age_dif3)) & (rest.Age <= (source_age + age_dif3)) &
               (rest.waist >= (source_waist - waist_dif3)) & (rest.waist <= (source_waist + waist_dif3)) &
               (rest.N_GTS_WHO == '0')]
 
    match = matches.iloc[0]
    match_age = match.loc['Age']
    match_waist = match.loc['waist']
    match_ID = match['RandomID']
    
    matching_distance[i,0] = np.abs(match_age-source_age)
    matching_distance[i,1] = np.abs(match_waist-source_waist)
    
    ## add to test set 2 and remove from remainder set
    test_set2_new = test_set2_new.append(match, ignore_index=True)
    rest = rest[rest.RandomID != match_ID]   

# Calculate mean distance of age and waist    
np.mean(matching_distance, axis=0) # array([0.876 year, 1.455 cm]) 

# continue with remaining IDs for others sets
randomized_IDs = rest.sample(frac=1).reset_index(drop=True)

## Select IDs	
test_IDs  = randomized_IDs.loc[0:999,'RandomID']
train_IDs = randomized_IDs.loc[1000:5399,'RandomID']
val_IDs  = randomized_IDs.loc[5400:,'RandomID']
test2_IDs = test_set2_new.loc[:,'RandomID']

## Obtain image features and save
train_features = all_features_selection[all_features_selection['RandomID'].isin(train_IDs)]
val_features   = all_features_selection[all_features_selection['RandomID'].isin(val_IDs)]
test_features  = all_features_selection[all_features_selection['RandomID'].isin(test_IDs)]
test2_features = all_features_selection[all_features_selection['RandomID'].isin(test2_IDs)]

train_features.to_excel(train_dir, index=False)
val_features.to_excel(val_dir, index=False)
test_features.to_excel(test_dir, index=False)
test2_features.to_excel(test_newT2D, index=False)
# note test2_data is cmposed of half new-diagnosed type-2 diabetes and half normal people.
