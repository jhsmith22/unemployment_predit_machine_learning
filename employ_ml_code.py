#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import needed Python Libraries

# Numpy and Pandas libraries
import numpy as np
import pandas as pd

# Matplot library
import matplotlib as mpl
from matplotlib import pyplot as plt

# sklearn library
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.tree import export_graphviz
from sklearn import datasets
from sklearn import tree


# In[4]:


# Read in csv data file into a pandas dataframe called "data" (Update the path of the csv file in the line below as needed)
# See Readme file to find where to download this data file
# Note that this data file was pre-processed to include only participants between the age of 25-65 that had Q4 outcome data.
# In the pre-processing, it was also merged with the O*NET skills dataset on "SOC code" of the most recent occupation (pirl403)
# Raw, unprocess data file available from https://www.dol.gov/agencies/eta/performance/results/pirl and https://www.onetcenter.org/dictionary/25.2/excel/skills.html
data = pd.read_csv('data2018Q2.csv')

# Rename the data columns to recognizeable variable names
data.rename(columns = {
                    'pirl100' :'id1', 
                    
                    # geography
                    'pirl3000' : 'state',
    
                    # demographics
                    'pirl201' :'sex',
                    'pirl3056' : 'age',
                    'pirl202' : 'disabled',
                    'pirl3023' : 'race_eth',
                    'pirl300' : 'veteran',
                    'pirl801' : 'criminal_entry',
                    'pirl803' : 'esl_entry',
                    'pirl806' : 'sparent_entry',
    
                    # education and employment status at entry
                    'pirl408' : 'edu_entry',
                    'pirl804' : 'lowskills_entry',
                    'pirl400' : 'emp_entry',
                    'pirl401' : 'uc_entry',
                    'pirl402' : 'unemplong_entry',
                    'pirl802' : 'lowincome_entry', 
  
                    # public assistance programs at entry
                    'pirl600' : 'tanf_entry',
                    'pirl602' : 'ssi_entry',
                    'pirl603' : 'snap_entry',
                    'pirl604' : 'publicother_entry',

                    # WIOA service dates
                    'pirl900' : 'entry_dt',
                    'pirl901' : 'exit_dt',
                    'pirl3057' : 'exit_yr',
  
                    # Outcome: Employment status 4 quarters after exit
                    'pirl3017' : 'empQ4_WIOApost' ,                
                    }, inplace = True)
                    
# Drop unneeded columns that I have not renamed that start with "pirl"
unwanted = data.columns[data.columns.str.startswith('pirl')]
data.drop(unwanted, axis=1, inplace=True)


# In[5]:


# Rename the dataframe as "df"
df = pd.DataFrame(data)


# In[6]:


# ENGINEER WIOA EXPERIENCE PREDICTORS

# 1. Calculate workforce service duration in days "service_days" for every spell of service = Exit Date - Entry Date

# Entry Date: Convert to string and then to a datetime format
df['entry_dt_str'] = df['entry_dt'].apply(str)
df['entry_dt_str1'] = pd.to_datetime(df['entry_dt_str'], format='%Y%m%d')

# Exit Date: Convert to string and then to a datetime format
df['exit_dt_str'] = df['exit_dt'].apply(str)
df['exit_dt_str1'] = pd.to_datetime(df['exit_dt_str'], format='%Y%m%d')

# Create "service_days": Subtract Entry Date from Exit Date. Pull out the days into a variable "service_days"
df['service_time'] = df['exit_dt_str1'] - df['entry_dt_str1']
df['service_days'] = df["service_time"].dt.days

# 2. Create a binary variable "exit_yr_2017" that =1 if the client exited services in 2017, =0 if exited in 2016
df['exityr'] = pd.DatetimeIndex(df['exit_dt_str1']).year
df['exityr_2017'] = df['exityr'].replace([2016, 2017], [0, 1])

# 3. Create "service_spells" in a new dataframe "dups." "service_spells" represents the number of duplicate rows per ID
# "id1" uniquely identifies participants
df['dups'] =data.duplicated(subset=['id1'], keep=False)
dups = df.groupby(["id1"]).size().reset_index(name='service_spells')

# 4. Create "service_days_total" in a new dataframe "servicetot" that represents the total duration in workforce system service accross all service spells
servicetot = df.groupby(['id1'])['service_days'].agg(['sum'])
servicetot.rename(columns = {'sum':'service_days_total'}, inplace = True)

# Merge the 2 new dataframes back into the main "df" dataframe and drop duplicates on "id1"
# The resulting dataset has 1 row per participant (1 row per "id1")

# Merge1: Merge "service_spells" into main dataframe using a left join. Unique ID = "id1"
merge1 = pd.merge(df, dups, on='id1', how='left')

# Merge2: Merge "service_days_total" into main dataframe using a left join. Unique ID = "id1"
df = pd.merge(merge1, servicetot, on='id1', how='left')

# Keep the spell of service with the the most recent exit. 
# First sort so the first entry is the most recent exit date
# Drop duplicates, keeping the "first" row of the duplicate set
df.sort_values(by = ['id1', 'exit_dt_str1'], ascending=False, inplace=True)
df.drop_duplicates(subset=['id1'], keep='first',inplace=True)


# In[7]:


# RECODE OUTCOME

# outcome: employment Q4 after exit. Set so 1= unemployed & 0 = employed.
df['empQ4_WIOApost'] = df['empQ4_WIOApost'].replace([1, 0], [0, 1])


# In[8]:


# RECODE FEATURES 

# 1. Recode State: Change US Territories to missing
df['state'] = df['state'].replace(["GU", "PR", "VI"], [np.nan, np.nan, np.nan])

# 2. Recode Veteran Status & Single Parent (Original: yes = 1, no = 0, did not identify = 9)
# Combine "did not identify" with "no"
recode_list = ["veteran", "sparent_entry"]
for x in recode_list:
    df[x] = df[x].replace([9], [0])

# 3. Recode Race/Ethnicity
# Combine #4 & #5 (4 = Native Hawaiian or Pacific Islander, 5 = American Indian or Alaska Native "Other") as new category #8
# Create a non-response category #9 since we don't want to loose 7% of the sample that is missing on this variable
df['race_eth'] = df['race_eth'].replace([4, 5, np.nan], [8, 8, 9])

# 4. Recode the continuous Age variable:Create a categorical age variable with 4 bins: age 25-30, age 31-40, age 41-50, age 51-65
category = pd.cut(df.age,bins=[0,30,40,50,65],labels=['25_30','31_40','41_50','51_65'])
df['age_cat'] =category

# 5. Recode Educational Attainment at Entry: 
# (Original: hs diploma =1, GED = 2, disabled IEP completion = 3, completed 1+ yrs post-sec = 4, 
# postsec technical or vocational certificate = 5, associates =6, BA = 7, more than BA = 8, none = 0))
# Update: Combine HS Diploma & GED (1 & 2 = cat #1), set IEP #3 to missing, combine BA & more than BA (7 & 8 to cat #7)
df['edu_entry'] = df['edu_entry'].replace([2, 3, 8], [1, np.nan, 7])

# 6. Recode "Employed at Entry" (Original: employed at entry =1, notice of termination = 2, not in labor force = 3, 0 = unemployed and seeking work)
# Combine "notice of termination (#2)" with "unemployed and seeking work (#0)"
df['emp_entry'] = df['emp_entry'].replace([2], [0])

# 7. Recode "Unemployment Compensation status" (Original: UC eligible & referred through RESEA = 1, UC eligible & referred through WPRS = 2, 
# UC eligible & referred through other =3, # eligible but exhausted UC =4, exempt from work search requirements =5, not a UC claimant or exhaustee = 0)
# Combine different UC eligible status (#1, #2, #3) to #1. Combine "not eligible" categories (#0, #4, #5) together to #0.
df['uc_entry'] = df['uc_entry'].replace([2,3,4, 5], [1, 1, 0, 0])

# 8. Create Public Assistance predictor =1 if tanf, ssi, snap, or other public assistance =1. Otherwise, set = 0.
df['pubassist'] = np.where((df['tanf_entry'] == 1) |
                         (df['ssi_entry'] > 0) |
                         (df['snap_entry'] == 1) |
                         (df['publicother_entry'] == 1),
                         1, 0)


# In[27]:


# CREATE FINAL FEATURE and OUTCOME DATASETS

# 1. Cleaning

# 1A. Create dataframe "final_features" that contains all features and the outcome variable. Include the skills ratings.
final_features = df[['service_spells', 'service_days_total', 'exityr_2017', 'age_cat', 'race_eth', 'edu_entry', 'criminal_entry',
           'sex', 'veteran', 'lowincome_entry', 'esl_entry', 'sparent_entry',
           'unemplong_entry', 'pubassist', 'state', 'empQ4_WIOApost']].copy()

# Include O*Net Skills Ratings as features in "final_features"
skills = df.filter(regex='^skill',axis=1)
final_features = pd.concat([final_features, skills], axis=1) 

# 1C. Remove any rows with missing values from the dataframe "final_features"
final_features.dropna(inplace=True)

# 2. Create final outcome dataset. Drop the outcome from the final_features dataset
final_outcome = final_features[['empQ4_WIOApost']].copy()
final_features.drop(['empQ4_WIOApost'], axis=1, inplace=True) 

# 3. Create final feature dataset

# 3A. Execute 1-hot encoding for categorical variables (convert them to 0/1 dummies)
dummy_list = ["age_cat", "race_eth", "sex", "edu_entry", "criminal_entry", "state"]

for x in dummy_list:
    dummies = pd.get_dummies(final_features[x], prefix=x)
    final_features = final_features.drop(x, 1)
    final_features = pd.concat([final_features, dummies], axis=1) 
    
# 3B. Drop 1 dummy from each catgegorical variable for interpretability
# From "age_cat," drop age cat #1 (25-30)
# From "race," drop "white" cat #6
# From "sex," drop male cat #1
# From "edu_entry," drop BA or above cat #7
# From "criminal_entry," drop "no" cat #0
# From "state," drop CA
final_features.drop(['age_cat_25_30', 'race_eth_6.0', 'sex_1.0', 'edu_entry_7.0', 'criminal_entry_0.0', 'state_CA'], axis=1, inplace=True) 

# 4. CREATE NUMPY ARRAYS TO LOAD INTO SCKLEARN MACHINE LEARNING MODELS
X = final_features.values
y = final_outcome.values


# In[29]:


# RUN MACHINE LEARNING ALGORITHM (ml_algorithm function)

# The code is set by default to run a Decision Tree model in the "ml_algorithm function"
# To switch to the Random Forest, you need to comment out the DECISION TREE MODEL block
# Then remove the comment marks from the RANDOM FOREST MODEL
# As desired, you may want to change the values of the parameters in the param_grid to further refine the model

# DECISION TREE MODEL BLOCK
model = DecisionTreeClassifier
param_grid = {'max_leaf_nodes': [2, 4, 10, 20, 30, 40, 50, 60, 100]}

#RANDOM FOREST MODEL BLOCK
model = RandomForestClassifier
# param_grid = {
#     'max_depth': [4, 10, 30, 50],
#     'max_features': [10, 14], 
#     'n_estimators': [100, 200]
# }

# The ml_algorithm takes 2 arguments as inputs, which are defined above in the DECISION TREE MODEL BLOCK or RANDOM FOREST MODEL BLOCK
# I modified code from the following textboook for steps 2-6 below:
#An Introduction to Machine Learning with Python by Andreas C. Muller and Sarah Guido (O’Reilly).
#Copyright 2017 Sarah Guido and Andreas Muller, 978-1-449-36941-5

def ml_algorithm(model, param_grid):

    # 2. Instantiate the grid search model to identify the *best* parameter in my param_grid
    # The *best* parameter is the one that creates a decision tree that most accurately predicts the outcomes in my cross-validation dataset
    # Param_grid = param_grid, meaning the algorithm is cacluating and comparing results for each the parameters listed in my param_grid.
    # verbose = 1 means prints wordier information on its progress
    # cv = 5 means use 5-fold cross-validation
    # n_jobs = use all CPU cores
    grid_search = GridSearchCV(model(random_state=0), param_grid = param_grid, verbose = 1, cv = 5, n_jobs=-1)

    # 3. Split data into training and test datasets. The test dataset is reserved until the final step.
    # Set Random seed to 0 so I can replicate the results.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # 4. Execute the model
    grid_search.fit(X_train, y_train)

    # 5. Print out grid-search results
    # BEST PARAMETERS prints out the *best* parameter in my param_grid
    print("Best parameters: {}".format(grid_search.best_params_))

    # BEST CROSS_VALIDATION SCORE prints out the accuracy rate (number of correctly classified outcomes in the validation datasets) for the *best* parameter in my param_grid
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    # PRINT OUT A SUMMARY OF RESULTS. 
    # The grid_search.best_estimator_ is what you use to access the model with the best paramters trained on the whole training dataset
    print("Best estimator:\n{}".format(grid_search.best_estimator_))

    # 6. DECISION TREE TEST SCORE
    # Evaluate how well the *best* parameters generalize to unseen data.
    # grid_search.score applies the *best* parameters to the reserved test dataset
    # It outputs the same accuracy metric: The number of correctly classified outcomes in the test dataset
    # This is the final score that we use to assess our model
    print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
    
    # 7. VISUALIZE 15 MOST IMPORTANT FEATURES FOR PREDICTING UNEMPLOYMENT
    feature_names = final_features.columns.tolist()

    # Transform the Feature Importance (FI) Scores to a Pandas Dataframe
    d = {'Features':feature_names,'FI':grid_search.best_estimator_.feature_importances_}
    df = pd.DataFrame(d)  
  
    # View list of top 15 features
    df.sort_values(['FI'], ascending=False, inplace=True)
    df2 = df.iloc[:15,:]
    print(df2)
    
    # Plot the Top 15 features in a vertical bar chart. First need to sort the FI scores in descending order.
    df2.plot.barh(x="Features", y="FI", rot=0, title="Top 15 Most Important Features");
    plt.show(block=True)
    plt.savefig('treechart.pdf')
    
    return(grid_search.best_params_, grid_search.best_estimator_, grid_search.score(X_test, y_test))
           
ml_algorithm(model, param_grid) 


