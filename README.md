# Readme for “Predicting Long-Term Unemployment for Workforce System Clients”

# A. Introduction:
This project uses 2 different machine learning models in Python, Decision Tree and Random Forest, to predict unemployment outcomes (1 = unemployed, 0 = employed) for clients who received services from the U.S. Department of Labor  4 quarters after they exited workforce system services

# B.	Project website:
I created a project website in html code using GitHub Pages. You may view my project website at the following link:
https://jhsmith22.github.io/workforce_ml/
The project website contains:
1.	An embedded 2-min YouTube video highlighting my results
2.	An embedded lengthier set of slides that more fully describes my data, methods, and results
3.	A link to my Github page with the Python code
4.	A link to the zipped csv file required to feed into my Python code to replicate my results.

# C.	Setup Requirements to Run Python Code.
This project is implemented using Python. You need Python software, the proper Python libraries installed, the data file, the Python code file, and substantial computer processing resources. 
1.	Python Software. The python code was written using Python Version 3.8.5. You may need to update your Python software to this version in order to run the code.
2.	Python Libraries. The first few lines of code in “employ_ml_code.py” imports the Python libraries needed to run the remaining code. Depending on what you have installed previously, you may need to install some of these libraries with the “pip install” command in your terminal window. For example, to install matplotlib, I entered the following in my terminal window: “pip install matplotlib.” The libraries you need are as follows:
o	Numpy
o	Pandas
o	Matplotlib
o	Sklearn

3.	Data File. You can download a zipfile of the full dataset (~4 million rows) entitled “data2018Q2.zip” from my project website at the following link: https://jhsmith22.github.io/workforce_ml/. You will need to unzip the dataset before loading it into the Python Code. The file was too large to upload to Gradescope. If you would like to instead use a small dataset for testing, I have uploaded "data2018Q2_small.csv" to Gradescope. This will not produce the same results as my full data file but will allow you to test the code. 
4.	Python Code. The python code file is entitled “employ_ml_code.py.” You need to update the following line of code with the location of the unzipped “data2018Q2” data file: "data = pd.read_csv('data2018Q2.csv')." 
5.	Computer Processing Resources. The data file is quite large (~4 million rows) and running the machine learning algorithms, in particular the Random Forest model, is time-intensive. I ran the code with an 8-core i7-9700K CPU running at 3.6 GHz, with all 8 cores running in parallel, on a computer with 64 GB DDR4 RAM. The Random Forest code required several hours to complete.

# D.	Selecting the Decision Tree or Random Forest Machine Learning Algorithms.
The code is set by default to run a Decision Tree model in the "ml_algorithm function. "To switch to the Random Forest, you need to comment out the DECISION TREE MODEL BLOCK. Then remove the comment marks from the RANDOM FOREST MODEL BLOCK. As desired, you may want to change the values of the parameters in the param_grid to further refine the model

# E.	Data Dictionary.
All variables are binary 0/1 unless marked as continuous in the definition

| Variable            | Definition                                                                                                             |
|---------------------|------------------------------------------------------------------------------------------------------------------------|
| empQ4_WIOApost      | Employment status 4 quarters after exiting workforce system services.   This is the outcome variable.                  |
| service_spells      | Count of the number of cycles of “entry” and “exit” into workforce system   services (continuous)                      |
| service_days_total  | Cumulative days of service receipt from the workforce system across all   spells of service (continuous)               |
| exityr_2017         | 2017 rather than 2016 Exit Year from workforce system service                                                          |
| veteran             | Flag for veteran                                                                                                       |
| age_cat_31_40       | Age 31-40 in years at program entry. Ommited Age 25-30 category for   interpretability.                                |
| age_cat_41_50       | Age 41-50 in years at program entry. Ommited Age 25-30 category for   interpretability.                                |
| age_cat_51_65       | Age 51-65 in years at program entry. Ommited Age 25-30 category for   interpretability.                                |
| race_eth_1.0        | Hispanic. Omitted “White” for interpretability.                                                                        |
| race_eth_2.0        | Asian (Not Hispanic). Omitted “White” for interpretability.                                                            |
| race_eth_3.0        | Black (Not Hispanic). Omitted “White” for interpretability.                                                            |
| race_eth_7.0        | Multiple Race (not Hispanic). Omitted “White” for interpretability.                                                    |
| race_eth_8.0        | Native Hawaiian/Pacific Islander/American Indian/Alaska Native (not   Hispanic). Omitted “White” for interpretability. |
| race_eth_9.0        | No response to race/ethnicity. Omitted “White” for interpretability.                                                   |
| sex_2.0             | Female.                                                                                                                |
| sex_9.0             | No response to race. Ommitted male for interpetability.                                                                |
| edu_entry_0.0       | Less than HS. Omitted “Bachelor’s Degree or higher” for interpretability.                                              |
| edu_entry_1.0       | HS diploma or GED.                                                                                                     |
| edu_entry_4.0       | Omitted “Bachelor’s Degree or higher” for interpretability.                                                            |
| edu_entry_5.0       | Postsecondary technical or vocational certificate. Omitted “Bachelor’s   Degree or higher” for interpretability.       |
| edu_entry_6.0       | Associate’s degree. Omitted “Bachelor’s Degree or higher” for   interpretability.                                      |
| criminal_entry_1.0  | Has a criminal history. Ommitted "no criminal history" for   interpetability                                           |
| criminal_entry_9.0  | Refused to answer criminal history. Ommitted "no criminal   history" for interpetability                               |
| lowincome_entry     | Flag for low-income at entry                                                                                           |
| esl_entry           | Flag for English as a Second Language at entry                                                                         |
| sparent_entry       | Flag for single parent at entry                                                                                        |
| unemplong_entry     | Flag for being unemployed for 27 or more consecutive weeks                                                             |
| pubassist           | Flag for receipt of TANF, SNAP, SSI, or other reported assistance                                                      |
| state_AK - state_WY | State that submitted the participant data. Omitted CA for   interpretability.                                          |
| skillrating1        | Active Learning- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                           |
| skillrating2        | Active Listening- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                          |
| skillrating3        | Complex Problem Solving- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                   |
| skillrating4        | Coordination- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                              |
| skillrating5        | Critical Thinking- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                         |
| skillrating6        | Equipment Maintenance- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                     |
| skillrating7        | Equipment Selection- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                       |
| skillrating8        | Installation- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                              |
| skillrating9        | Instructing- Skill Rating on a scale from 0-5 for most recent occcupation   (continuous)                               |
| skillrating10       | Judgment and Decision Making- Skill Rating on a scale from 0-5 for most   recent occcupation (continuous)              |
| skillrating11       | Learning Strategies- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                       |
| skillrating12       | Management of Financial Resources- Skill Rating on a scale from 0-5 for   most recent occcupation (continuous)         |
| skillrating13       | Management of Material Resources- Skill Rating on a scale from 0-5 for   most recent occcupation (continuous)          |
| skillrating14       | Management of Personnel Resources- Skill Rating on a scale from 0-5 for   most recent occcupation (continuous)         |
| skillrating15       | Mathematics- Skill Rating on a scale from 0-5 for most recent occcupation   (continuous)                               |
| skillrating16       | Monitoring- Skill Rating on a scale from 0-5 for most recent occcupation   (continuous)                                |
| skillrating17       | Negotiation- Skill Rating on a scale from 0-5 for most recent occcupation   (continuous)                               |
| skillrating18       | Operation Monitoring- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                      |
| skillrating19       | Operation and Control- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                     |
| skillrating20       | Operations Analysis- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                       |
| skillrating21       | Persuasion- Skill Rating on a scale from 0-5 for most recent occcupation   (continuous)                                |
| skillrating22       | Programming- Skill Rating on a scale from 0-5 for most recent occcupation   (continuous)                               |
| skillrating23       | Quality Control Analysis- Skill Rating on a scale from 0-5 for most   recent occcupation (continuous)                  |
| skillrating24       | Reading Comprehension- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                     |
| skillrating25       | Repairing- Skill Rating on a scale from 0-5 for most recent occcupation   (continuous)                                 |
| skillrating26       | Science- Skill Rating on a scale from 0-5 for most recent occcupation   (continuous)                                   |
| skillrating27       | Service Orientation- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                       |
| skillrating28       | Social Perceptiveness- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                     |
| skillrating29       | Speaking- Skill Rating on a scale from 0-5 for most recent occcupation   (continuous)                                  |
| skillrating30       | Systems Analysis- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                          |
| skillrating31       | Systems Evaluation- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                        |
| skillrating32       | Technology Design- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                         |
| skillrating33       | Time Management- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                           |
| skillrating34       | Troubleshooting- Skill Rating on a scale from 0-5 for most recent   occcupation (continuous)                           |
| skillrating35       | Writing- Skill Rating on a scale from 0-5 for most recent occcupation   (continuous)                                   |
