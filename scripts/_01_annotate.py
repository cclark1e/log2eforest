import os
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys

from Bio.SeqUtils.ProtParam import ProteinAnalysis, ProtParamData
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix

from scipy.stats import randint

###### GLOBALS #######

if len(sys.argv) > 1:
    args = sys.argv[1:]
    file_to_load = args[0]
    chemo = args[1]
    optruncount = int(args[2])
else:
    file_to_load = '../data/collatedenrichment.csv'
    chemo = 'all'
    optruncount = 1

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
print(f"DT_STRING: {dt_string}")


aa_onehots ={
    'A': '10000000000000000000',
    'C': '01000000000000000000',
    'D': '00100000000000000000',
    'E': '00010000000000000000',
    'F': '00001000000000000000',
    'G': '00000100000000000000',
    'H': '00000010000000000000',
    'I': '00000001000000000000',
    'K': '00000000100000000000',
    'L': '00000000010000000000',
    'M': '00000000001000000000',
    'N': '00000000000100000000',
    'P': '00000000000010000000',
    'Q': '00000000000001000000',
    'R': '00000000000000100000',
    'S': '00000000000000010000',
    'T': '00000000000000001000',
    'V': '00000000000000000100',
    'W': '00000000000000000010',
    'Y': '00000000000000000001'
}

scale_dicts = ['kd', 'ab', 'al', 'ag', 'bm', 'bb', 'cs', 'ci', 'es', 'eg', 'fs', 'fc', 'gd', 'gy', 'jo', 'ju', 'ki', 'mi', 'pa', 'po', 'ro', 'rm', 'sw', 'ta', 'wi', 'zi', 'Flex', 'hw', 'em', 'ja']

########################
###### FUNCTIONS #######
########################

def peptide_to_onehot_list(peptide):
    # Returns peptides as a 352x1 array of onehots
    peptide = peptide[0]
    onehots = ""
    for aa in peptide:
        onehot = aa_onehots[aa]
        onehots = onehots + onehot
    onehots = list(onehots)
    onehot_array = np.array(onehots)
    return(onehot_array)

def annotate_residues(X):
    print('annotating residues...')
    X_annotated = []
    for peptide in X:
        peptide = peptide[0].upper()
        peptide = ProteinAnalysis(peptide)
        molweight = peptide.molecular_weight()
        aromaticity = peptide.aromaticity()
        instabilityi = peptide.instability_index()
        isoelectric = peptide.isoelectric_point()
        extinctioncoef = peptide.molar_extinction_coefficient()
        gravy = peptide.gravy()
        scales = {}
        for scale_dict in scale_dicts:
            param_dict = getattr(ProtParamData, scale_dict)
            scale = peptide.protein_scale(param_dict, 5) # window size of 5 used because we already know of motifs around 5bp long
            scales[scale_dict] = scale
        combined_scales = []
        for i in range(len(scales['kd'])):
            for scale in scales.keys():
                combined_scales.append(scales[scale][i])
        X_annotated.append(combined_scales)
    return np.array(X_annotated)

#######################################
###### Set up X and y, annotate #######
#######################################

df = pd.read_csv(file_to_load, sep='\t')

if chemo == "cc":
    df = df.loc[(df['selecting.chemokine'] == 'CCL1_HUMAN') | (df['selecting.chemokine'] == 'CCL11_HUMAN') | (df['selecting.chemokine'] == 'CCL15_HUMAN') | (df['selecting.chemokine'] == 'CCL17_HUMAN') | (df['selecting.chemokine'] == 'CCL18_HUMAN') | (df['selecting.chemokine'] == 'CCL19_HUMAN') | (df['selecting.chemokine'] == 'CCL2_HUMAN') | (df['selecting.chemokine'] == 'CCL20_HUMAN') | (df['selecting.chemokine'] == 'CCL22_HUMAN') | (df['selecting.chemokine'] == 'CCL25_HUMAN') | (df['selecting.chemokine'] == 'CCL28_HUMAN') | (df['selecting.chemokine'] == 'CCL3_HUMAN') | (df['selecting.chemokine'] == 'CCL4_HUMAN') | (df['selecting.chemokine'] == 'CCL5_HUMAN') | (df['selecting.chemokine'] == 'CCL8_HUMAN')]
elif chemo == "cx":
    df = df.loc[(df['selecting.chemokine'] == 'CXCL5_HUMAN') | (df['selecting.chemokine'] == 'CXL10_HUMAN') | (df['selecting.chemokine'] == 'CXL11_HUMAN') | (df['selecting.chemokine'] == 'CXL13_HUMAN') | (df['selecting.chemokine'] == 'CXL14_HUMAN') | (df['selecting.chemokine'] == 'GROA_HUMAN') | (df['selecting.chemokine'] == 'IL8_HUMAN') | (df['selecting.chemokine'] == 'SDF1_HUMAN')]
elif chemo != "all":
        df = df.loc[df['selecting.chemokine'] == chemo]

X = np.array(df.loc[:, ['peptide']])
y = np.array(df.loc[:, ['log2E']])
y = df['log2E'].tolist()
newy = []
for val in y:
    if val >= 0.5:
        newy.append(1)
    else:
        newy.append(0)
y = newy
y = np.array(y)


X_onehots = []
for seq in X:
    X_onehot = peptide_to_onehot_list(seq)
    X_onehots.append(X_onehot)
X_onehots = np.array(X_onehots)
X_onehots = X_onehots.astype(float)
X_annotated = annotate_residues(X)

X_concatenated = []
n = 0
for onehot in X_onehots:
    concatenated = np.concatenate((onehot, X_annotated[n]), axis=0)
    X_concatenated.append(concatenated)

##################################
###### Set up forest model #######
##################################

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_concatenated, y, test_size=0.2)

print(f'X_train len: {len(X_train)}')
print(f'X_test len: {len(X_test)}')
print(f'y_train len: {len(y_train)}')
print(f'y_test len: {len(y_test)}')

# Define the model
rfmodel = RandomForestClassifier(n_estimators=100)

# Define the hyperparameters to optimize
param_distributions = {
    'n_estimators': randint(10, 1000),
    'max_depth': randint(2, 100),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}

# Define the search strategy
search = RandomizedSearchCV(
    rfmodel,
    param_distributions=param_distributions,
    n_iter=optruncount,
    cv=5,
    n_jobs=-1
)

# Train the model with hyperparameter optimization
search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = search.best_params_

# Train the final model with the best hyperparameters
rfmodel = RandomForestClassifier(**best_params)
rfmodel.fit(X_train, y_train)

################################
###### Test forest model #######
################################

predicted = rfmodel.predict(X_test)

cm = confusion_matrix(y_test, predicted)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(f'../data/{dt_string}.png')

print(f"DT_STRING: {dt_string}")

#################################
###### Save Model + Params ######
#################################

joblib.dump(rfmodel, f'../data/{dt_string}_rfmodel.pkl')
joblib.dump(search.best_params_, f'../data/{dt_string}_rfmodel_params.pkl')

