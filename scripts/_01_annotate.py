import os
import numpy as np
import pandas as pd
import pickle
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys

from Bio.SeqUtils.ProtParam import ProteinAnalysis, ProtParamData
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score, accuracy_score, jaccard_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import randint

###### GLOBALS #######

sys.argv = ['poop', '../data/HD2XC_collated.csv', '1'] # DELETE WHEN MOVING TO CLUST

if len(sys.argv) > 1:
    args = sys.argv[1:]
    file_to_load = args[0]
    optruncount = int(args[1])

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

### Convert df so that 1 peptide has may log2e values, one for each chemokine
# How to handle missing chemokines? Add 'None' for now and then count and assess.

unique_chemokines = df['selecting.chemokine'].unique()
unique_peptides = df['peptide'].unique()
print(f'CHEMOKINE ORDER: {unique_chemokines}')

if os.path.exists('../data/dfsorted.pickle'):
    with open('../data/dfsorted.pickle', 'rb') as f:
        dfsorted = pickle.load(f)
    # dfsorted = pd.read_csv('../data/dfsorted.csv', sep='\t')
else:
    dfsorted = pd.DataFrame(columns=['peptide', 'log2E'])
    n = 0
    for unique_peptide in unique_peptides:
        if n % 100 == 0:
            print(f"{n} / {len(unique_peptides)} unique peptides assigned multiple log2es")
        log2es = []
        for unique_chemokine in unique_chemokines:
            matchingrow = df.loc[((df['selecting.chemokine'] == unique_chemokine) & (df['peptide'] == unique_peptide))]
            if matchingrow.empty:
                log2es.append(None)
                continue
            else:
                log2es.append(matchingrow['log2E'])
        dict = {
            'peptide': unique_peptide,
            'log2E': [log2es]
        }
        toadd = pd.DataFrame(dict)
        dfsorted = pd.concat([dfsorted, toadd])
        n += 1
    dfsorted_bytes = pickle.dumps(dfsorted)
    with open('../data/dfsorted.pickle', 'wb') as f:
        f.write(dfsorted_bytes)
    print('finished assigning multiple log2es to single peptides')

df = dfsorted

'''
# Check how many missing log2es are in dfsorted
countnone = 0
for k, v in df["log2E"].iteritems():
    v = np.array(v)
    print(v)
    for val in v:
        if not isinstance(val[0], float):
            countnone +=1
'''

X = np.array(df.loc[:, ['peptide']])
y = df['log2E'].tolist()
newy = []
for lst in y:
    newlst = []
    for val in lst:
        if val.iloc[0] <= 0:
            newlst.append(0)
        else:
            newlst.append(1)
    newy.append(newlst)

y = newy
y = np.array(y)

X_onehots = []
print('getting onehots...')
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

pos_weight = 10
neg_weight = 1
class_weights = [{0: neg_weight, 1: pos_weight} for i in range(y_train.shape[1])]

print(f'X_train len: {len(X_train)}')
print(f'X_test len: {len(X_test)}')
print(f'y_train len: {len(y_train)}')
print(f'y_test len: {len(y_test)}')

# Define the model
rfmodel = RandomForestClassifier()

# Define the hyperparameters to optimize
param_distributions = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(2, 450),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False],
    'class_weight': [class_weights]
}

# Define the search strategy
f1_scorer = make_scorer(f1_score, average='weighted')
search = RandomizedSearchCV(
    rfmodel,
    param_distributions=param_distributions,
    n_iter=optruncount,
    cv=5,
    n_jobs=-1,
    scoring=f1_scorer,
    verbose = 10
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

print("now predicting with best model...")
predicted = rfmodel.predict(X_test)

# Compute the F1 score and accuracy
f1 = f1_score(y_test, predicted, average='weighted')

# Plot the results
fig, ax = plt.subplots()
ax.bar(['F1 score'], [f1])
ax.set_ylim(0, 1)
plt.show()

plt.savefig(f'../data/{dt_string}.png')

print(f"DT_STRING: {dt_string}")
print(f'CHEMOKINE ORDER: {unique_chemokines}')
print(f'optruncount: {optruncount}')
print('best_params:')
print(best_params)

#################################
###### Save Model + Params ######
#################################

joblib.dump(rfmodel, f'../data/{dt_string}_rfmodel.pkl')
joblib.dump(search.best_params_, f'../data/{dt_string}_rfmodel_params.pkl')

