import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

###### GLOBALS #######
'''
The properties in this dictionary are, in order:

NEED TO VERIFY THESE VALUES!!!

Hydropathy index: KYTE-DUTTON (J. Mol. Biol. (1982) 157, 105-132)
Charge: GRANTHAM (J. Mol. Biol. (1973) 78, 349-370)
Molecular weight: HOOVER (Protein Sci. (1991) 1, 1715-1718)
Molecular formula: HOOVER (Protein Sci. (1991) 1, 1715-1718)
Isoelectric point: ZIMMERMANN-GRIMM (Proteins (1987) 2, 170-185)
'''
aa_properties = {'A': [0.62, 0.87, 15.5, 27, 1.8], 'C': [0.29, 0.52, 47.0, 44, 2.5], 'D': [-0.90, -3.50, 49.0, 80, 1.9], 'E': [-0.74, -3.50, 49.9, 95, 2.1], 
                 'F': [1.19, 0.87, 28.5, 58, 2.8], 'G': [0.48, 0.07, 60.1, 0, 1.6], 'H': [-0.40, -0.64, 79.0, 96, 3.0], 'I': [1.38, 0.94, 22.8, 45, 4.5],
                 'K': [-1.50, -3.50, 56.2, 102, 2.8], 'L': [1.06, 0.85, 23.8, 58, 3.8], 'M': [0.64, 0.17, 25.7, 75, 1.9], 'N': [-0.60, -0.64, 45.0, 56, 2.8], 
                 'P': [0.12, 0.14, 42.5, 41, 2.2], 'Q': [-0.22, -0.69, 42.3, 85, 2.8], 'R': [-2.53, -3.50, 101.3, 120, 4.5], 'S': [-0.18, -0.26, 32.7, 32, 1.6], 
                 'T': [-0.05, 0.25, 45.0, 61, 2.6], 'V': [1.08, 0.62, 23.7, 46, 3.3], 'W': [0.81, 0.37, 13.0, 84, 3.4], 'Y': [0.26, 0.23, 21.5, 59, 2.8]}

amino_acids = {'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys', 'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile', 'L': 'Leu', 
               'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro', 'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'}

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

###### FUCNTIONS #######

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
    X_annotated = []
    for peptide in X:
        peptide = peptide[0]
        peptide_annotated = []
        for residue in peptide:
            if residue in amino_acids:
                aa_property = aa_properties[residue]
                peptide_annotated.extend(aa_property)
        X_annotated.append(peptide_annotated)
    return X_annotated

###### Set up X and y, annotate #######

dfpath = '../data/collatedenrichment.csv'

df = pd.read_csv(dfpath, sep='\t')
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
X_annotated = annotate_residues(X)
X_concatenated = []
n = 0
for onehot in X_onehots:
    concatenated = np.concatenate((onehot, X_annotated[n]), axis=0)
    X_concatenated.append(concatenated)

###### Set up forest model #######

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_concatenated, y, test_size=0.2)

# Define the model
rfmodel = RandomForestClassifier(n_estimators=100)

# Define the hyperparameters to optimize
param_distributions = {
    'n_estimators': randint(10, 1000),
    'max_depth': randint(2, 50),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['auto', 'sqrt', 'log2']
}

# Define the search strategy
search = RandomizedSearchCV(
    rfmodel,
    param_distributions=param_distributions,
    n_iter=1,
    cv=5,
    random_state=42,
    n_jobs=-1
)

# Train the model with hyperparameter optimization
search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = search.best_params_

# Train the final model with the best hyperparameters
rfmodel = RandomForestClassifier(**best_params)
rfmodel.fit(X_concatenated, y)

predicted = rfmodel.predict(X_test)

print('done')

plt.scatter(y_test, predicted, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Model: True vs. Predicted Values')
plt.show()