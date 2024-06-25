import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

print('Loading...')
og_betas = pd.read_csv('C:/Users/jack/PycharmProjects/TruDiagnostic/Covariate Models/Blood Type/Data/BetaMatrix_Funnorm_RCP_normalized_betas_1642081709.csv').set_index('Unnamed: 0')
print('Transposing...')
og_betas = og_betas.transpose()
print('Setting index...')
df_length = len(og_betas.index)

num_of_sets = 500
loops = range(num_of_sets)[39:]

print('Original length of df: ', df_length)
idx1s = []
idx2s = []
for s in loops:
    idx1s.append(round(df_length / 500 * s))
    idx2s.append(round(df_length / 500 * (s+1)))

print(loops)
og_betas.drop(og_betas.iloc[:(int(39 * df_length / 500))].index, inplace=True)
print('nlen ', len(og_betas))
# print('idx> ', og_betas.iloc[:(39 * 500)].index)
print(og_betas)


for s in loops:
    idx1, idx2 = idx1s[s], idx2s[s]
    c = og_betas.iloc[idx1:idx2]
    print(idx1, idx2)
    c.to_csv('data/Chunked Betas/Chunk' + str(s) + '.csv')

    og_betas.drop(index=og_betas.iloc[idx1:idx2].index, inplace=True)
    # del og_betas.iloc[idx1:idx2]
    print(len(og_betas.index))


print('Calculating principal components...')
pca = PCA(n_components=200)
x_train = pca.fit_transform(og_betas)
pd.DataFrame(x_train).to_csv('data/selected data/PCA.csv')
print(x_train, len(x_train))

# og_ages = pd.read_csv('data/PopulationData_060822.csv').set_index('Patient ID')['Decimal Chronological Age'].fillna(0)

# shared_patients = (set(og_betas.index) & set(og_ages.index))
# for patient in shared_patients:
    # og_betas.loc[patient, 'Chronological Age'] = og_ages[patient]

# train_x_labeled, test_x_labeled, train_y_labeled, test_y_labeled = train_test_split(og_betas.iloc[:, 1:], og_betas.loc[:, 'Chronological Age'], test_size=.1)

# scaler = StandardScaler()

# x_train = scaler.fit_transform(train_x_labeled)
# x_test = scaler.transform(test_x_labeled)

# x_test = pca.transform(x_test)

# explained_variance = pca.explained_variance_ratio_
# classifier = LogisticRegression()
# classifier.fit(x_train, np.array(train_y_labeled))
#
# y_pred = classifier.predict(x_test)
#
# cm = confusion_matrix(np.array(test_y_labeled), y_pred)