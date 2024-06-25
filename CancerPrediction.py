import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import random
import matplotlib as plt

'''Notes
# Ensure the distribution of synthetic data matches the distribution of the original data

# Check to make sure that the patients match with their statuses and methylation data **

# Check out/use differential methylation feature selection
'''

'''Customization parameters'''
modelType = 'lung'.lower()  # can be heme, lung, or any
feat_select_method = 'all'.lower()  # can be differential or all
new_feature_select = False  # determines whether to execute feature selection or import previously selected CpG's
normalize_distrib = True  # whether to normalize class distribution using synthetic data
overall_upsample = True  # whether to up-sample the entire dataset
percent_variance = 5  # the variance found in loaded-in upsampled data and to be used in data construction


'''Data Preparation'''
og_cancer = pd.read_csv('data/OhioState_pData.csv').set_index('IDAT_ID')
og_cancer.sort_index(0, inplace=True)

if new_feature_select:
    read_file = 'OhioState_BMIQ_betas.csv'
else:
    if normalize_distrib:
        read_file = 'AllSelectedData' + modelType + '.csv'
    elif overall_upsample:
        read_file = 'Upsampled' + str(percent_variance) + '%.csv'
    else:
        read_file = 'OverallUpsampled' + str(percent_variance) + '%.csv'

og_betas = pd.read_csv(read_file)
og_betas.set_index('Unnamed: 0', inplace=True)

relevant_cpgs = pd.read_csv('data/selected data/lungCancerRelevantCpGs.csv')
# og_betas.rename(columns={'Unnamed: 0': 'CpG Site'}, inplace=True)

# og_betas.sort_index(1, inplace=True)

## Removing x's in patient ID's
# renamed = [str(to_rename).replace('X', '') for to_rename in og_betas.columns]
# og_betas.rename(columns=dict(zip(list(og_betas.columns), renamed)), inplace=True)
#
# og_betas.to_csv('data/OhioState_BMIQ_betas2.csv')

print('Cancer data: \n', og_cancer)

cpgs = og_betas.index
patients = []
new_c_status = []
print(len(og_cancer.index))
for patient in og_cancer.index:
    # adding only patients in cancer data
    if patient in og_betas.index:
        # num_replicates = len(og_betas.loc[patient].index)  # number of copies of a patient from up-sampling
        # print(og_betas.loc[patient].index)
        # for i in range(num_replicates):
        patients.append(patient)

        # Building cancer status list
        status = str(og_cancer.loc[patient, 'Clinic']).lower()
        if status == 'healthy':
            og_betas.loc[patient, 'Blood Type'] = 0
        else:
            # Checking for model type and identifying cancer presence accordingly as a 1
            if modelType == 'any':
                og_betas.loc[patient, 'Blood Type'] = 1
            elif modelType == 'lung':
                if status == 'lung':
                    og_betas.loc[patient, 'Blood Type'] = 1
                else:
                    og_betas.loc[patient, 'Blood Type'] = 0
            elif modelType == 'heme':
                if status == 'heme':
                    og_betas.loc[patient, 'Blood Type'] = 1
                else:
                    og_betas.loc[patient, 'Blood Type'] = 0
            else:
                print('error in modelType input')
    else:
        print('Missing:', patient)

# Removing any beta data not in the cancer data
og_betas = og_betas.loc[patients]

print('Number of positive IDs: ', list(og_betas['Cancer Status'].values).count(1))
print('Beta data: \n', og_betas)

# Feature selection
if new_feature_select:
    print('Starting feature selection... ')

    if modelType == 'all':
        to_select = og_betas.index[:-1]
        threshold = 3
    elif modelType == 'lung':
        to_select = pd.read_csv('data/LungvsHealthy_DMPs_Filtered.csv')['CG.site']
        threshold = 3
    elif modelType == 'heme':
        to_select = pd.read_csv('data/HemevsHealthy_DMPs_Filtered.csv')['CG.site']
        threshold = 3
    else:
        to_select = pd.read_csv('data/RelevantDifferentialMethylation.csv')
        to_select = to_select[to_select.columns[0]]
        threshold = 3

    importances = mutual_info_classif(og_betas[to_select], og_betas.loc[:, 'Cancer Status'])

    # PLOT IMPORTANCES, organize them, and look for an "elbow"

    feat_importances = pd.Series(importances, to_select)
    average_importance = np.average(feat_importances)
    print(feat_importances, '\n', average_importance)

    relevant_cpgs = feat_importances[feat_importances > average_importance*threshold]
    relevant_cpgs.to_csv('data/' + modelType + 'CancerRelevantCpGs.csv')
    relevant_cpgs = relevant_cpgs.index

    final_cpgs = list(relevant_cpgs)
    final_cpgs.append('Cancer Status')
    og_betas = og_betas[final_cpgs]
    og_betas.to_csv('data/AllSelectedData' + modelType + '.csv')
# else:
#     if modelType == 'all':
#         if feat_select_method == 'all':
#             relevant_cpgs = pd.read_csv('data/allCancerRelevantCpGs.csv')['CpG Site']
#         else:
#             relevant_cpgs = pd.read_csv('data/RelevantDifferentialMethylation.csv')['CG.site']
#     elif modelType == 'lung':
#         if feat_select_method == 'all':
#             relevant_cpgs = pd.read_csv('data/lungCancerRelevantCpGs.csv')['CG.site']
#         else:
#             relevant_cpgs = pd.read_csv('data/LungvsHealthy_DMPs_Filtered.csv')['CG.site']
#     elif modelType == 'heme':
#         if feat_select_method == 'all':
#             relevant_cpgs = pd.read_csv('data/hemeCancerRelevantCpGs.csv')['CG.Site']
#         else:
#             relevant_cpgs = pd.read_csv('data/LungvsHealthy_DMPs_Filtered.csv')['CG.site']
#     else:
#         relevant_cpgs = pd.read_csv('data/RelevantDifferentialMethylation.csv')['CG.site']
#         relevant_cpgs.set_index(relevant_cpgs.columns[0], inplace=True)

print('Number of selected features: ', len(relevant_cpgs))
print('Final Number of CpGs: ', len(final_cpgs))


'''Synthetic data construction'''
is_cancerous = og_betas[og_betas['Cancer Status'] == 1]
not_cancerous = og_betas[og_betas['Cancer Status'] == 0]
pre_distribution = len(is_cancerous) / (len(og_betas.index)-1) * 100
print('Distribution of positive cancer status:', pre_distribution)


def create_duplicate(to_replicate, pct_variance):
    df_duplicate = to_replicate
    duplicate_length = len(df_duplicate.index)

    lower_limit = (100 - pct_variance) / 100
    upper_limit = (100 + pct_variance) / 100

    idx1, idx2 = df_duplicate.index[round(.1 * duplicate_length)], df_duplicate.index[round(.2 * duplicate_length)]
    idx3, idx4 = df_duplicate.index[round(.3 * duplicate_length)], df_duplicate.index[round(.4 * duplicate_length)]
    idx5, idx6 = df_duplicate.index[round(.5 * duplicate_length)], df_duplicate.index[round(.6 * duplicate_length)]
    idx7, idx8 = df_duplicate.index[round(.7 * duplicate_length)], df_duplicate.index[round(.8 * duplicate_length)]
    idx9 = df_duplicate.index[round(.9 * duplicate_length)]

    features_completed = 0
    for feature in to_replicate.columns[:-1]:
        df_duplicate.loc[:idx1, feature] = df_duplicate.loc[:idx1, feature] * random.uniform(lower_limit, upper_limit)
        df_duplicate.loc[idx1:idx2, feature] = df_duplicate.loc[idx1:idx2, feature] * random.uniform(lower_limit, upper_limit)
        df_duplicate.loc[idx2:idx3, feature] = df_duplicate.loc[idx2:idx3, feature] * random.uniform(lower_limit, upper_limit)
        df_duplicate.loc[idx3:idx4, feature] = df_duplicate.loc[idx3:idx4, feature] * random.uniform(lower_limit, upper_limit)
        df_duplicate.loc[idx4:idx5, feature] = df_duplicate.loc[idx4:idx5, feature] * random.uniform(lower_limit, upper_limit)
        df_duplicate.loc[idx5:idx6, feature] = df_duplicate.loc[idx5:idx6, feature] * random.uniform(lower_limit, upper_limit)
        df_duplicate.loc[idx6:idx7, feature] = df_duplicate.loc[idx6:idx7, feature] * random.uniform(lower_limit, upper_limit)
        df_duplicate.loc[idx7:idx8, feature] = df_duplicate.loc[idx7:idx8, feature] * random.uniform(lower_limit, upper_limit)
        df_duplicate.loc[idx8:idx9, feature] = df_duplicate.loc[idx8:idx9, feature] * random.uniform(lower_limit, upper_limit)
        df_duplicate.loc[idx9:, feature] = df_duplicate.loc[idx9:, feature] * random.uniform(lower_limit, upper_limit)

        features_completed += 1
        if (len(to_replicate.columns) - features_completed) % 10000 == 0:
            print(str(int(features_completed / len(to_replicate.columns) * 100)), '% complete generating data')

    return df_duplicate, duplicate_length


if normalize_distrib:
    # Ensuring that the less prevalent subset is the one being up-sampled
    if len(is_cancerous) < len(not_cancerous):
        to_upsample = is_cancerous
    else:
        to_upsample = not_cancerous

    pre_distribution = len(is_cancerous) / (len(og_betas.index)-1) * 100
    upsample = 10

    df_duplicate, duplicate_length = create_duplicate(og_betas, percent_variance)

    if pre_distribution > 30:
        df_duplicate = df_duplicate.sample(frac=1).iloc[:int(.5*len(df_duplicate.index))]
        duplicate_length = len(df_duplicate.index)

    if pre_distribution > 20:
        bound = 25
    else:
        bound = 15

    runs = 0
    while 42 < pre_distribution or pre_distribution < 58:
        if upsample < 1:
            upsample = 1
            break
        print(pre_distribution, upsample)
        deviance = 50 - pre_distribution
        if 0 < deviance < bound:
            upsample += 1
        elif 0 > deviance > -bound:
            upsample -= 1
        else:
            upsample += round(deviance / 5)

        new_total = upsample * duplicate_length + (len(og_betas.index) - 1)
        pre_distribution = upsample * duplicate_length / new_total * 100

        runs += 1
        if runs > 20:
            print('Took too long, adjust upper and lower bounds')
            break
        elif runs > 10:
            print('Taking too long...')
            print('Deviance, upsample, new total , pre distribution:', deviance, upsample, new_total, pre_distribution)

    print(og_betas)
    og_betas = og_betas.append([df_duplicate]*upsample, ignore_index=False)
    print(og_betas)
    percent_distribution = list(og_betas['Cancer Status']).count(1) / len(og_betas.index) * 100
    print('Percent distribution of cancer status after redistribution: ', percent_distribution)

    og_betas.to_csv('data/Upsampled' + str(percent_variance) + '%.csv')

percent_distribution = list(og_betas['Cancer Status']).count(1) / len(og_betas.index) * 100

# Start of overall up-sampling
if overall_upsample:
    overall_upsamples = 2
    df_duplicate, duplicate_length = create_duplicate(to_replicate=og_betas, pct_variance=percent_variance)
    og_betas = og_betas.append([df_duplicate]*2)
    og_betas.to_csv('data/OverallUpsampled' + str(percent_variance) + '%.csv')

og_betas.sort_index(axis=1, inplace=True)
og_betas = og_betas.sample(frac=1)

# Building training, testing, and validation datasets
train_x_labled, test_x_labled, train_y_labled, test_y_labled = train_test_split(og_betas.iloc[:, 1:], og_betas.loc[:, 'Cancer Status'], test_size=.2)
train_x_labled, val_x_labled, train_y_labled, val_y_labled = train_test_split(train_x_labled, train_y_labled, test_size=.2)

scaler = StandardScaler()
train_x, train_y = scaler.fit_transform(np.array(train_x_labled)), np.array(train_y_labled)
test_x, test_y = scaler.transform(np.array(test_x_labled)), np.array(test_y_labled)
val_x, val_y = scaler.transform(np.array(val_x_labled)), np.array(val_y_labled)

print('Train x: ', train_x.shape, '\n', 'Train y: ', train_y.shape, '\n', 'Test x: ',
      test_x.shape, '\n', 'Test y: ', test_y.shape, '\n', 'Val x: ', val_x.shape, '\n', 'Val y: ', val_y.shape)

'''Metrics and Hyperparameters'''
metrics = [keras.metrics.TruePositives(name='tp'),
           keras.metrics.FalsePositives(name='fp'),
           keras.metrics.TrueNegatives(name='tn'),
           keras.metrics.FalseNegatives(name='fn'),
           keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall'),
           keras.metrics.BinaryAccuracy(name='accuracy')]

callback = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                         patience=3,
                                         baseline=.00001,
                                         mode='max')

loss = keras.losses.BinaryCrossentropy()
initial_bias = keras.initializers.Constant(float(np.log((1 / percent_distribution))))

batch_size = 16
learning_rate = .0001
num_layers = 5
epochs = 10
layer_size = 500

opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.99)

'''Model'''
model = keras.Sequential(layers.Dense(len(og_betas.index)-1, activation='relu'))

for layer in range(num_layers):
    model.add(keras.layers.Dense(layer_size, activation='relu'))

output_layer = model.add(keras.layers.Dense(1, activation='sigmoid', bias_initializer=initial_bias))

model.compile(optimizer=opt,
              loss=loss,
              metrics=metrics)

model.fit(x=train_x,
          y=train_y,
          validation_data=[val_x, val_y],
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[callback],
          verbose=1)

validation_output = model.predict(val_x)
for otp in validation_output:
    output = otp[0]
    print(round(output), output)
