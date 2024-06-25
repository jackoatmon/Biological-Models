import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import sklearn
from scipy import stats
import joblib
import warnings
from glob import glob
from datetime import datetime
import os
warnings.filterwarnings('ignore')

dPath = 'C:/Users/jack/PycharmProjects/TruDiagnostic/Chronological Age/'

# drops = ['ChronoAge0.01-0.001-8-100-11-0.3-0.06-0-0.3 Chain Model',
#          'ChronoAge0.01-0.001-8-100-11-0.3-0.01-0-0.3 Chain Model',
#          'ChronoAge0.01-0.001-8-100-11-0.3-0.01-0-0.08 Chain Model',
#          'ChronoAge0.01-0.001-8-100-11-0.15-0.01-0-0.08 Chain Model',
#          'ChronoAge0.005-0.005-8-125-7-0.1-0.02-0.01-0.2',
#          'ChronoAge0.005-0.001-8-125-7-0.2-0-0.01-0.4 Chain Model',
#          'ChronoAge0.01-0.001-8-125-7-0-0.1-0.01-0.3 Chain Model',
#          'ChronoAge0.01-0.001-8-125-7-0-0.02-0.01-0.3 Chain Model',
#          ]
# x = 0
# for f in glob('data/Chunk*'):
#     print(f)
#     # new_end = 'Chunk' + str(int(f.split('Chunk')[-1].replace('.csv', '')) + 1)
#     # new_path = f.replace(f.split('/')[-1], new_end) + '.csv'
#     new_path = f.replace('Chunk', 'Chunk ')
#     print(new_path)
#     try:
#         os.rename(f, new_path)
#     except Exception as e:
#         print(e)
# exit()

dasat = pd.read_csv('TruDeepDasatnibPerformance.csv')

testing = True
validating = True
mode = 'comite'

print('\n\nDATA FORMATTING + DEPENDENCY REQUIREMENTS:')
print('*****************************')
print('o Need the following python packages installed with pip:'
      '\n    o tensorflow'
      '\n    o numpy'
      '\n    o pandas'
      '\n    o sklearn'
      '\n    o warnings\n'
      'o Data must be BMIQ-normalized \n'
      'o Choose "validating" if you have a file with known ages. Column w ages must be labeled as "Decimal.Chronological.Age"\n'
      "o Column w patient ID's should be labeled as " + '"Patient.ID"\n' 
      'o Do not include file extension in the inputs\n'
      'o Patients/individuals on vertical axis, features and CpGs in columns \n'
      'o If validating model accuracy, must include "Chronological Age" in columns \n'
      'o Do not change file names\n')
print('*****************************\n')

# C:\Users\jack\PycharmProjects\TruDiagnostic\Chronological Age\Data\Selected Data\SelectedMethylationData.csv
# C:\Users\jack\PycharmProjects\TruDiagnostic\Chronological Age\Data\Selected Data\RelevantCpGs.csv
# C:\Users\jack\PycharmProjects\TruDiagnostic\Chronological Age\Models\ChronoAge

if not testing:
    good_re = False
    while not good_re:
        validating = input('Are you validating the model or exclusively getting predictions? ("v" for validating/"p" for predicting)').lower()
        if validating == 'v':
            validating = True
            good_re = True
        elif validating == 'p':
            validating = False
            good_re = True
        else:
            print('Invalid response, respond with "validating" or "predicting"')

    valid = False
    while not valid:
        matching = False
        while not matching:
            pred_file = input('\nEnter the name of the file to predict: \n')
            if validating:
                if pred_file in ['Comite+Elastic+Chains', 'ChronAge_test_Comite']:
                    try:
                        age_file = 'Copy of Plasma_PatientMetrics_GMQN'
                        ages = pd.read_csv('data/' + age_file + '.csv').set_index('Patient.ID')['Decimal.Chronological.Age']
                    except Exception as e:
                        print('Error reading age file: ', e)
                    matching = True
                elif pred_file in ['Dasatnib+Elastic+Chains', 'DasatinibQuercitin_GMQN_Blood_Betas']:
                    try:
                        age_file = 'Quercitin_Dasatinib_pdata.txt'
                        ages = pd.read_csv('data/' + age_file + '.csv').set_index('Patient.ID')['Decimal.Chronological.Age']
                        matching = True
                    except Exception as e:
                        print('Error reading age file: ', e)
                elif pred_file == 'Sample250Test_betas.GMQN':
                    try:
                        age_file = 'Sample199_sample_ages'
                        ages = pd.read_csv('data/' + age_file + '.csv').set_index('Patient.ID')['Decimal.Chronological.Age']
                        matching = True
                    except Exception as e:
                        print('Error reading age file: ', e)
                else:
                    try:
                        age_file = input('\nEnter the name of the age file: \n')
                        ages = pd.read_csv('data/' + age_file + '.csv').set_index('Patient.ID')['Decimal.Chronological.Age']
                        matching = True
                    except Exception as e:
                        print('Error reading age file: ', e)
            else:
                matching = True
        try:
            print('Testing file validity and loading file...')
            to_predict = pd.read_csv('data/' + pred_file + '.csv').fillna(0)
            to_predict = to_predict.set_index(to_predict.columns[0])
            print('Total na values: ', to_predict.isna().sum().sum())
            to_predict.dropna(axis=1, inplace=True)
            valid = True
        except Exception as e:
            print('Invalid file, try another file path or adjust contents: ', e)

    cpgs = list(pd.read_csv('ChronoFeatureOrder.csv').set_index('Unnamed: 0'))

    if len(to_predict.columns) < len(to_predict.index):
        to_predict = to_predict.transpose()

    print(to_predict)
    # model_name = 'ChronoAge0.01-125-9-0.1-0.07-0.1'
    model_name = 'ChronoAge0.01-0.00016050942242004044-14.0-108.0-4.0-0.18763341529304034-0.021061051305153403-0.08789363014572782-0.22596500456138013'
    ChronoAge = tf.keras.models.load_model('models/' + model_name)
    reg = joblib.load('BaseElasticChronoAge_0.001 final - Shrunk.joblib')
    # regChain = joblib.load('ChainElasticChronoAge_0.001 NEW.joblib')
    scalerChain = joblib.load('ChainTruDeepScaler.joblib')
    scalerBase = joblib.load('BaseTruDeepScaler - Shrunk.joblib')
    # scaler = joblib.load('BaseTruDeepScaler.joblib')

    columns = pd.read_csv('data/selected data/AllSelectedMethylationData+Age+Elastic Shrunk.csv').set_index('Unnamed: 0').columns
    # feature_order = columns.values.flatten()
    # base_feature_order = pd.read_csv('BaseChronoFeatureOrder.csv').set_index('Unnamed: 0').values.flatten()
    feature_order = np.array(columns).flatten()
    feature_order.sort()
    base_feature_order = list(np.array(pd.read_csv('data/Selected Data/AllSelectedMethylationData+Age+Elastic Shrunk.csv').set_index('Unnamed: 0').columns).flatten())
    base_feature_order.sort()

else:
    if mode == 'comite':
        pred_file = 'selected data/Comite+Elastic+Chains.csv'  # ChronAge_test_Comite
        # pred_file = 'ChronAge_test_Comite.csv'  #
        age_file = 'Copy of Plasma_PatientMetrics_GMQN'
        ages = pd.read_csv(dPath + 'data/' + age_file + '.csv').set_index('Patient.ID')['Decimal.Chronological.Age']
    elif mode == 'dasatnib':
        pred_file = 'selected data/Dasatnib+Elastic+Chains.csv'  # DasatinibQuercitin_GMQN_Blood_Betas
        # pred_file = 'DasatinibQuercitin_GMQN_Blood_Betas.csv'  #
        age_file = 'Quercitin_Dasatinib_pdata.txt'
        ages = pd.read_csv(dPath + 'data/' + age_file + '.csv').set_index('Patient.ID')['Decimal.Chronological.Age']
    elif mode == 'other':
        pred_file = 'selected data/Dasatnib+Elastic+Chains.csv'  # DasatinibQuercitin_GMQN_Blood_Betas
        # pred_file = 'DasatinibQuercitin_GMQN_Blood_Betas.csv'  #
        age_file = 'Quercitin_Dasatinib_pdata.txt'
        ages = pd.read_csv(dPath + 'data/' + age_file + '.csv').set_index('Patient.ID')['Decimal.Chronological.Age']
    elif mode == 'all':
        comite_file = 'selected data/Comite+Elastic+Chains.csv'
        comite_age_file = 'Copy of Plasma_PatientMetrics_GMQN'
        comite_ages = pd.read_csv(dPath + 'data/' + comite_age_file + '.csv').set_index('Patient.ID')['Decimal.Chronological.Age']
        dasatnib_file = 'selected data/Dasatnib+Elastic+Chains.csv'
        dasatnib_age_file = 'Quercitin_Dasatinib_pdata.txt'
        dasatnib_ages = pd.read_csv(dPath + 'data/' + dasatnib_age_file + '.csv').set_index('Patient.ID')['Decimal.Chronological.Age']
        s199_file = 'Sample250Test_betas.GMQN.csv'
        s199_age_file = 'Sample199_sample_ages'
        s199_ages = pd.read_csv(dPath + 'data/' + s199_age_file + '.csv').set_index('Patient.ID')['Decimal.Chronological.Age']

    if mode == 'all':
        feature_order = list(np.array(pd.read_csv('BaseChronoFeatureOrder.csv').set_index('Unnamed: 0').values).flatten())

        dasatnib_predict = pd.read_csv('Data/' + dasatnib_file).set_index('Unnamed: 0').fillna(0)
        comite_predict = pd.read_csv('Data/' + comite_file).set_index('Unnamed: 0').fillna(0)
        s199_predict = pd.read_csv('Data/' + s199_file).set_index('Unnamed: 0').fillna(0)

        if len(comite_predict.columns) < len(comite_predict.index):
            comite_predict = comite_predict.transpose()
        if len(dasatnib_predict.columns) < len(dasatnib_predict.index):
            dasatnib_predict = dasatnib_predict.transpose()
        if len(s199_predict.columns) < len(s199_predict.index):
            s199_predict = s199_predict.transpose()

        # if len(comite_ages.index) > len(comite_predict.index):
        #     comite_ages = comite_ages.transpose()
        # if len(dasatnib_ages.index) > len(dasatnib_predict.index):
        #     dasatnib_ages = dasatnib_ages.transpose()
        # if len(s199_ages.index) > len(s199_predict.index):
        #     s199_ages = s199_ages.transpose()

        shared_patients = (set(comite_predict.index) & set(comite_ages.index))
        comite_predict = comite_predict.loc[shared_patients, feature_order]
        comite_ages = comite_ages.loc[shared_patients]

        shared_patients = (set(dasatnib_predict.index) & set(dasatnib_ages.index))
        dasatnib_predict = dasatnib_predict.loc[shared_patients, feature_order]
        dasatnib_ages = dasatnib_ages.loc[shared_patients]

        shared_patients = (set(s199_predict.index) & set(s199_ages.index))
        s199_order = feature_order.copy()
        s199_order.remove('Elastic Predictions')
        s199_predict = s199_predict.loc[shared_patients, s199_order]
        s199_ages = s199_ages.loc[shared_patients]
    else:
        to_predict = pd.read_csv('Data/' + pred_file).set_index('Unnamed: 0').fillna(0)

        if len(to_predict.columns) < len(to_predict.index):
            to_predict = to_predict.transpose()

        if len(ages.index) > len(to_predict.index):
            ages = ages.transpose()

        shared_patients = (set(to_predict.index) & set(ages.index))
        to_predict = to_predict.loc[shared_patients]
        ages = ages.loc[shared_patients]

    model_name = 'ChronoAge0.01-0.003345894201153873-11.0-124.0-5.0-0.2636001344409017-0.03407934956976158-0.020846184574898615-0.2476097257662218'

    ChronoAge = tf.keras.models.load_model('models/' + model_name)
    reg = joblib.load('BaseElasticChronoAge_0.001 Final - Shrunk.joblib')
    # regChain = joblib.load('ChainElasticChronoAge_0.001 NEW.joblib')
    # scalerChain = joblib.load('TruDeepScaler.joblib')
    scalerBase = joblib.load('BaseTruDeepScaler.joblib')
    scalerChain = joblib.load('ChainTruDeepScaler.joblib')

    columns = pd.read_csv('data/Selected Data/BaseChronoFeatureOrder.csv').set_index('Unnamed: 0').values.flatten()
    # feature_order = columns.values.flatten()
    feature_order = columns
    print(feature_order)
    cpgs = list(columns)

dropChain = ['Chain Output 1', 'Chain Output 2', 'Chain Output 3', 'Chain Output 4', 'Elastic Predictions', 'Chronological Age']
elastic_features = list(feature_order).copy()
to_predict = to_predict[cpgs]
feature_order = list(feature_order)
print(len(feature_order))
feature_order = sorted(feature_order, key=str.lower)


chain_order = feature_order.copy()
for c in dropChain:
    print(c)
    if c in elastic_features:
        elastic_features.remove(c)
        if c != 'Elastic Predictions':
            feature_order.remove(c)
            base_feature_order.remove(c)

print(elastic_features[:3], elastic_features[-3:])
# if 'Chains' in pred_file:
#     base_predict = to_predict[feature_order]
#     chain_predict = to_predict[chain_order]
# else:
#     print(to_predict, len(elastic_features), elastic_features[:4], elastic_features[-3:])
#     base_predict = to_predict[elastic_features]
#     chain_predict = to_predict[elastic_features]

if validating:
    if mode != 'all':
        shared_patients = (set(ages.index) & set(to_predict.index))
        print(shared_patients, ages.index, to_predict.index)
        ages = ages.loc[shared_patients]
        to_predict = to_predict.loc[shared_patients]
        elastic_x = to_predict[elastic_features]

        x, y = to_predict, ages
        preds = reg.predict(elastic_x)
        to_predict['Elastic Predictions'] = preds
        print('Elastic net r-squared: ', reg.score(elastic_x, y))
        print('Elastic net mae: ', sklearn.metrics.mean_absolute_error(y, preds))
    else:
        print(comite_ages, dasatnib_ages, s199_ages)
        print(comite_predict, dasatnib_predict, s199_predict)
        shared_patients = (set(comite_ages.index) & set(comite_predict.index))
        print(shared_patients, comite_ages.index, comite_predict.index)
        comite_ages = comite_ages.loc[shared_patients]
        comite_predict = comite_predict.loc[shared_patients]
        print(comite_predict)
        comite_elastic_x = comite_predict[s199_order]

        shared_patients = (set(dasatnib_ages.index) & set(dasatnib_predict.index))
        print(shared_patients, dasatnib_ages.index, dasatnib_predict.index)
        dasatnib_ages = dasatnib_ages.loc[shared_patients]
        dasatnib_predict = dasatnib_predict.loc[shared_patients]
        dasatnib_elastic_x = dasatnib_predict[s199_order]

        shared_patients = (set(s199_ages.index) & set(s199_predict.index))
        print(shared_patients, s199_ages.index, s199_predict.index)
        s199_ages = s199_ages.loc[shared_patients]
        s199_predict = s199_predict.loc[shared_patients]
        s199_elastic_x = s199_predict[s199_order]

        dasatnib_x, dasatnib_y = dasatnib_predict, dasatnib_ages
        preds = reg.predict(dasatnib_elastic_x)
        dasatnib_predict['Elastic Predictions'] = preds
        print('Dasatnib elastic net score: ', reg.score(dasatnib_elastic_x, dasatnib_y))
        print('Dasatnib elastic net mae: ', sklearn.metrics.mean_absolute_error(dasatnib_y, preds))

        comite_x, comite_y = comite_predict, comite_ages
        preds = reg.predict(comite_elastic_x)
        comite_predict['Elastic Predictions'] = preds
        print('Comite elastic net score: ', reg.score(comite_elastic_x, comite_y))
        print('Comite elastic net mae: ', sklearn.metrics.mean_absolute_error(comite_y, preds))

        s199_x, s199_y = s199_predict, s199_ages
        preds = reg.predict(s199_elastic_x)
        s199_predict['Elastic Predictions'] = preds
        print('S199 elastic net score: ', reg.score(s199_elastic_x, s199_y))
        print('S199 elastic net mae: ', sklearn.metrics.mean_absolute_error(s199_y, preds))
else:
    if len(to_predict.columns) < len(to_predict.index):
        to_predict = to_predict.transpose()
    elastic_x = to_predict[elastic_features]

    # print('To predict and cpg nan count: ', np.sum(to_predict.isna().sum()), np.sum(cpgs.isna().sum()))
    # print('Cols: ', to_predict.loc[cpgs].columns)
    elastic_preds = reg.predict(elastic_x)
    to_predict['Elastic Predictions'] = elastic_preds

    # print('Elastic net accuracy: ', reg.score(elastic_x, ages))
# to_predict.to_csv('data/selected data/AllSelectedMethylationData+Age+Elastic.csv')

# if 'Chains' not in pred_file:
#     model_names = ['ChronoAge0.005-0.005-8-125-7-0.2-0-0.01-0.2',
#                    'ChronoAge0.005-0.005-8-125-7-0-0-0-0.2',
#                    'ChronoAge0.005-0.005-10-150-8-0.1-0-0-0.2',
#                    'ChronoAge0.005-0.005-8-125-7-0-0.02-0.01-0.3'
#                    ]
#     all_predictions = pd.DataFrame(index=to_predict.index)
#
#     pre_chain_order = list(feature_order).copy()
#     dropChain = ['Chain Output 1', 'Chain Output 2', 'Chain Output 3', 'Chain Output 4']
#     for x in dropChain:
#         try:
#             pre_chain_order.remove(x)
#         except Exception as e:
#             print(e)
#     chain_x = scalerBase.transform(to_predict[pre_chain_order])
#
#     x = 0
#     for model_name in model_names:
#         x += 1
#         model = keras.models.load_model('models/' + model_name)
#         predictions = model.predict(chain_x)
#
#         all_predictions['Chain Output ' + str(x)] = predictions
#     to_predict = pd.concat([to_predict, all_predictions], axis=1)

if mode == 'all':
    dasatnib_predict.sort_index(axis=1, inplace=True)
    comite_predict.sort_index(axis=1, inplace=True)
    s199_predict.sort_index(axis=1, inplace=True)
else:
    to_predict.sort_index(axis=1, inplace=True)
'''Beginning of preprocessing'''
# Determines orientation of data, reorganizes if needed, filtering CpGs
if not testing:
    if len(to_predict.columns) < len(to_predict.index):
        to_predict = to_predict.transpose()

if validating:
    cpgs.append('Chronological Age')

# print('FO: ', len(dasatnib_predict.columns), dasatnib_predict.columns[:3], dasatnib_predict.columns[-3:])
print('FO: ', len(feature_order), feature_order[:3], feature_order[-3:])
# chain_predict = to_predict[chain_order]

gridSearch = pd.read_csv('data/performances/GridSearch.csv').set_index('Unnamed: 0')
# model_names = gridSearch.index
# model_names = [x.split('\\')[-1] for x in glob(dPath + 'models/ChronoAge*')][14:]
model_names = ['ChronoAge0.01-0.003345894201153873-11.0-124.0-5.0-0.2636001344409017-0.03407934956976158-0.020846184574898615-0.2476097257662218']


def test_model(m_name):
    try:
        ChronoAge = tf.keras.models.load_model('Models/' + m_name)  # m_name)
        s = scalerBase

        if mode == 'all' and testing:
            m_name = m_name.replace(dPath, '')

            dasatnib_x = dasatnib_predict
            comite_x = comite_predict
            s199_x = s199_predict

            print('\n\n' + m_name)

            if validating:
                dasatnib_y_labeled = dasatnib_ages
                comite_y_labeled = comite_ages
                s199_y_labeled = s199_ages

                dasatnib_x_labeled, predict_y = dasatnib_x, np.array(dasatnib_ages)
                dasatnib_x = s.transform(np.array(dasatnib_x_labeled))

                comite_x_labeled, comite_y = comite_x, np.array(comite_ages)
                comite_x = s.transform(np.array(comite_x_labeled))

                s199_x_labeled, s199_y = s199_x, np.array(s199_ages)
                s199_x = s.transform(np.array(s199_x_labeled))
            else:
                dasatnib_x = s.transform(np.array(dasatnib_x))
                comite_x = s.transform(np.array(comite_x))
                s199_x = s.transform(np.array(s199_x))

            predictionary = {}
            if validating:

                '''Start of Comite Prediction'''
                ChronoAge.evaluate(comite_x, np.array(comite_y_labeled))
                output = ChronoAge.predict(comite_x)

                total_error = 0
                total_squared_error = 0
                total_preds = 0
                net_error = 0
                for idx in range(len(output)):
                    total_preds += 1
                    patient = comite_x_labeled.index[idx]

                    chronoOutput = output[idx][0]
                    real_age = comite_y_labeled[idx]
                    error = chronoOutput - real_age
                    total_error += abs(error)
                    net_error += error
                    total_squared_error += error * error

                    predictionary[patient] = [chronoOutput, real_age, error]

                print('Comite Total samples, patients: ', total_preds, len(np.unique(comite_x_labeled.index)),
                      len(predictionary.keys()))
                print('Comite Mean Absolute Error: ', total_error / total_preds, '(confirm): ',
                      sklearn.metrics.mean_absolute_error(comite_y, output))
                print('Comite MAE Confirmation: ', sklearn.metrics.mean_absolute_error(comite_y, output))
                print('Comite Root Mean Squared Error: ', np.sqrt(total_squared_error / total_preds), '(confirm): ',
                      np.sqrt(sklearn.metrics.mean_squared_error(comite_y, output)))
                print('Comite RMSE Confirmation: ', sklearn.metrics.mean_squared_error(comite_y, output))
                print('Comite R-Squared: ', sklearn.metrics.r2_score(comite_y, output))
                print('Comite Net average error (under/over-predict): ', net_error / len(output))

                '''Start of Dasatnib Prediction'''
                ChronoAge.evaluate(dasatnib_x, np.array(dasatnib_y_labeled))
                output = ChronoAge.predict(dasatnib_x)

                total_error = 0
                total_squared_error = 0
                total_preds = 0
                net_error = 0
                for idx in range(len(output)):
                    total_preds += 1
                    patient = dasatnib_x_labeled.index[idx]

                    chronoOutput = output[idx][0]
                    real_age = dasatnib_y_labeled[idx]
                    error = chronoOutput - real_age
                    total_error += abs(error)
                    net_error += error
                    total_squared_error += error * error

                    predictionary[patient] = [chronoOutput, real_age, error]

                pd.DataFrame(predictionary).to_csv('predictions/Dasatnib' + model_name + '.csv')
                print('Dasatnib Total samples, patients: ', total_preds, len(np.unique(dasatnib_x_labeled.index)),
                      len(predictionary.keys()))
                print('Dasatnib Mean Absolute Error: ', total_error / total_preds, '(confirm): ',
                      sklearn.metrics.mean_absolute_error(dasatnib_y, output))
                print('Dasatnib MAE Confirmation: ', sklearn.metrics.mean_absolute_error(dasatnib_y, output))
                print('Dasatnib Root Mean Squared Error: ', np.sqrt(total_squared_error / total_preds), '(confirm): ',
                      np.sqrt(sklearn.metrics.mean_squared_error(dasatnib_y, output)))
                print('Dasatnib RMSE Confirmation: ', sklearn.metrics.mean_squared_error(dasatnib_y, output))
                print('Dasatnib R-Squared: ', sklearn.metrics.r2_score(dasatnib_y, output))
                print('Dasatnib Net average error (under/over-predict): ', net_error / len(output))

                '''Start of Sample 199 Prediction'''
                ChronoAge.evaluate(s199_x, np.array(s199_y_labeled))
                output = ChronoAge.predict(s199_x)

                total_error = 0
                total_squared_error = 0
                total_preds = 0
                net_error = 0
                for idx in range(len(output)):
                    total_preds += 1
                    patient = s199_x_labeled.index[idx]

                    chronoOutput = output[idx][0]
                    real_age = s199_y_labeled[idx]
                    error = chronoOutput - real_age
                    total_error += abs(error)
                    net_error += error
                    total_squared_error += error * error

                    predictionary[patient] = [chronoOutput, real_age, error]

                pd.DataFrame(predictionary).to_csv('predictions/Sample199' + model_name + '.csv')
                print('Sample 199 Total samples, patients: ', total_preds, len(np.unique(s199_x_labeled.index)),
                      len(predictionary.keys()))
                print('Sample 199 Mean Absolute Error: ', total_error / total_preds, '(confirm): ',
                      sklearn.metrics.mean_absolute_error(s199_y, output))
                print('Sample 199 MAE Confirmation: ', sklearn.metrics.mean_absolute_error(s199_y, output))
                print('Sample 199 Root Mean Squared Error: ', np.sqrt(total_squared_error / total_preds), '(confirm): ',
                      np.sqrt(sklearn.metrics.mean_squared_error(s199_y, output)))
                print('Sample 199 RMSE Confirmation: ', sklearn.metrics.mean_squared_error(s199_y, output))
                print('Sample 199 R-Squared: ', sklearn.metrics.r2_score(s199_y, output))
                print('Sample 199 Net average error (under/over-predict): ', net_error / len(output))

        else:
            m_name = m_name.replace(dPath, '')
            predictions = pd.DataFrame(index=['TruDeep Prediction', 'Real Age', 'Error'])

            predict_x = to_predict
            print('\n\n' + m_name)

            if validating:
                y = ages

                predict_x_labeled, predict_y = predict_x, np.array(ages)
                predict_x = s.transform(np.array(predict_x_labeled))
            else:
                predict_x = s.transform(np.array(predict_x))

            elastic_x = np.array(to_predict[elastic_features])

            if testing:
                accuracy = ChronoAge.evaluate(predict_x, predict_y)
            output = ChronoAge.predict(predict_x).flatten()
            if testing:
                pearson = stats.pearsonr(predict_y, output)
                spearman = stats.spearmanr(predict_y, output)

                reg.score(elastic_x, predict_y)

                net_error = 0
                predictionary = {}
                for idx in range(len(output)):
                    chronoOutput = output[idx]
                    patient = x.index[idx]
                    real_age = predict_y[idx]

                    error = chronoOutput - real_age
                    net_error += error

                    predictionary[patient] = chronoOutput

                    predictions.loc['TruDeep Prediction', patient], predictions.loc['Real Age', patient], predictions.loc['Error', patient] = chronoOutput, real_age, error

                print('Model evaluation accuracy: ', accuracy)
                print('Pearson correlation, p-value: ', pearson[0], ', ', pearson[1])
                print('Spearman correlation: ', spearman)
                print('R-Squared: ', sklearn.metrics.r2_score(predict_y, output))
                print('Net average error (under/over-predict): ', net_error / len(output))


            # print(predictions)
            # print('predictions/ChronoAgePredict' + pred_file.split('/')[-1].split('.')[0] + '.csv')
            # predictions.to_csv('predictions/ChronoAgePredict' + pred_file.split('\\')[-1].split('.')[0] + '.csv')
            predictions.to_csv(
                'predictions/ChronoAgePredict' + pred_file.split('/')[-1].split('.')[0] + model_name + '.csv')
    except Exception as e:
        print(e)


if testing:
    for model_name in model_names:
        test_model(m_name=model_name)
else:
    test_model(m_name=model_name)
