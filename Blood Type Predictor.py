import tensorflow as tf
from tensorflow import keras
from keras import layers
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
import sklearn
import joblib
import category_encoders as ce
import random
import warnings
warnings.filterwarnings('ignore')

'''Notes
# Ensure the distribution of synthetic data matches the distribution of the original data

# Check to make sure that the patients match with their statuses and methylation data **

# Check out/use differential methylation feature selection

'''

test_path = 'D:/ftp/jackraymond/Test/'

'''Customization parameters'''
feat_select_method = 'differential'  # can be differential or all
new_feature_select = True  # whether to run a new feature selection
feat_select_testing = False  # true when developing/adjusting feature selection method/data being used
shap_select = False  # whether to select features using shap
normalize_distrib = False  # whether to run the synthetic data constructor for class distribution normalization
overall_upsample = False  # whether to up-sample the entire dataset with synthetic data
ABO = False  # whether to use the full-spectrum classifier or the ABO-specific classifier
load_model = False  # whether to use existing model or train a new one
percent_variance = 3

'''Data Section'''
og_type = pd.read_csv('data/PatientMetaData_052322.csv', encoding='cp1252').set_index('PatientID')
og_type.sort_index(0, inplace=True)
og_type = og_type[og_type['Blood Type'] != 'Not Sure']['Blood Type']
og_type.dropna(axis=0, inplace=True)
print(og_type)


if new_feature_select:
    read_file = 'data/selected data/Upsampled3%.csv'
    ABO_test_pred_file = 'data/ABO_test1.csv'
    ABO_test2_pred_file = 'data/ABO_test2.csv'
    ABO_test3_pred_file = 'data/ABO_test3.csv'

    og_betas = pd.read_csv(read_file).set_index('Unnamed: 0').dropna(axis=1)
    cpgs = pd.Series(og_betas.drop(columns=['Blood Type']).columns)
    print(cpgs)
    cpgs.to_csv('data/HemoType_CpG_Order.csv')
    cpgs = cpgs.values
    # og_betas = og_betas.transpose()
    # og_betas = og_betas.iloc[:int(len(og_betas.index)*.6)]
    # print(og_betas)
    # og_betas.to_csv('data/selected data/ShrunkBetas.csv')
    # exit()
else:
    if normalize_distrib:
        read_file = 'data/selected data/AllPatientsSelected NEW.csv    '
        # read_file = 'data/selected data/Upsampled' + str(percent_variance) + '%.csv'
    else:
        read_file = 'data/selected data/Upsampled' + str(percent_variance) + '%.csv'
        # read_file = 'data/selected data/AllPatientsSelected NEW.csv    '
    ABO_test_pred_file = 'data/selected data/SelectedABOTest1.csv'
    ABO_test2_pred_file = 'data/selected data/SelectedABOTest2.csv'
    ABO_test3_pred_file = 'data/selected data/SelectedABOTest3.csv'

    og_betas = pd.read_csv(read_file).set_index('Unnamed: 0').dropna(axis=1)
    cpgs = pd.Series(og_betas.drop(columns=['Blood Type']).columns)
    print(cpgs)
    cpgs.to_csv('data/HemoType_CpG_Order.csv')
    cpgs = cpgs.values


val_bTypes = pd.read_csv('data/PatientMetaData New.csv', encoding='cp1252').set_index('PatientID')['Blood Type']

ABO_test = pd.read_csv(ABO_test_pred_file).set_index('Unnamed: 0').fillna(0)
if len(ABO_test.columns) < len(ABO_test.index):
    ABO_test = ABO_test.transpose()
ABO_test = ABO_test[cpgs]
test1_shared = (set(ABO_test.index) & set(val_bTypes[val_bTypes != 'Not Sure'].index))
ABO_test = ABO_test.loc[test1_shared]

ABO_test2 = pd.read_csv(ABO_test2_pred_file).set_index('Unnamed: 0').fillna(0)
if len(ABO_test2.columns) < len(ABO_test2.index):
    ABO_test2 = ABO_test2.transpose()
ABO_test2 = ABO_test2[cpgs]
test2_shared = (set(ABO_test2.index) & set(val_bTypes[val_bTypes != 'Not Sure'].index))
ABO_test2 = ABO_test2.loc[test2_shared]

ABO_test3 = pd.read_csv(ABO_test3_pred_file).set_index('Unnamed: 0').fillna(0)
if len(ABO_test3.columns) < len(ABO_test3.index):
    ABO_test3 = ABO_test3.transpose()
ABO_test3 = ABO_test3[cpgs]
test3_shared = (set(ABO_test3.index) & set(val_bTypes[val_bTypes != 'Not Sure'].index))
ABO_test3 = ABO_test3.loc[test3_shared]

test1_shared = (set(ABO_test.index) & set(val_bTypes.index))
test1_types = val_bTypes.loc[test1_shared]
test2_shared = (set(ABO_test2.index) & set(val_bTypes.index))
test2_types = val_bTypes.loc[test2_shared]
test3_shared = (set(ABO_test3.index) & set(val_bTypes.index))
test3_types = val_bTypes.loc[test3_shared]

print(ABO_test)
print(ABO_test2)
print(ABO_test3)

print('Beta file used: ', read_file)
print(og_betas)

total_na = og_betas.isna().sum()
print('Total na values: ', total_na[total_na > 0])
print('Total rows with nas: ', len(total_na[total_na > 0]))
print('Sum of all nas: ', total_na.sum())


blood_types = ['A Negative',
               'A Positive',
               'B Positive',
               'B Negative',
               'O Positive',
               'O Negative',
               'AB Positive',
               'AB Negative']

rename_pairs = {'AB Negative ': 'AB Negative',
                'AB Postive ': 'AB Positive',
                'B Postive': 'B Positive',
                'O Postive': 'O Positive'}

'''Removing bad blood type values, organizing CpGs'''
unique_types = np.unique(og_type.astype(str))

for to_rename in rename_pairs.keys():
    og_type.replace(to_rename, rename_pairs[to_rename], inplace=True)

print('Original number of CpGs: ', len(og_betas.columns))

'''Removing all patients not shared between blood type data and methylation data'''
all_patients = og_type.index
beta_patients = []
missing_patients = []
for patient in all_patients:
    if patient in og_betas.index:
        beta_patients.append(patient)
        if ABO:
            og_betas.loc[patient, 'Blood Type'] = og_type.loc[patient].replace(' Positive', '').replace(' Negative', '')
        else:
            og_betas.loc[patient, 'Blood Type'] = og_type.loc[patient]
    else:
        missing_patients.append(patient)

print('Patients missing from beta values: ', len(missing_patients))
og_betas = og_betas.loc[beta_patients]
patients = beta_patients
print('Total shared patients: ', len(patients))

pd.Series(patients, index=range(len(patients))).to_csv('Shared patients.csv')

# # Feature selection
if new_feature_select:
    if shap_select:
        pass
    else:
        print('Starting feature selection: ')
        # importances = mutual_info_classif(og_betas.drop(columns=['Blood Type']), og_betas.loc[:, 'Blood Type'].astype(str))
        # feat_importances = pd.Series(importances, og_betas.drop(columns=['Blood Type']).columns)
        feat_importances = pd.read_csv('data/selected data/RelevantCpGs NEW.csv').set_index('Unnamed: 0')
        print(feat_importances)
        average_importance = np.average(feat_importances)
        threshold = average_importance * .9
        print('ooga booga Average relevance, threshold: ', average_importance)

        relevant_cpgs = feat_importances[feat_importances[feat_importances.columns[0]] > threshold]
        print(relevant_cpgs)
        relevant_cpgs.to_csv('data/selected data/RelevantCpGs NEW.csv')

        cpgs_target = list(relevant_cpgs.index)
        cpgs_only = cpgs_target.copy()

        cpgs_target.append('Blood Type')
        og_betas = og_betas[cpgs_target]
        ABO_test = ABO_test[cpgs_only]
        ABO_test2 = ABO_test2[cpgs_only]
        ABO_test3 = ABO_test3[cpgs_only]
        og_betas.to_csv('data/selected data/SelectedMethylationDataBlood NEW.csv')
        ABO_test.to_csv('data/selected data/SelectedABOTest1 NEW.csv')
        ABO_test2.to_csv('data/selected data/SelectedABOTest2 NEW.csv')
        ABO_test3.to_csv('data/selected data/SelectedABOTest3 NEW.csv')
        print('Final Number of CpGs: ', len(cpgs_only))
elif feat_select_testing:
    relevant_cpgs = pd.read_csv('data/selected data/RelevantCpgs2.csv')['Unnamed: 0'].to_list()
    cpgs_target = []
    [cpgs_target.append(cg) for cg in relevant_cpgs if cg in og_betas.columns]
    cpgs_only = cpgs_target.copy()
    cpgs_target.append('Blood Type')

    og_betas = og_betas[cpgs_target]
    ABO_test = ABO_test[cpgs_only]
    ABO_test2 = ABO_test2[cpgs_only]
    ABO_test3 = ABO_test3[cpgs_only]

    importances = mutual_info_classif(og_betas.drop(columns=['Blood Type']), og_betas.loc[:, 'Blood Type'].astype(str))
    feat_importances = pd.Series(importances, og_betas.drop(columns=['Blood Type']).columns)
    average_importance = np.average(feat_importances)
    print('Average relevance: ', average_importance)

    threshold = average_importance
    relevant_cpgs = feat_importances[feat_importances > threshold]
    relevant_cpgs.to_csv('data/selected data/RelevantCpGs2.csv')

    cpgs_target = list(relevant_cpgs.index)
    cpgs_target.append('Blood Type')
    og_betas = og_betas[cpgs_target]
    og_betas.to_csv('data/selected data/SelectedMethylationDataBlood.csv')

    print('Final Number of CpGs: ', len(cpgs_target))

all_types = np.unique(og_type.astype(str))
all_types.sort()
sd_original = og_betas.copy()

# Synthetic data constructor
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

    for feature in to_replicate.columns[1:]:
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

    return df_duplicate, duplicate_length

og_betas.sort_index(axis=1, inplace=True)

# Start of synthetic data distribution normalizer
if normalize_distrib:
    # even = False
    target = 1 / len(all_types) * 100
    adj_target = target + 5
    print('Original number of patients: ', len(og_betas.index))

    # Defining distribution of each blood type
    pre_distribution = {}

    def check_distribution(df_o, individual, blood_type=''):
        for bt in all_types:
            t_total = len(df_o[df_o['Blood Type'] == bt])
            pre_distribution[bt] = t_total / (len(df_o.index) - 1) * 100
        if individual:
            return float(pre_distribution[blood_type])

    check_distribution(df_o=og_betas, individual=False)
    print('Distribution of each blood type before upsampling: \n', pre_distribution)

    # overall_runs = 0
    # while not even:
    # overall_runs += 1
    # print('\nBeginning run number', overall_runs, 'over all blood types\n')

    print(all_types)
    for b in range(len(all_types[:])):
        # bType = all_types[b]
        bType = all_types[b]
        no_upsample = False
            # if even:
            #     break
            # else:
        to_upsample = og_betas['Blood Type'] == bType
        distrib = float(pre_distribution[bType])
        print('Distribution of ' + bType + ' before up-sample: ', distrib)

        if distrib > target-3:
            pass
            print('Blood type ' + bType, ' is at an acceptable distribution, passing...\n')
        else:
            # if overall_runs > 1:
            #     df_duplicate, duplicate_length = create_duplicate(sd_original[to_upsample], percent_variance)
            # else:
            df_duplicate, duplicate_length = create_duplicate(og_betas[to_upsample], percent_variance)

            upsample = 2
            print('Intermediate/final target distribution percentage for', bType, ':', adj_target, '%', '/', target)

            try:
                taking_too_long = False
                runs = 0
                max_runs = 35
                if b < 3:
                    upp = adj_target+4
                    low = adj_target
                elif b < 5:
                    upp = adj_target+1
                    low = adj_target-3
                else:
                    upp = target+3
                    low = target-3
                while upp < distrib or distrib < low:
                    if upsample < 1:
                        no_upsample = True
                        break
                    else:
                        deviance = adj_target - distrib
                        print('Deviance: ', deviance)
                        if 0 < deviance:
                            upsample += 1
                        else:
                            upsample -= 1

                        new_total = upsample*duplicate_length + (len(og_betas.index) - 1)
                        distrib = upsample*duplicate_length / new_total * 100

                        runs += 1
                        if runs > max_runs:
                            print('Took too long, exiting program, adjust the target')
                            upsample = 1
                            break
                        elif runs > 9:
                            print("Taking a while, here's the upsample and distribution: ", upsample, ', ', distrib)
            except Exception as e:
                print('Error in upsample loop: ', e, target, distrib)
                new_total = upsample*duplicate_length + (len(og_betas.index) - 1)
                distrib = upsample * duplicate_length / new_total * 100

            if no_upsample:
                print('Did not up-sample')
            else:
                og_betas = og_betas.append([df_duplicate]*upsample, ignore_index=False)
                post_distrib = check_distribution(individual=True, blood_type=bType, df_o=og_betas)

                print('Up-sampled', bType, 'by following up-sample size:', upsample)
                print('All distributions: ', pre_distribution)
                print('Sum of distributions: ', sum(list(pre_distribution.values())), '%')
                print('Percent distribution of Blood Type', bType, 'after up-sampling: ', post_distrib, '\n')

        # total_even = 0
        # for t in range(len(pre_distribution.keys())):
        #     typ = list(pre_distribution.keys())[t]
        #     if pre_distribution[typ] < .3*float(np.max(list(pre_distribution.values()))):
        #         pass
        #     else:
        #         total_even += 1
        #
        # if total_even == len(pre_distribution.keys()):
        #     even = True
        print('Final number of patients: ', len(og_betas.index))
        print('Final distributions: ', pre_distribution)
        check_distribution(df_o=og_betas, individual=False)
        print('Checking final distribution is correct: ', pre_distribution,
              '\n Sum of distributions:', sum(list(pre_distribution.values())), '%')
        og_betas.to_csv('data/Selected Data/Upsampled' + str(percent_variance) + '% NEW.csv')
else:
    pre_distribution = {}
    for bt in all_types:
        type_total = len(og_betas[og_betas['Blood Type'] == bt].index)
        pre_distribution[bt] = type_total / (len(og_betas.index) - 1) * 100
    print('Distribution of each blood type: \n', pre_distribution)

'''Overall up-sampling'''
# og_betas = og_betas.append([og_betas]*2)
if overall_upsample:
    overall_upsamples = 2
    df_duplicate, duplicate_length = create_duplicate(to_replicate=og_betas, pct_variance=percent_variance)
    og_betas = og_betas.append([df_duplicate]*2)
    og_betas.to_csv('data/selected data/OverallUpsampled' + str(percent_variance) + '% NEW.csv')

# Shuffling of patients to prevent bias, sorting CpGs
og_betas = og_betas.sample(frac=1)
print(og_betas)

'''Building training, testing, validation'''
test_val_size = .003
train_x_labeled, test_x_labeled, train_y_labeled, test_y_labeled = train_test_split(og_betas.drop(columns=['Blood Type']), og_betas.loc[:, 'Blood Type'], test_size=test_val_size)
train_x_labeled, val_x_labeled, train_y_labeled, val_y_labeled = train_test_split(train_x_labeled, train_y_labeled, test_size=test_val_size)

train_d = {}
for typ in all_types:
    dist = len(train_y_labeled[train_y_labeled==typ]) / len(train_y_labeled.index)
    train_d[typ] = str(dist*100) + ' %'
print('Distrib of training data: ', train_d)
test_d = {}
for typ in all_types:
    dist = len(test_y_labeled[test_y_labeled==typ]) / len(test_y_labeled.index)
    test_d[typ] = str(dist*100) + ' %'
print('\nDistrib of testing data: ', test_d)
val_d = {}
for typ in all_types:
    dist = len(val_y_labeled[val_y_labeled==typ]) / len(val_y_labeled.index)
    val_d[typ] = str(dist*100) + ' %'
print('\nDistrib of validation data: ', val_d)

scaler = StandardScaler()
encoder = ce.OneHotEncoder(verbose=True).fit(og_betas['Blood Type'])
# logisticEncoder = LabelEncoder().fit(train_y_labeled.values)

# l_train_y, l_test_y = logisticEncoder.transform(train_y_labeled), logisticEncoder.transform(test_y_labeled)
to_replace = {'A Negative': 0,
              'A Positive': 1,
              'B Positive': 2,
              'B Negative': 3,
              'O Positive': 4,
              'O Negative': 5,
              'AB Positive': 6,
              'AB Negative': 7}

l_train_y, l_test_y = np.array(train_y_labeled.replace(to_replace)), np.array(test_y_labeled.replace(to_replace))
train_y, test_y, val_y = encoder.transform(train_y_labeled), encoder.transform(test_y_labeled), encoder.transform(val_y_labeled)

l_test1_y = np.array(test1_types.replace(to_replace))
test1_y = encoder.transform(test1_types)

l_test2_y = np.array(test2_types.replace(to_replace))
test2_y = encoder.transform(test2_types)

l_test3_y = np.array(test3_types.replace(to_replace))
test3_y = encoder.transform(test3_types)

'''Creating/saving a dictionary of the encoding pattern of the blood types'''
if load_model:
    if ABO:
        bType_encodings = pd.read_csv('data/BloodTypeEncodingABO' + str(percent_variance) + '%.csv').set_index('Unnamed: 0').to_dict('list')
    else:
        bType_encodings = pd.read_csv('data/BloodTypeEncoding' + str(percent_variance) + '%.csv').set_index('Unnamed: 0').to_dict('list')
else:
    bType_encodings = {}
    read_types = []
    for i in range(len(val_y_labeled)):
        bt = val_y_labeled.values[i]
        # print("BLOOD TYPE BABYYYYY", bt)
        enc_bt = val_y.loc[val_y.index[i]]
        if isinstance(enc_bt.iloc[0], pd.Series):
            enc_bt = enc_bt.iloc[0]
        # print('\n', bt, '\n', enc_bt)

        try:
            # print(enc_bt)
            trues = list(enc_bt == 1)
            # print(bt, trues)
            for v in range(len(trues)):
                val = trues[v]
                if val:
                    bt_idx = v
                    break
            if bt not in read_types:
                read_types.append(bt)
                read_types.sort()
                bType_encodings[bt_idx] = bt
                # print(bType_encodings)
            if list(read_types) == list(all_types):
                break

        except Exception as e:
            print('Encoding error: ', e, read_types, bt_idx)
    print('Encodings: ', bType_encodings, '\n')
    if ABO:
        pd.DataFrame(bType_encodings, index=['Blood Type']).to_csv('data/BloodTypeEncodingABO' + str(percent_variance) + '%.csv')
    else:
        pd.DataFrame(bType_encodings, index=['Blood Type']).to_csv('data/BloodTypeEncoding' + str(percent_variance) + '%.csv')

# print(train_x[0], train_y[0])
# print(train_x[0], train_y[0])
# print(train_x[0], train_y[0])

train_x_logistic, train_y = scaler.fit_transform(np.array(train_x_labeled)), np.array(train_y)
test_x_logistic, test_y = scaler.transform(np.array(test_x_labeled)), np.array(test_y)
val_x_logistic, val_y = scaler.transform(np.array(val_x_labeled)), np.array(val_y)

test1_x_logistic = scaler.transform(np.array(ABO_test))
test2_x_logistic = scaler.transform(np.array(ABO_test2))
test3_x_logistic = scaler.transform(np.array(ABO_test3))

# print(train_x[0])
# print(train_y[0])
# print(test_x[0])
# print(test_y[0])
# print(val_x[0])
# print(val_y[0])

'''Logistic Regression Method'''
# clf = sklearn.linear_model.LogisticRegression(penalty='l1', multi_class='auto', solver='liblinear', l1_ratio=.4)
# clf.fit(og_betas.drop(columns=['Blood Type']), og_betas.loc[:, 'Blood Type'])
# joblib.dump(clf, 'models/LogRegBloodType' + str(percent_variance) + '%.joblib')
#
# accuracy = clf.score(test_x_logistic, l_test_y)
# print(test1_x_logistic, l_test1_y)
# t1_accuracy = clf.score(test1_x_logistic, test1_y)
# t2_accuracy = clf.score(test2_x_logistic, test2_y)
# t3_accuracy = clf.score(test3_x_logistic, test3_y)
# print('Training accuracy of logistic regression: ', accuracy)
# print('ABO_test1 accuracy of logistic regression: ', t1_accuracy)
# print('ABO_test2 accuracy of logistic regression: ', t2_accuracy)
# print('ABO_test3 accuracy of logistic regression: ', t3_accuracy)
#
# logistic_preds = clf.predict(og_betas.drop(columns=['Blood Type']))
# t1_logistic_preds = clf.predict(test1_x_logistic)
# t2_logistic_preds = clf.predict(test2_x_logistic)
# t3_logistic_preds = clf.predict(test3_x_logistic)
#
# og_betas['Logistic Predictions'] = logistic_preds
# ABO_test['Logistic Predictions'] = t1_logistic_preds
# ABO_test2['Logistic Predictions'] = t2_logistic_preds
# ABO_test3['Logistic Predictions'] = t3_logistic_preds

# train_x_labeled, test_x_labeled, train_y_labeled, test_y_labeled = train_test_split(og_betas.drop(columns=['Blood Type']), og_betas.loc[:, 'Blood Type'], test_size=test_val_size)
# train_x_labeled, val_x_labeled, train_y_labeled, val_y_labeled = train_test_split(train_x_labeled, train_y_labeled, test_size=test_val_size)

train_x = scaler.transform(np.array(train_x_labeled))
test_x = scaler.transform(np.array(test_x_labeled))
val_x = scaler.transform(np.array(val_x_labeled))

print('Order of training data: ', train_x_labeled)
print('Order of test1: ', ABO_test.columns)
print('Order of test2: ', ABO_test2.columns)
print('Order of test3: ', ABO_test3.columns)

test1_x, test1_y = scaler.transform(np.array(ABO_test)), np.array(test1_y)
test2_x, test2_y = scaler.transform(np.array(ABO_test2)), np.array(test2_y)
test3_x, test3_y = scaler.transform(np.array(ABO_test3)), np.array(test3_y)

print('Train x: ', train_x.shape, '\nTrain y: ', train_y.shape, '\nTest x: ',
      test_x.shape, '\nTest y: ', test_y.shape, '\nVal x: ', val_x.shape, '\nVal y: ', val_y.shape)


'''Metrics and Hyperparameters'''
metrics = [keras.metrics.TruePositives(name='tp'),
           keras.metrics.FalsePositives(name='fp'),
           keras.metrics.TrueNegatives(name='tn'),
           keras.metrics.FalseNegatives(name='fn'),
           keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall'),
           keras.metrics.CategoricalAccuracy(name='accuracy')]

# loss = keras.losses.categorical_crossentropy()
# initial_bias = keras.initializers.Constant(float(np.log((1 / percent_distribution))))

batch_size = 512
epochs = 250

learning_rates = [.01]  # , .001]
min_lrs = [.005, .0001]
patiences = [8, 5]
layer_sizes = [100, 150]  # , 125]
num_layers2 = [6, 9]
noises = [0, .1, .2]  # [.2, .1, .01, .001]
l1s = [0]  # [.2, .1]
l2s = [0]  # [.2, .1]
b1s = [.99]  # [.5, .9, .99]
b2s = [.999]  # [.5, .9, .99]
dropout_rates = [.3, .2]

# df_log = pd.read_csv('data/performances/GridSearchPerformance.csv').set_index('Unnamed: 0')
df_log = pd.DataFrame(
    columns=['Learning Rate', 'Min Learning Rate', 'Patience', 'Layer Size', 'Number of Layers', 'Noise', 'L1', 'L2',
             'Dropout Rate', 'Val_precision', 'Val_Accuracy'])

'''Model'''
if load_model:
    model = keras.models.load_model('models/BloodTypePredictor' + str(percent_variance) + '%')
    validation_output = model.predict(val_x)
else:
    for learning_rate in learning_rates:
        for min_lr in min_lrs:
            for patience in patiences:
                for layer_size in layer_sizes:
                    for num_layers in num_layers2:
                        for noise in noises:
                            for l1 in l1s:
                                for l2 in l2s:
                                    for b1 in b1s:
                                        for b2 in b2s:
                                            for dropout_rate in dropout_rates:
                                                model_name = 'HemoType' + str(learning_rate) + '-' + str(min_lr) \
                                                             + '-' + str(patience) + '-' + str(layer_size) + '-' \
                                                             + str(num_layers) + '-' + str(noise) + '-' + str(l1) \
                                                             + '-' + str(l2) + '-' + str(dropout_rate)
                                                print(model_name)

                                                callbacks = [keras.callbacks.EarlyStopping(monitor='accuracy',
                                                                                           patience=patience,
                                                                                           mode='max',
                                                                                           restore_best_weights=True),
                                                             keras.callbacks.ReduceLROnPlateau(monitor='accuracy',
                                                                                               patience=patience/2,
                                                                                               factor=.50,
                                                                                               verbose=1,
                                                                                               mode='max',
                                                                                               min_lr=min_lr),
                                                             keras.callbacks.ModelCheckpoint(
                                                                 filepath='models/BloodTypePredictor' + str(
                                                                     percent_variance) + '%',
                                                                 monitor='val_accuracy',
                                                                 mode='min',
                                                                 save_freq=30)]

                                                opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=b1, beta_2=b2)
                                                model = keras.Sequential(layers.Dense(len(train_x[0]), activation='relu'))

                                                for layer in range(num_layers):
                                                    model.add(keras.layers.BatchNormalization())
                                                    model.add(keras.layers.GaussianNoise(noise))
                                                    model.add(keras.layers.ActivityRegularization(l1, l2))
                                                    model.add(keras.layers.Dense(layer_size, activation='relu'))
                                                    model.add(keras.layers.Dropout(dropout_rate))

                                                output_layer = model.add(keras.layers.Dense(len(train_y[0]), activation='softmax'))  # , bias_initializer=initial_bias))

                                                model.compile(optimizer=opt,
                                                              loss='categorical_crossentropy',
                                                              metrics=metrics)

                                                model.fit(x=train_x,
                                                          y=train_y,
                                                          validation_data=[val_x, val_y],
                                                          batch_size=batch_size,
                                                          epochs=epochs,
                                                          callbacks=callbacks,
                                                          verbose=1)

                                                mets = model.evaluate(val_x, val_y)

                                                df_log.loc[model_name, ['Learning Rate', 'Min Learning Rate',
                                                                        'Patience', 'Layer Size', 'Number of Layers',
                                                                        'Noise', 'L1', 'L2', 'Dropout Rate',
                                                                        'Val_Precision', 'Val_Accuracy']] = \
                                                    learning_rate, min_lr, patience, layer_size, num_layers, \
                                                    noise, l1, l2, dropout_rate, mets[6], mets[7]

                                                print(df_log.loc[model_name])

                                                df_log.to_csv('data/performances/GridSearch' + model_name + '.csv')

                                                validation_output = model.predict(val_x)

                                                predictionary = {}
                                                accurate = 0
                                                for i in range(len(validation_output)):
                                                    patient = val_y_labeled.index[i]
                                                    predicted_type = validation_output[i]
                                                    real_type = val_y_labeled[patient]

                                                    predictionary[patient] = [predicted_type, real_type, accurate]

                                                print('\n\n*************************')
                                                print('Test 1 performance...')
                                                test1_accuracy = model.evaluate(test1_x, test1_y)
                                                test1_output = model.predict(test1_x).flatten()

                                                # print('Dasatnib logistic r-squared: ', clf.score(test1_y, test1_output))
                                                print('Model evaluation (MSE, MAE): ', test1_accuracy)

                                                print('\n\n*************************')
                                                print('Test 2 performance...')
                                                test2_accuracy = model.evaluate(test2_x, test2_y)
                                                test2_output = model.predict(test2_x).flatten()

                                                # print('Logistic r-squared: ', clf.score(test2_y, test2_output))
                                                print('Model evaluation (MSE, MAE): ', test2_accuracy)

                                                print('\n\n*************************')
                                                print('Test 3 performance...')
                                                test3_accuracy = model.evaluate(test3_x, test3_y)
                                                test3_output = model.predict(test3_x).flatten()

                                                # print('Logistic r-squared: ', clf.score(test3_y, test3_output))
                                                print('Model evaluation (MSE, MAE): ', test3_accuracy)
    validation_output = model.predict(val_x)

    if ABO:
        model.save('models/BloodTypePredictor' + str(percent_variance) + '% ABO')
    else:
        model.save('models/BloodTypePredictor' + str(percent_variance) + '%')


print(bType_encodings)
num_correct = 0
for idx in range(len(validation_output)):
    otp = validation_output[idx]
    if not load_model:
        bTypeOutput = bType_encodings[int(tf.math.argmax(otp))]
    else:
        bTypeOutput = bType_encodings[str(int(tf.math.argmax(otp)))][0]

    patient = val_x_labeled.index[idx]
    if ABO:
        real_type = val_y_labeled[idx].replace(' Positive', '').replace(' Negative', '')
    else:
        real_type = val_y_labeled[idx]
        
    if real_type.lower() == bTypeOutput.lower():
        num_correct += 1
    print('Predicted vs actual for ', patient, ': ', bTypeOutput, 'vs.', real_type)

total_preds = len(validation_output)
true_accuracy = num_correct / total_preds * 100
print('Final validation accuracy: ', true_accuracy)

# explainer = shap.KernelExplainer(model=model, data=val_x)
# shap_values = explainer.shap_values(val_x)
# interactions = explainer
