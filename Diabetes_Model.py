import tensorflow as tf
from tensorflow import keras
from keras import layers
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
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
new_feature_select = False  # whether to run a new feature selection
feat_select_testing = False  # true when developing/adjusting feature selection method/data being used
shap_select = False  # whether to select features using shap
normalize_distrib = True  # whether to run the synthetic data constructor for class distribution normalization
overall_upsample = False  # whether to up-sample the entire dataset with synthetic data
load_model = False  # whether to use existing model or train a new one
new_logReg = True
diabetes_type = 'Type 2 Diabetes'  # ['Type 1 Diabetes', 'Type 2 Diabetes', 'Prediabetes', 'pre diabetes]
percent_variance = 3

'''Data Section'''

og_diabetes = pd.read_csv('data/PatientMetaData_081822.csv', encoding='cp1252').set_index('PatientID').astype(str)['Endocrine Disease']
print('Total number of patient covariate data entries: ', len(og_diabetes.index))
print(np.unique(og_diabetes))

og_diabetes.sort_index(0, inplace=True)
og_diabetes.dropna(inplace=True)

if new_feature_select:
    read_file = "C:/Users/jack/PycharmProjects/TruDiagnostic/Covariate Models/Blood Type/data/BetaMatrix_Funnorm_RCP_normalized_betas_1642081709.csv"  # D:/ftp/jackraymond/Test/
    og_betas = pd.read_csv(read_file, encoding='').set_index('Unnamed: 0')
    og_betas = og_betas.transpose()
else:
    if normalize_distrib:
        read_file = 'data/selected data/SelectedMethylationDataT2D NEW.csv'
    else:
        read_file = 'data/selected data/SelectedMethylationDataT2D NEW.csv'

    og_betas = pd.read_csv(read_file).set_index('Unnamed: 0')
    if len(og_betas.columns) < len(og_betas.index):
        og_betas = og_betas.transpose()

print('Total number of patient entries in beta file: ', len(og_betas.index))

relevant_cpgs = pd.read_csv('data/selected data/RelevantCpGs2.csv')['Unnamed: 0']
og_betas = og_betas[relevant_cpgs]
og_betas.to_csv('data/Selected data/SelectedMethylationDtaT2D NEW.csv')

total_na = og_betas.isna().sum()
print('Total na values: ', total_na[total_na > 0])
print('Total rows with nas: ', len(total_na[total_na > 0]))
print('Sum of all nas: ', total_na.sum())
og_betas.dropna(axis=1, inplace=True)

print('Original number of CpGs: ', len(og_betas.columns))

'''Removing all patients not shared between Diabetes Status data and methylation data'''
all_patients = og_diabetes.index
beta_patients = []
missing_patients = []
for patient in all_patients:
    if patient in og_betas.index:
        beta_patients.append(patient)
        if diabetes_type in str(og_diabetes.loc[patient]):
            og_betas.loc[patient, 'Diabetes Status'] = 1
        else:
            og_betas.loc[patient, 'Diabetes Status'] = 0
    else:
        missing_patients.append(patient)

print('Patients missing from beta values: ', len(missing_patients))
og_betas = og_betas.loc[beta_patients]
patients = beta_patients
print('Total shared patients: ', len(patients))

pd.Series(patients, index=range(len(patients))).to_csv('Shared patients.csv')

# # Feature selection
if new_feature_select:
    print('Starting feature selection: ')
    importances = mutual_info_classif(og_betas.drop(columns=['Diabetes Status']), og_betas['Diabetes Status'].astype(str))
    feat_importances = pd.Series(importances, og_betas.columns[1:])
    average_importance = np.average(feat_importances)
    print('Average relevance: ', average_importance)

    threshold = 3 * average_importance
    relevant_cpgs = feat_importances[feat_importances > threshold]
    relevant_cpgs.to_csv('data/selected data/RelevantCpGs.csv')

    cpgs_target = list(relevant_cpgs.index)
    cpgs_target.append('Diabetes Status')
    og_betas = og_betas[cpgs_target]
    og_betas.to_csv('data/selected data/SelectedMethylationDataT2D.csv')
    print('Final Number of CpGs: ', len(cpgs_target))
elif feat_select_testing:
    relevant_cpgs = pd.read_csv('data/selected data/RelevantCpGs.csv')['Unnamed: 0'].to_list()
    cpgs_target = []
    [cpgs_target.append(cg) for cg in relevant_cpgs if cg in og_betas.columns]
    cpgs_target.append('Diabetes Status')

    og_betas = og_betas[cpgs_target]

    importances = mutual_info_classif(og_betas.drop(columns=['Diabetes Status']), og_betas['Diabetes Status'].astype(str))
    feat_importances = pd.Series(importances, og_betas.columns[1:])
    average_importance = np.average(feat_importances)
    print('Average relevance: ', average_importance)

    threshold = average_importance * 2.5
    relevant_cpgs = feat_importances[feat_importances > threshold]
    relevant_cpgs.to_csv('data/selected data/RelevantCpGs2.csv')

    cpgs_target = list(relevant_cpgs.index)
    cpgs_target.append('Diabetes Status')
    og_betas = og_betas[cpgs_target]
    og_betas.to_csv('data/selected data/SelectedMethylationDataT2D NEW.csv')

    print('Final Number of CpGs: ', len(cpgs_target))


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

    features = list(to_replicate.columns)
    features.remove('Diabetes Status')
    for feature in features:
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

# Start of synthetic data distribution normalizer
if normalize_distrib:
    even = False
    target = 50
    adj_target = target + 5
    print('Original number of patients: ', len(og_betas.index))

    # Defining distribution of each Diabetes Status
    if list(og_betas['Diabetes Status']).count(1) > list(og_betas['Diabetes Status']).count(0):
        to_upsample = 0
    else:
        to_upsample = 1

    df_duplicate = og_betas[og_betas['Diabetes Status'] == to_upsample]
    pre_distribution = len(df_duplicate) / len(og_betas.index)

    # Add variable here
    print('Distribution Diabetes Status before upsampling: \n', pre_distribution)
    upsample = 2

    df_duplicate, duplicate_length = create_duplicate(df_duplicate, pct_variance=percent_variance)

    print(df_duplicate)
    runs = 0
    max_runs = 50

    distrib = pre_distribution
    try:
        taking_too_long = False
        upp = target + 4
        low = target - 4
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
                elif runs > max_runs * .8:
                    print("Taking a while, here's the upsample and distribution: ", upsample, ', ', distrib)
    except Exception as e:
        print('Error in upsample loop: ', e, target, distrib)
        new_total = upsample*duplicate_length + (len(og_betas.index) - 1)
        post_distrib = upsample * duplicate_length / new_total * 100

    og_betas = og_betas.append([df_duplicate]*upsample, ignore_index=False)
    post_distrib = len(og_betas[og_betas['Diabetes Status'] == to_upsample]) / len(og_betas.index)

    print('Final distribution: ', post_distrib)

    og_betas.to_csv('data/Selected Data/Upsampled' + str(percent_variance) + '% NEW.csv')
else:
    post_distrib = len(og_betas[og_betas['Diabetes Status'] == 1]) / len(og_betas.index)
    print('Distribution of each positive T2D factor: \n', post_distrib*100, '%')

'''Overall up-sampling'''
# og_betas = og_betas.append([og_betas]*2)
if overall_upsample:
    overall_upsamples = 2
    df_duplicate, duplicate_length = create_duplicate(to_replicate=og_betas, pct_variance=percent_variance)
    og_betas = og_betas.append([df_duplicate]*2)
    og_betas.to_csv('data/selected data/OverallUpsampled' + str(percent_variance) + '% NEW.csv')

# Shuffling of patients to prevent bias, sorting CpGs
og_betas = og_betas.sample(frac=1)

'''Building training, testing, validation'''
test_val_size = .005

scaler = StandardScaler()

train_x_labeled, test_x_labeled, train_y_labeled, test_y_labeled = train_test_split(og_betas.drop(columns=['Diabetes Status']), og_betas['Diabetes Status'], test_size=test_val_size)
train_x_labeled, val_x_labeled, train_y_labeled, val_y_labeled = train_test_split(train_x_labeled, train_y_labeled, test_size=test_val_size)

train_x, train_y = scaler.fit_transform(np.array(train_x_labeled)).astype(float), np.array(train_y_labeled).astype(float)
test_x, test_y = scaler.transform(np.array(test_x_labeled)).astype(float), np.array(test_y_labeled).astype(float)
val_x, val_y = scaler.transform(np.array(val_x_labeled)).astype(float), np.array(val_y_labeled).astype(float)

joblib.dump(scaler, 'T2DScaler.joblib')

print('Train x: ', train_x.shape, '\nTrain y: ', train_y.shape, '\nTest x: ',
      test_x.shape, '\nTest y: ', test_y.shape, '\nVal x: ', val_x.shape, '\nVal y: ', val_y.shape)
print('Order of x-train: ', train_x_labeled.columns)
print('Train x: ', train_x_labeled, '\nTrain y: ', train_y, '\nTest x: ',
      test_x, '\nTest y: ', test_y, '\nVal x: ', val_x, '\nVal y: ', val_y)

'''Metrics and Hyperparameters'''
metrics = [keras.metrics.TruePositives(name='tp'),
           keras.metrics.FalsePositives(name='fp'),
           keras.metrics.TrueNegatives(name='tn'),
           keras.metrics.FalseNegatives(name='fn'),
           keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall'),
           keras.metrics.BinaryAccuracy(name='accuracy')]


loss = keras.losses.BinaryCrossentropy()  # BinaryCrossentropy() | CategoricalHinge() | SquaredHinge()
initial_bias = keras.initializers.Constant(float(np.log((1 / post_distrib))))

batch_size = 128
epochs = 250

learning_rates = [.01]  #, .001]
min_lrs = [.001, .0001]
patiences = [7]
layer_sizes = [125]  # , 125]
num_layers2 = [7]
noises = [.1, .005, .01]  # [.2, .1, .01, .001]
l1s = [0]  # [.2, .1]
l2s = [0]  # [.2, .1]
b1s = [.9]  # [.5, .9, .99]
b2s = [.99]  # [.5, .9, .99]
dropout_rates = [.4, .5]

df_log = pd.DataFrame(columns=['Learning Rate', 'Min Learning Rate', 'Patience', 'Number of Layers', 'Layer Size', 'Noise', 'L1', 'L2', 'Dropout Rate', 'Val_Recall', 'Val_Accuracy', 'Train Accuracy'])
# df_log = pd.read_csv('data/performance/GridSearch.csv').set_index('Unnamed: 0')

'''Model'''
if load_model:
    # model = keras.models.load_model('models/T2DPredictor' + str(percent_variance) + '%')
    model = keras.models.load_model('models/T2DPredictor0.01-0.01-10-100-5-0.15-0-0-0.1')
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
                                                model_name = 'T2DPredictor' + str(learning_rate) + '-' + str(min_lr)\
                                                             + '-' + str(patience) + '-' + str(layer_size) + '-' + \
                                                             str(num_layers) + '-' + str(noise) + '-' + str(l1) + '-' \
                                                             + str(l2) + '-' + str(dropout_rate)
                                                print(model_name)

                                                callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                                                                           patience=patience,
                                                                                           mode='min',
                                                                                           restore_best_weights=True),
                                                             keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                                               patience=patience/2,
                                                                                               factor=.5,
                                                                                               verbose=1,
                                                                                               mode='min',
                                                                                               min_lr=min_lr),
                                                             keras.callbacks.ModelCheckpoint(filepath='models/' + model_name,
                                                                                             monitor='accuracy',
                                                                                             mode='min',
                                                                                             save_freq=100,
                                                                                             save_best_only=True)]

                                                opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=b1, beta_2=b2)
                                                model = keras.Sequential(layers.Dense(len(train_x[0]), activation='relu'))

                                                for layer in range(num_layers):
                                                    model.add(keras.layers.BatchNormalization())
                                                    model.add(keras.layers.GaussianNoise(noise))
                                                    # model.add(keras.layers.ActivityRegularization(l1, l2))
                                                    model.add(keras.layers.Dense(layer_size, activation='relu'))
                                                    model.add(keras.layers.Dropout(dropout_rate))

                                                output_layer = model.add(keras.layers.Dense(1, activation='sigmoid', bias_initializer=initial_bias))

                                                model.compile(optimizer=opt,
                                                              loss='binary_crossentropy',  # hinge_loss
                                                              metrics=metrics)

                                                model.fit(x=train_x,
                                                          y=train_y,
                                                          validation_data=[val_x, val_y],
                                                          batch_size=batch_size,
                                                          epochs=epochs,
                                                          callbacks=callbacks,
                                                          verbose=1)

                                                validation_output = model.predict(val_x)

                                                model.save('models/' + model_name)

                                                mets = model.evaluate(val_x, val_y)

                                                print(mets)
                                                df_log.loc[
                                                    model_name, ['Learning Rate', 'Min Learning Rate', 'Patience', 'Number of Layers', 'Layer Size', 'Noise', 'L1', 'L2', 'Dropout Rate', 'Val_Recall', 'Val_Accuracy']] = \
                                                        learning_rate, min_lr, patience, num_layers, layer_size, noise, l1, l2, dropout_rate, mets[6], mets[7]

                                                df_log.to_csv('data/performances/GridSearch.csv')

                                                predictionary = {}
                                                accurate = 0
                                                for i in range(len(validation_output)):
                                                    patient = val_y_labeled.index[i]
                                                    predicted_type = validation_output[i]
                                                    real_type = val_y_labeled[patient]

                                                    predictionary[patient] = [predicted_type, real_type, accurate]

                                                # print('\n\n*************************')
                                                # print('Test 1 performance...')
                                                # test1_accuracy = model.evaluate(test1_x, test1_y)
                                                # test1_output = model.predict(test1_x).flatten()
                                                #
                                                # print('Model evaluation (MSE, MAE): ', test1_accuracy)


num_correct = 0
false_positives = 0
false_negatives = 0
for idx in range(len(validation_output)):
    output = validation_output[idx]

    threshold = .60
    if output > threshold:
        output = 1
    else:
        output = 0

    patient = val_x_labeled.index[idx]
    real_type = val_y_labeled[idx]

    if real_type == output:
        num_correct += 1
    else:
        if output == 1:
            false_positives += 1
        else:
            false_negatives += 1
    print('Predicted vs actual for ', patient, ': ', output, 'vs.', real_type)

print('\n\n*************************')
# print('Test 1 performance...')
# test1_accuracy = model.evaluate(test1_x, test1_y)
# test1_output = model.predict(test1_x).flatten()

# print('Model evaluation (MSE, MAE): ', test1_accuracy)

total_preds = len(validation_output)
true_accuracy = num_correct / total_preds * 100
print('Final validation accuracy: ', true_accuracy)
print('False negatives: ', false_negatives, '\nFalse positives: ', false_positives)

# mets = model.evaluate(val_x, val_y)
#
# df_log.loc[model_name, ['Learning Rate', 'Number of Layers', 'Layer Size', 'Noise', 'L1', 'L2', 'Dropout Rate', 'Val_Recall', 'Val_Accuracy']] = learning_rate, layer_size, num_layers, noise, l1, l2, mets[7], mets[8]
# df_log.to_csv('data/performances/GridSearch NEW NEW.csv')

# explainer = shap.KernelExplainer(model=model, data=val_x)
# shap_values = explainer.shap_values(val_x)
# shap.force_plot(explainer.expected_value, shap_values, features=train_x_labeled, feature_names=train_x.columns)
# pd.DataFrame(shap_values).to_csv('data/SHAP_Output_' + str(percent_variance) + '%.csv')
# print(shap_values)
