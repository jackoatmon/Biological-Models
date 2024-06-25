import tensorflow as tf
from tensorflow import keras
from keras import layers
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from glob import glob
import sklearn
import joblib
import category_encoders as ce
import random
from hyperopt import STATUS_OK, hp, tpe, fmin, Trials
import warnings
import os
warnings.filterwarnings('ignore')


test_path = 'D:/ftp/jackraymond/Test/'

'''Customization parameters'''
new_feature_select = True
overall_upsample = False
load_model = False
new_elastic = True
run_covariates = False
use_all_betas = False
ensemble = False
chaining = False
new_chain = False
auto_tune = False


os.chdir('/home/varundwaraka/trudeep_10k/')

def train_model(f='', com_file='', dasat_file='', ensemble_train=False, chunk_predictions={}, d_chunks={}, c_chunks={}, c_name='', c=pd.DataFrame(), d=pd.DataFrame()):
    if ensemble_train:
           beta_file = f
    else:
        if new_elastic:
            # beta_file = 'selected data/AllSelectedMethylationData+Age.csv'  # 'SelectedMethylationData NEW.csv'
            beta_file = 'Betas_RCPFunnorm_Transposed_cleaned.csv'
        else:
            if new_chain and chaining:
                beta_file = 'AllSelectedMethylationData+Age+Elastic.csv'
            elif chaining:
                beta_file = 'AllSelectedMethylationData+Age+Elastic+Chains.csv'
            else:
                beta_file = 'selected data/AllSelectedMethylationData+Age.csv'
                # beta_file = 'Betas_RCPFunnorm_Transposed_cleaned.csv'

    for beta_file in ['Betas_Section_Samplewise1.csv', 'Betas_Section_Samplewise2.csv', 'Betas_Section_Samplewise3.csv']:
        print('Loading beta file... (n) ...', beta_file)
        if ensemble_train:
            og_betas = pd.read_csv('data/' + beta_file).set_index('Unnamed: 0').dropna(axis=1)
        else:
            og_betas = pd.read_csv('' + beta_file).dropna(axis=1)
            if new_feature_select:
                if len(og_betas.columns) < len(og_betas.index):
                    og_betas = og_betas.transpose()

            og_betas.set_index(og_betas.columns[0], inplace=True)

        print('Finished loading beta file: ', beta_file)
        print('Original betas dataframe: ', og_betas)

        if run_covariates:
            validation_set = pd.read_csv('TruDiagnostic_ValidationCohort_Population_093022.FunnormRCP.csv')
            covariates = pd.read_csv('data/TruD_10k_Full_Covariates_Filtered.csv').set_index('Patient.ID')
            if len(covariates.index) < len(covariates.columns):
                covariates = covariates.transpose()
            shared = list(set(covariates) & set(og_betas.index))
            covariates, og_betas = covariates.loc[shared], og_betas.loc[shared]
            print('Covariates dataframe: ', covariates)

            og_betas = pd.concat([og_betas, covariates], axis=1)

        dropChain = ['Chain Output 1', 'Chain Output 2', 'Chain Output 3', 'Chain Output 4']

        # if ensemble_train:
        #     dasat_pred_file = dasat_file
        #     comite_pred_file = com_file
        # else:
        #     if new_chain and chaining:
        #         comite_pred_file = 'data/selected data/Comite+Elastic.csv'
        #         dasat_pred_file = 'data/selected data/Dasatnib+Elastic.csv'
        #     else:
        #         comite_pred_file = 'data/selected data/Comite+Elastic+Chains.csv'
        #         dasat_pred_file = 'data/selected data/Dasatnib+Elastic+Chains.csv'

        if new_elastic:
            dropCols = ['Chronological Age']
        else:
            dropCols = ['Chronological Age', 'Elastic Predictions']


        age_file = 'PopulationData_100422.csv'
        og_ages = pd.read_csv('data/' + age_file).set_index('Patient.ID')['Decimal.Chronological.Age'].dropna()

        if ensemble and ensemble_train:
            shared_patients = (set(og_betas.index) & set(og_ages.index))
            for patient in shared_patients:
                og_betas.loc[patient, 'Chronological Age'] = og_ages[patient]

        og_betas = og_betas[~og_betas['Chronological Age'].isna()]
        og_betas.sort_index(axis=1, inplace=True)

        if overall_upsample:
            og_betas = og_betas.append([og_betas] * 2)

        if new_feature_select:
            mutual = True
            print('Starting feature selection: ')
            if mutual:
                importances = mutual_info_regression(og_betas.iloc[:, 1:], og_betas.loc[:, 'Chronological Age'].astype(float))
                feat_importances = pd.Series(importances, og_betas.columns[1:])
            else:
                feat_importances = pd.DataFrame(columns=['p-Value'])
                for col in og_betas.columns:
                    corr, pVal = stats.pearsonr(og_betas[col], og_betas['Chronological Age'])
                    feat_importances.loc[col] = pVal
            feat_importances = pd.read_csv('data/selected data/RelevantCpGs.csv')
            feat_importances = feat_importances.set_index(feat_importances.columns[0]).astype(float)

            average_importance = np.average(feat_importances.values)
            print('Average relevance: ', average_importance)

            threshold = 2.5 * average_importance
            relevant_cpgs = feat_importances[feat_importances['0'] > threshold]
            if mutual:
                relevant_cpgs.to_csv('data/selected data/RelevantCpGs' + beta_file + '.csv')
            else:
                relevant_cpgs.to_csv('data/selected data/p-valRelevantCpGs.csv')

            cpgs_target = list(relevant_cpgs.index)
            cpgs_target.append('Chronological Age')
            if not new_elastic:
                cpgs_target.append('Elastic Predictions')
            og_betas = og_betas[cpgs_target]
            og_betas.sort_index(axis=1, inplace=True)
            og_betas.to_csv('data/selected data/SelectedMethylationData' + beta_file + ' NEW.csv')
            print('Final Number of CpGs: ', len(cpgs_target))

        if new_elastic:
            feature_order = list(og_betas.columns)
            if 'Chronological Age' in feature_order:
                feature_order.remove('Chronological Age')
            elastic_features = list(feature_order).copy()
            for c in dropChain:
                try:
                    elastic_features.remove(c)
                except Exception as e:
                    print('Elastic feature remove error: ', e)

            if 'Chronological Age' in elastic_features:
                elastic_features.remove('Chronological Age')

            print('Lengths: ', len(og_betas.columns), ', ', len(elastic_features))
            elastic_x, elastic_y = np.array(og_betas[elastic_features]), np.array(og_betas['Chronological Age'])

        else:
            feature_order = list(og_betas.drop(columns=['Chronological Age']).columns)

            if ensemble:
                for x in dropChain:
                    try:
                        feature_order.remove(x)
                    except Exception as e:
                        print(e)

            print(og_betas, og_betas.columns[:3], og_betas.columns[-4:])
            if not ensemble:
                elastic_features = list(og_betas.drop(columns=dropCols).columns)
                pd.Series(elastic_features).to_csv('data/ElasticFeatureOrder' + beta_file + '.csv')
            else:
                elastic_features = list(og_betas.columns)
                if 'Chronological Age' in elastic_features:
                    elastic_features.remove('Chronological Age')
                if not ensemble_train:
                    pd.Series(elastic_features).to_csv('data/EnsembleElasticFeatureOrder' + beta_file + '.csv')

            if not new_chain and not ensemble:
                for x in dropChain:
                    try:
                        elastic_features.remove(x)
                    except:
                        pass

            elastic_x, elastic_y = np.array(og_betas[elastic_features]), np.array(og_betas['Chronological Age'])

        if not ensemble:
            # og_betas = og_betas.sample(frac=1)
            test_val_size = .003

            train_x_labeled, test_x_labeled, train_y_labeled, test_y_labeled = train_test_split(
                og_betas.drop(columns=['Chronological Age']), og_betas.loc[:, 'Chronological Age'], test_size=test_val_size)

            scaler = StandardScaler()

            train_x, train_y = scaler.fit_transform(np.array(og_betas.drop(columns=['Chronological Age']))), np.array(train_y_labeled)

            if chaining and new_chain:
                joblib.dump(scaler, 'BaseTruDeepScaler - Shrunk.joblib')
            else:
                joblib.dump(scaler, 'BaseTruDeepScaler - Shrunk.joblib')

            print('Starting ElasticNet...')
            if new_elastic:
                if use_all_betas:
                    print('Reading big file...')
                    elastic_betas = pd.read_csv('C:/Users/jack/PycharmProjects/TruDiagnostic/Covariate Models/Blood Type/Data/BetaMatrix_Funnorm_RCP_normalized_betas_1642081709.csv').dropna(axis=0)
                    print('Transposing...')
                    elastic_betas = elastic_betas.transpose()
                    elastic_betas.set_index(elastic_betas.columns[0], inplace=True)
                    elastic_betas = elastic_betas.loc[og_betas.index]
                else:
                    elastic_betas = og_betas

                print(elastic_betas[elastic_features].isna().sum().sum(), elastic_betas['Chronological Age'].isna().sum())
                x, y = elastic_betas[elastic_features], elastic_betas['Chronological Age']
                pd.Series(x.columns).to_csv('data/ElasticChronoAge.csv')

                alpha_vals = [.001]
                all_scores = {}
                for a in alpha_vals:
                    print('Order of cpgs for elasticnet: ', x.columns)
                    reg = sklearn.linear_model.ElasticNet(alpha=a, max_iter=10000)
                    reg.fit(x, y)
                    all_scores[a] = reg.score(x, y)
                    if chaining:
                        joblib.dump(reg, 'ChainElasticChronoAge_' + str(a) + ' NEW.joblib')
                    else:
                        joblib.dump(reg, 'BaseElasticChronoAge_' + str(a) + ' NEW - Shrunk.joblib')
                print('Alpha values and elastic performances: ', all_scores)
                reg = sklearn.linear_model.ElasticNet(alpha=list(all_scores.keys())[list(all_scores.values()).index(np.max(list(all_scores.values())))], max_iter=10000)
                reg.fit(x, y)
            else:
                alpha_val = 0.001
                if new_chain and chaining:
                    reg = joblib.load('BaseElasticChronoAge_' + str(alpha_val) + ' Final - Shrunk.joblib')
                elif not chaining:
                    reg = joblib.load('BaseElasticChronoAge_' + str(alpha_val) + ' Final - Shrunk.joblib')
                else:
                    reg = joblib.load('ChainElasticChronoAge_' + str(alpha_val) + ' Final - Shrunk.joblib')

            if chaining and new_chain:
                model_names = ['models/ChronoAge0.005-0.005-8-125-7-0.2-0-0.01-0.2',
                               'models/ChronoAge0.005-0.005-8-125-7-0-0-0-0.2',
                               'models/ChronoAge0.005-0.005-10-150-8-0.1-0-0-0.2',
                               'models/ChronoAge0.005-0.005-8-125-7-0-0.02-0.01-0.3'
                               ]
                all_predictions = pd.DataFrame(index=og_betas.index)

                rn = 0
                for model_name in model_names:
                    rn += 1
                    model = keras.models.load_model(model_name)
                    predictions = model.predict(train_x)

                    all_predictions['Chain Output ' + str(rn)] = predictions

                og_betas = pd.concat([og_betas, all_predictions], axis=1)

                if not ensemble:
                    og_betas.to_csv('data/selected data/AllSelectedMethylationData+Age+Elastic+Chains.csv')

                train_x_labeled, test_x_labeled, train_y_labeled, test_y_labeled = train_test_split(
                    og_betas.drop(columns=['Chronological Age']), og_betas.loc[:, 'Chronological Age'], test_size=test_val_size)


            if new_elastic:
                print('Scoring the model...')

                elastic_preds = reg.predict(elastic_x)

                del elastic_betas

                if not ensemble:
                    og_betas['Elastic Predictions'] = elastic_preds
                    og_betas.to_csv('data/selected data/AllSelectedMethylationData+Age+Elastic.csv')

                train_x_labeled, test_x_labeled, train_y_labeled, test_y_labeled = train_test_split(
                    og_betas.drop(columns=['Chronological Age']), og_betas.loc[:, 'Chronological Age'], test_size=test_val_size)

            if chaining:
                scaler = StandardScaler()
                train_x, train_y = scaler.fit_transform(np.array(train_x_labeled)), np.array(train_y_labeled)
                joblib.dump(scaler, 'ChainTruDeepScaler.joblib')
            else:
                print(train_x_labeled)
                scaler = StandardScaler()
                train_x, train_y = scaler.fit_transform(np.array(train_x_labeled)), np.array(train_y_labeled)
                joblib.dump(scaler, 'BaseTruDeepScaler - Shrunk.joblib')

            og_betas = og_betas.sort_index(axis=1)
            all_x, all_y = scaler.transform(og_betas.drop(columns=['Chronological Age'])), og_betas.loc[:, 'Chronological Age']
            test_x, test_y = scaler.transform(np.array(test_x_labeled)), np.array(test_y_labeled)

            print('Final order of cpgs: ', train_x_labeled.columns)
            if chaining and new_chain:
                pd.Series(train_x_labeled.columns).to_csv('BaseChronoFeatureOrder.csv')
            elif chaining and not new_chain and not ensemble:
                pd.Series(train_x_labeled.columns).to_csv('ChainChronoFeatureOrder.csv')
            else:
                pd.Series(train_x_labeled.columns).to_csv('BaseChronoFeatureOrder' + beta_file + '.csv')

            print('Train x: ', train_x.shape, '\nTrain y: ', train_y.shape, '\nTest x: ',
                  test_x.shape, '\nTest y: ', test_y.shape)

            elastic_preds = reg.predict(elastic_x)

            print('Training r-squared of elastic net: ', reg.score(np.array(elastic_x), elastic_y))
            print('Training MAE of elastic net: ', sklearn.metrics.mean_absolute_error(elastic_y, elastic_preds))  # reg.score(og_betas.drop(columns=['Elastic Predictions', 'Chronological Age']), og_betas.loc[:, 'Chronological Age']))

        else:
            if ensemble_train:
                cols = list(og_betas.columns)
                cols.remove('Chronological Age')
            else:
                chunks_df = pd.DataFrame(index=og_betas.index)
                for chunk in chunk_predictions.keys():
                    chunks_df[chunk] = chunk_predictions[chunk]

            test_val_size = .003
            train_x_labeled, test_x_labeled, train_y_labeled, test_y_labeled = train_test_split(
                og_betas.drop(columns=['Chronological Age']), og_betas.loc[:, 'Chronological Age'], test_size=test_val_size)

            scaler = StandardScaler()
            all_x, all_y = scaler.fit_transform(og_betas.drop(columns=['Chronological Age'])), og_betas.loc[:, 'Chronological Age']
            train_x, train_y = scaler.transform(train_x_labeled), np.array(train_y_labeled)
            test_x, test_y = scaler.transform(test_x_labeled), np.array(test_y_labeled)

            if ensemble_train:
                scaler_path = c_name + 'EnsembleAgeScaler.joblib'
                joblib.dump(scaler, scaler_path)
            elif chaining:
                scaler_path = 'ChainEnsembleAgeScaler.joblib'
                joblib.dump(scaler, scaler_path)
            else:
                scaler_path = 'EnsembleAgeScaler.joblib'
                joblib.dump(scaler, scaler_path)
            print('Beta path, scaler path, model path')

        patient_order = og_betas.index
        print('Overall patient ordrer: ', patient_order)

        '''Metrics and Hyperparameters'''
        metrics = [keras.metrics.MeanAbsoluteError(name='MAE'),
                   keras.metrics.MeanSquaredError(name='MSE')]

        batch_size = 256
        epochs = 250

        if ensemble_train:
            learning_rates = [.01]  # , .001]
            min_lrs = [.002]
            patiences = [8]
            layer_sizes = [93]  # , 125]
            num_layers2 = [5]
            noises = [.08]  # [.2, .1, .01, .001]
            l1s = [.034]  # [.2, .1]
            l2s = [.085]  # [.2, .1]
            b1s = [.99]  # [.5, .9, .99]
            b2s = [.999]  # [.5, .9, .99]
            dropout_rates = [.02]
        else:   # Current best: ChronoAge0.01-0.0019247923925306844-8.0-93.0-5.0-0.05635867926717027-0.034533334567416266-0.08505781598392871-0.011274876020738067
            learning_rates = [.01]  # , .001]
            min_lrs = [.001, .0005]
            patiences = [14]
            layer_sizes = [125]  # , 125]
            num_layers2 = [7]  # 7 current best
            noises = [.4]  # .4 current best
            l1s = [.005]  # .005 current best
            l2s = [0.001]  # .01 current best
            b1s = [.99]  # [.5, .9, .99]
            b2s = [.999]  # [.5, .9, .99]
            dropout_rates = [.3]   # .3 current best
        # df_log = pd.read_csv('data/performances/GridSearchPerformance.csv').set_index('Unnamed: 0')
        df_log = pd.DataFrame(
            columns=['Learning Rate', 'Min Learning Rate', 'Patience', 'Layer Size', 'Number of Layers', 'Noise', 'L1', 'L2',
                     'Dropout Rate', 'Val_RMSE', 'Val_MAE'])

        if load_model:
            model_name = 'ChronoAge0.005-125-9-0.01-0.07-0.1'
            model = keras.models.load_model('models/' + model_name)
        else:
            if ensemble_train and ensemble_train or not ensemble:
                if auto_tune:
                    space = {'min_lr': hp.uniform('min_lr', .0001, .05),
                             'patience': hp.quniform('patience', 8, 15, 1),
                             'layer_size': hp.quniform('layer_size', 80, 150, 1),
                             'num_layers': hp.quniform('num_layers', 4, 8, 1),
                             'noise': hp.uniform('noise', .01, .4),
                             'l1': hp.uniform('l1', 0, .09),
                             'l2': hp.uniform('l2', 0, .09),
                             'dropout_rate': hp.uniform('dropout_rate', .01, .4)}
                    hyper_algorithm = tpe.suggest

                    def tuning_objective(hyperparameters={}):
                        min_lr, patience, layer_size, num_layers, noise, l1, l2, dropout_rate = hyperparameters['min_lr'], \
                                                                                                hyperparameters['patience'],\
                                                                                                hyperparameters['layer_size'],\
                                                                                                hyperparameters['num_layers'],\
                                                                                                hyperparameters['noise'], \
                                                                                                hyperparameters['l1'], \
                                                                                                hyperparameters['l2'], \
                                                                                                hyperparameters['dropout_rate']
                        model_name = 'ChronoAge' + str(.01) + '-' + str(min_lr) \
                                     + '-' + str(patience) + '-' + str(layer_size) + '-' \
                                     + str(num_layers) + '-' + str(noise) + '-' + str(l1) \
                                     + '-' + str(l2) + '-' + str(dropout_rate)
                        if chaining:
                            model_name = model_name + ' Chain Model'
                        if ensemble_train:
                            model_name = model_name + ' Ensemble'
                            file_path = 'chunk models/' + model_name + c_name
                        else:
                            file_path = 'models/' + model_name
                        print(model_name)

                        callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                                                   patience=patience,
                                                                   mode='min',
                                                                   restore_best_weights=True),
                                     keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                       patience=int(patience / 2),
                                                                       factor=.5,
                                                                       verbose=1,
                                                                       mode='min',
                                                                       min_lr=min_lr), ]

                        opt = keras.optimizers.Adam(learning_rate=.01, beta_1=.99,
                                                    beta_2=.999)

                        mod = keras.Sequential(layers.Dense(len(train_x[0]), activation='relu'))

                        for layer in range(int(num_layers)):
                            mod.add(keras.layers.BatchNormalization())
                            mod.add(keras.layers.GaussianNoise(noise))
                            mod.add(keras.layers.ActivityRegularization(l1=l1, l2=l2))
                            mod.add(keras.layers.Dense(layer_size, activation='relu'))
                            mod.add(keras.layers.Dropout(dropout_rate))

                        output_layer = mod.add(keras.layers.Dense(1))  # , bias_initializer=initial_bias))

                        mod.compile(optimizer=opt,
                                      loss='mean_squared_error',
                                      metrics=metrics)

                        mod.fit(x=train_x,
                                  y=train_y,
                                  validation_data=[test_x, test_y],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  verbose=1)

                        mod.save(file_path)
                        print('Save path of model: ', file_path)

                        # loss = mod.evaluate(dasat_x, dasat_y)
                        loss = mod.evaluate(test_x, test_y)
                        # loss = mod.evaluate(validation_x, validation_y)

                        print('Parameters: ', min_lr, patience, layer_size, num_layers, noise, l1, l2, dropout_rate)
                        print('Loss: ', loss, '\n')

                        return {'Loss': loss, 'Params': hyperparameters, 'Status': STATUS_OK}

                    best = fmin(fn=tuning_objective, space=space, algo=tpe.suggest,
                                max_evals=200, trials=Trials())

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
                                                                model_name = 'ChronoAge' + str(learning_rate) + '-' + str(min_lr)\
                                                                             + '-' + str(patience) + '-' + str(layer_size) + '-' \
                                                                             + str(num_layers) + '-' + str(noise) + '-' + str(l1)\
                                                                             + '-' + str(l2) + '-' + str(dropout_rate)
                                                                if chaining:
                                                                    model_name = model_name + ' Chain Model'
                                                                if ensemble_train:
                                                                    model_name = model_name + ' Ensemble'
                                                                    file_path = 'chunk models/' + model_name + c_name
                                                                else:
                                                                    file_path = 'models/' + model_name
                                                                print(model_name)

                                                                callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                                                                                           patience=patience,
                                                                                                           mode='min',
                                                                                                           restore_best_weights=True),
                                                                             keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                                                               patience=int(patience/2),
                                                                                                               factor=.5,
                                                                                                               verbose=1,
                                                                                                               mode='min',
                                                                                                               min_lr=min_lr),]
                                                                             # keras.callbacks.ModelCheckpoint(
                                                                             #     filepath=file_path,
                                                                             #     monitor='MAE',
                                                                             #     mode='min',
                                                                             #     save_freq=100)]

                                                                opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=b1,
                                                                                            beta_2=b2)

                                                                model = keras.Sequential(layers.Dense(len(train_x[0]), activation='relu'))

                                                                for layer in range(num_layers):
                                                                    model.add(keras.layers.BatchNormalization())
                                                                    model.add(keras.layers.GaussianNoise(noise))
                                                                    model.add(keras.layers.ActivityRegularization(l1=l1, l2=l2))
                                                                    model.add(keras.layers.Dense(layer_size, activation='relu'))
                                                                    model.add(keras.layers.Dropout(dropout_rate))

                                                                output_layer = model.add(keras.layers.Dense(1))  # , bias_initializer=initial_bias))

                                                                model.compile(optimizer=opt,
                                                                              loss='mean_squared_error',
                                                                              metrics=metrics)

                                                                model.fit(x=train_x,
                                                                          y=train_y,
                                                                          validation_data=[test_x, test_y],
                                                                          batch_size=batch_size,
                                                                          epochs=epochs,
                                                                          callbacks=callbacks,
                                                                          verbose=1)

                                                                model.save(file_path)

                                                                print('Save path of model: ', file_path)

                                                                mets = model.evaluate(test_x, test_y)

                                                                df_log.loc[model_name, ['Learning Rate', 'Min Learning Rate',
                                                                                        'Patience', 'Layer Size', 'Number of Layers',
                                                                                        'Noise', 'L1', 'L2', 'Dropout Rate',
                                                                                        'Val_RMSE', 'Val_MAE']] = \
                                                                                        learning_rate, min_lr, patience, layer_size, num_layers, \
                                                                                        noise, l1, l2, dropout_rate, np.sqrt(mets[0]), mets[1]

                                                                print(df_log.loc[model_name])

                                                                df_log.to_csv('data/performances/GridSearch' + model_name + '.csv')

                                                                validation_output = model.predict(test_x)

                                                                predictionary = {}
                                                                for i in range(len(validation_output)):
                                                                    patient = test_y_labeled.index[i]
                                                                    predicted_age = validation_output[i]
                                                                    real_age = test_y_labeled[patient]
                                                                    error = real_age - predicted_age

                                                                    predictionary[patient] = [predicted_age, real_age, error]

                                                                print('\n\n*************************')
                                                                print('Dasatnib performance...')
                                                                dasat_accuracy = model.evaluate(dasat_x, dasat_y)
                                                                dasat_output = model.predict(dasat_x).flatten()

                                                                pearson = stats.pearsonr(dasat_output, dasat_y)
                                                                spearman = stats.spearmanr(dasat_output, dasat_y)
                                                                rsquared = sklearn.metrics.r2_score(dasat_y, dasat_output)

                                                                print('Dasatnib order: ', dasat_ages.index)

                                                                # print('Dasatnib elastic r-squared: ', reg.score(dasat_elastic_x, dasat_y))

                                                                print('Model evaluation (MSE, MAE): ', dasat_accuracy[:2])
                                                                print('R-squared : ', rsquared)
                                                                print('MAE confirmation: ', sklearn.metrics.mean_absolute_error(dasat_y, dasat_output))
                                                                print('RMSE confirmation: ', np.sqrt(sklearn.metrics.mean_squared_error(dasat_y, dasat_output)))
                                                                print('Pearson correlation: ', pearson)
                                                                print('Spearman correlation: ', spearman)
                                                                print('R-value confirmation: ', np.corrcoef(dasat_output, dasat_y)[0][1])
                                                                print('\n\n*************************')

                                                                print('Comite performance...')
                                                                comite_accuracy = model.evaluate(comite_x, comite_y)
                                                                comite_output = model.predict(comite_x).flatten()

                                                                pearson = stats.pearsonr(comite_output, comite_y)
                                                                spearman = stats.spearmanr(comite_output, comite_y)
                                                                rsquared = sklearn.metrics.r2_score(comite_y, comite_output)

                                                                # print('Elastic r-squared: ', reg.score(comite_elastic_x, comite_y))

                                                                print('Model evaluation (MSE, MAE): ', comite_accuracy)
                                                                print('R-squared: ', rsquared)
                                                                print('MAE confirmation: ', sklearn.metrics.mean_absolute_error(comite_y, comite_output))
                                                                print('RMSE confirmation: ', np.sqrt(sklearn.metrics.mean_squared_error(comite_y, comite_output)))
                                                                print('Pearson correlation: ', pearson)
                                                                print('Spearman correlation: ', spearman)
                                                                print('R-value confirmation: ', np.corrcoef(comite_output, comite_y)[0][1])
                                                                # explainer = shap.KernelExplainer(model=model, data=val_x)
                                                                # shap_values = explainer.shap_values(val_x)
                                                                # shap.force_plot(explainer.expected_value, shap_values, features=train_x_labeled,
                                                                #                 feature_names=train_x.columns)
                                                                # pd.DataFrame(shap_values).to_csv('data/SHAP_Output_' + model_name + '.csv')
                                                                # print(shap_values)

                                                                if ensemble_train:
                                                                    chunk_output = model.predict(all_x)
                                                                    order = og_betas.index
                                                                    print('Beta, scaler, and model paths: ', beta_file, scaler_path, file_path)
                                                                    print('Order of chunked predictions: ', order)
                                                                    return chunk_output, dasat_output, comite_output
                    else:
                        learning_rate = .01  # , .001]
                        min_lr = .001
                        patience = 8
                        layer_size = 100  # , 125]
                        num_layers = 7
                        noise = .15
                        l1 = .06  # [.2, .1]
                        l2 = 0  # [.2, .1]
                        b1 = .99  # [.5, .9, .99]
                        b2 = .999  # [.5, .9, .99]
                        dropout_rate = .08

                        model_name = 'ChronoAge' + str(learning_rate) + '-' + str(min_lr) \
                                     + '-' + str(patience) + '-' + str(layer_size) + '-' \
                                     + str(num_layers) + '-' + str(noise) + '-' + str(l1) \
                                     + '-' + str(l2) + '-' + str(dropout_rate)
                        if chaining:
                            model_name = model_name + ' Chain Model'
                        print(model_name)

                        callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                                                   patience=patience,
                                                                   mode='min',
                                                                   restore_best_weights=True),
                                     keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                       patience=int(patience / 2),
                                                                       factor=.5,
                                                                       verbose=1,
                                                                       mode='min',
                                                                       min_lr=min_lr),
                                     keras.callbacks.ModelCheckpoint(
                                         filepath='models/' + model_name,
                                         monitor='MAE',
                                         mode='min',
                                         save_freq=100)]

                        opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=b1,
                                                    beta_2=b2)

                        model = keras.Sequential(layers.Dense(len(train_x[0]), activation='relu'))

                        for layer in range(num_layers):
                            model.add(keras.layers.BatchNormalization())
                            model.add(keras.layers.GaussianNoise(noise))
                            model.add(keras.layers.ActivityRegularization(l1=l1, l2=l2))
                            model.add(keras.layers.Dense(layer_size, activation='relu'))
                            model.add(keras.layers.Dropout(dropout_rate))

                        output_layer = model.add(keras.layers.Dense(1))  # , bias_initializer=initial_bias))

                        model.compile(optimizer=opt,
                                      loss='mean_squared_error',
                                      metrics=metrics)

                        model.fit(x=train_x,
                                  y=train_y,
                                  validation_data=[test_x, test_y],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  verbose=1)

                        mets = model.evaluate(test_x, test_y)

                        model.save('models/' + model_name)

                        df_log.loc[model_name, ['Learning Rate', 'Min Learning Rate',
                                                'Patience', 'Layer Size', 'Number of Layers',
                                                'Noise', 'L1', 'L2', 'Dropout Rate',
                                                'Val_RMSE', 'Val_MAE']] = \
                            learning_rate, min_lr, patience, layer_size, num_layers, \
                            noise, l1, l2, dropout_rate, np.sqrt(mets[0]), mets[1]

                        print(df_log.loc[model_name])

                        df_log.to_csv('data/performances/GridSearch' + model_name + '.csv')

        # model_name = 'ChronoAge0.01-50-5-0.01-0.01-0.1'
        # model = keras.models.load_model('Models/' + model_name)


files = glob('data/chunked betas/*.csv')
# dasatnib = pd.read_csv('data/DasatinibQuercitin_GMQN_Blood_Betas.csv').set_index('Unnamed: 0')
# dasatnib = dasatnib.transpose()
# print(dasatnib)

# comite = pd.read_csv('data/Comite_GMQN_BMIQ_betas.csv').set_index('Unnamed: 0')

# all_cpgs = []
new_ensemble_train = False
if ensemble:
    if new_ensemble_train:
        chunk_outputs = {}
        dasat_outputs = {}
        comite_outputs = {}
        comite = pd.read_csv('ChronAge_test_Comite.csv').set_index('Unnamed: 0')
        dasatnib = pd.read_csv('DasatinibQuercitin_GMQN_Blood_Betas.csv').set_index('Unnamed: 0')
        dasatnib = dasatnib.transpose()
        comite = comite.transpose()

        for file in files[:]:
            # data = pd.read_csv(file).set_index('Unnamed: 0')
            # print(data)
            # cpgs = list(data.columns)
            # print(cpgs)
            # all_cpgs.append(cpgs)
            #
            # new_dasat = dasatnib[cpgs]
            # new_comite = comite[cpgs]
            #
            # new_dasat.to_csv('data/Dasatnib Chunks/DasatnibChunk' + str(run_num) + '.csv')
            # new_comite.to_csv('data/Comite Chunks/ComiteChunk' + str(run_num) + '.csv')
            # print(new_dasat)
            # print(new_comite)
            #
            file = file.split('/')[-1].replace('\\', '/')
            chunk_name = file.split('/')[-1].split('.')[0]
            print(file)

            # comite_file = 'data/comite chunks/Comite' + file.split('/')[-1]
            # comite_file = comite_file.replace(number, str(int(number)+1))
            # dasatnib_file = 'data/dasatnib chunks/Dasatnib' + file.split('/')[-1]
            # dasatnib_file = dasatnib_file.replace(number, str(int(number)+1))

            chunk_output, dasat_output, comite_output = train_model(f=file, c=comite, d=dasatnib, ensemble_train=True, c_name=chunk_name)
            chunk_outputs[chunk_name], dasat_outputs[chunk_name], comite_outputs[chunk_name] = chunk_output, dasat_output, comite_output

        pd.DataFrame(chunk_outputs).to_csv('data/Chunk Outputs.csv')
        pd.DataFrame(dasat_outputs).to_csv('data/Dasatnib Chunk Outputs.csv')
        pd.DataFrame(comite_outputs).to_csv('data/Comite Chunk Outputs.csv')
    else:
        chunk_outputs = {}
        dasat_outputs = {}
        comite_outputs = {}

        dasatnib = pd.read_csv('DasatinibQuercitin_GMQN_Blood_Betas.csv').set_index('Unnamed: 0').transpose()
        comite = pd.read_csv('Comite_GMQN_BMIQ_betas.csv').set_index('Unnamed: 0')
        age_file = 'Copy of Plasma_PatientMetrics_GMQN'
        comite_ages = pd.read_csv('data/' + age_file + '.csv').set_index('Patient.ID')['Decimal.Chronological.Age']
        age_file = 'Quercitin_Dasatinib_pdata.txt'
        dasat_ages = pd.read_csv('data/' + age_file + '.csv').set_index('Patient.ID')['Decimal.Chronological.Age']

        shared_dasat = (set(dasat_ages.index) & set(dasatnib.index))
        shared_comite = (set(comite_ages.index) & set(comite.index))
        dasatnib, dasat_ages = dasatnib.loc[shared_dasat], dasat_ages.loc[shared_dasat]
        comite, comite_ages = comite.loc[shared_comite], comite_ages.loc[shared_comite]

        ages = pd.read_csv('data/' + age_file + '.csv').set_index('Patient.ID')['Decimal.Chronological.Age'].dropna()

        for file in files:
            data = pd.read_csv(file).set_index('Unnamed: 0')
            # chunk_num = file.split('\\')[-1].split('.')[0].split('Chunk')[-1]
            chunk_name = file.split('/')[-1].split('.')[0]
            chunk_num = chunk_name.split(' ')[-1]
            print(chunk_name)
            model_path = 'chunk models/ChronoAge0.01-0.002-8-93-5-0.08-0.034-0.085-0.02 EnsembleChunk ' + str(chunk_num)
            model = keras.models.load_model(model_path)
            print(model_path)
            print(file)
            scaler = joblib.load('chunk scalers/Chunk ' + str(chunk_num) + 'EnsembleAgeScaler.joblib')
            print('chunk scalers/Chunk ' + str(chunk_num) + 'EnsembleAgeScaler.joblib')
            comite_data = comite[data.columns]
            dasat_data = dasatnib[data.columns]

            shared_patients = (set(data.index) & set(ages.index))
            for patient in shared_patients:
                data.loc[patient, 'Chronological Age'] = ages[patient]

            x, y = scaler.transform(data.drop(columns=['Chronological Age'])), data['Chronological Age']
            dasat_x, dasat_y = scaler.transform(dasat_data), dasat_ages
            comite_x, comite_y = scaler.transform(comite_data), comite_ages

            chunk_outputs[chunk_num] = model.predict(x)
            dasat_outputs[chunk_num] = model.predict(dasat_x)
            comite_outputs[chunk_num] = model.predict(comite_x)

        chunk_outputs = pd.read_csv('data/Chunk Outputs.csv')
        dasat_outputs = pd.read_csv('data/Chunk Outputs.csv')
        comite_outputs = pd.read_csv('data/Chunk Outputs.csv')

    train_model(chunk_predictions=chunk_outputs, d_chunks=dasat_outputs, c_chunks=comite_outputs)

else:
    train_model()


