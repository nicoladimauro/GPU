import os
import csv 
import numpy as np

import arff
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import sys

# Load a CSV file
def load_csv(filename):
    file = open(filename, "r")
    lines = csv.reader(file)
    dataset = list(lines)
    return np.array(dataset)

def write_csv(data, path):
    with open(path, "w") as csv_file:
        writer_ = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for line in data:
            writer_.writerow(line)

def is_int(value):
     try:
         int(value)
         return True
     except ValueError:
         return False

datasets = ['audiology','breast-cancer', 'chess', 'dermatology',
            'hepatitis', 'lymph',  'nursery', 'pima', 'soybean', 'vote']
class_labels = ['cochlear_age','no-recurrence-events', 'won', 
                '<50', 'DIE', 'malign_lymph',  'not_recom', '0', 'brown-spot', 'democrat']

neg_class_labels = ['cochlear_unknown', 'recurrence-events', 'nowin', 
                    '>50_1', 'LIVE', 'metastases',  'priority', '1', 'alternarialeaf-spot', 'republican']

C_values = [10**v for v in range(-8, 4, 1)]
gamma_values = [10**v for v in range(-6, 6, 1)]


with open('results', 'w') as output:
    output.write('Dataset,perc,pos,ones,precision,recall,f1-score\n')

    for dataset, class_label, neg_class_label in zip(datasets,class_labels, neg_class_labels):
        #creating feature filename for bnlearn
        data_filename = 'data/' + dataset + '_train_pos_50_1.arff'
        data = arff.load(open(data_filename, 'r'))
        features_name = 'data/' + dataset + '.features'
        log_file = 'data/' + dataset + '.log'

        out_log_file = open(log_file,"w")
        out_log_file.write('perc,fold,ones,gamma,c,precision,recall,f1-score\n')
        out_log_file.flush()

        with open(features_name, 'w') as features_file:
            for attr in data['attributes'][:-1]:
                features_file.write('"' + attr[0] + '":categorical:')
                for val in attr[1][:-1]:
                    features_file.write('"' + val + '",')
                if attr[1][-1] != '':
                    features_file.write('"' + attr[1][-1] + '".\n')
                else:
                    features_file.write('".\n')


        for perc in ['30', '40', '50']:

            precision_f =  []
            recall_f = []
            f1_score_f = []
            ones_f = []

            for fold in range(1,11):
                print('Fold:', fold)

                pos_name = 'data/' + dataset + '_train_pos_' + perc + '_' + str(fold) + '.arff'
                unl_name = 'data/' + dataset + '_train_unl_' + perc + '_' + str(fold) + '.arff'
                test_name = 'data/' + dataset + '_test_' + perc + '_' + str(fold) + '.arff'


                train_pos = arff.load(open(pos_name, 'r'))
                train_unl = arff.load(open(unl_name, 'r'))
                test = arff.load(open(test_name, 'r'))

                train_pos_data = np.array(train_pos['data'])
                train_unl_data = np.array(train_unl['data'])

                test_data = np.array(test['data'])

                write_csv(train_pos_data[:,:-1],'./data/pos.data')
                write_csv(train_unl_data[:,:-1],'./data/unl.data')

                command = 'R --no-save --args ./data/' + dataset + '.features ./data/pos ./data/unl outfile < bn_k2.R > /dev/null'

                os.system(command)

                lls = np.loadtxt('outfile')

                argsort = np.argsort(lls)

                ones = 0
                for index in argsort[:train_pos_data.shape[0]]:
                    if train_unl_data[index,-1]==class_label:
                        ones = ones + 1


                X_train_pos_neg = np.concatenate((train_pos_data[:,:-1], train_unl_data[argsort[:train_pos_data.shape[0]],:-1]), axis=0)
                y_train_pos_neg = np.array([class_label]*train_pos_data.shape[0] + [neg_class_label]*train_pos_data.shape[0])

                X_train_pos_neg_int = np.zeros((X_train_pos_neg.shape[0],X_train_pos_neg.shape[1]))
                attributes = train_pos['attributes']
                for i in range(X_train_pos_neg.shape[1]):
                    values = attributes[i][1]
                    for j in range(X_train_pos_neg.shape[0]):
                        X_train_pos_neg_int[j,i] = values.index(X_train_pos_neg[j,i])

                X_test_int = np.zeros((test_data.shape[0],test_data.shape[1]-1))
                attributes = train_pos['attributes']
                for i in range(test_data.shape[1]-1):
                    values = attributes[i][1]
                    for j in range(test_data.shape[0]):
                        X_test_int[j,i] = values.index(test_data[j,i])

                X_all_int = np.concatenate((X_train_pos_neg_int, X_test_int), axis=0)

                encoder = OneHotEncoder()
                encoder.fit(X_all_int)
                A = encoder.transform(X_train_pos_neg_int).toarray()
                B = encoder.transform(X_test_int).toarray()

                param_grid = dict(gamma=gamma_values, C=C_values)
                #cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=177)
                cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=177)
                f1_scorer = make_scorer(f1_score, pos_label=neg_class_label)
                grid = GridSearchCV(SVC(), param_grid=param_grid,  cv=cv, scoring=f1_scorer)
                grid.fit(A, y_train_pos_neg)

                print("The best parameters are %s with a score of %0.2f"
                      % (grid.best_params_, grid.best_score_))

                gamma = grid.best_params_['gamma']
                C = grid.best_params_['C']


                clf = SVC(kernel='rbf', gamma=gamma, C=C)
                clf.fit(A, y_train_pos_neg) 

                y_train_pred = clf.predict(A)

                print('Train stats')
                pr = precision_score(y_train_pos_neg,y_train_pred, pos_label=neg_class_label, average='binary')
                re = recall_score(y_train_pos_neg,y_train_pred, pos_label=neg_class_label, average='binary')
                f1 = f1_score(y_train_pos_neg,y_train_pred, pos_label=neg_class_label, average='binary')
                print('Precision:', pr)
                print('Recall:', re)
                print('F1-score:', f1)

                y_test_pred = clf.predict(B)

                print('Test stats')
                pr = precision_score(test_data[:,-1],y_test_pred, pos_label=neg_class_label, average='binary')
                re = recall_score(test_data[:,-1],y_test_pred, pos_label=neg_class_label, average='binary')
                f1 = f1_score(test_data[:,-1],y_test_pred, pos_label=neg_class_label, average='binary')
                print('Precision:', pr)
                print('Recall:', re)
                print('F1-score:', f1)

                precision_f.append(pr)
                recall_f.append(re)
                f1_score_f.append(f1)
                ones_f.append(ones)

                out_log_file.write(str(perc) + ',' + str(fold) + ',' +str(ones) + ',' +str(gamma) + ',' +
                                   str(C) + ',' + str(pr) + ',' + str(re) + ',' + str(f1) +'\n')
                out_log_file.flush()

            output.write(dataset + ',' + perc + ',')
            output.write(str(train_pos_data.shape[0]) + ',' +
                         str(np.mean(ones_f)) + ',' +
                         str(np.mean(precision_f)) + ',' +
                         str(np.mean(recall_f)) + ',' +
                         str(np.mean(f1_score_f)) + '\n')

            output.flush()
        out_log_file.close()
