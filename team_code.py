#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.
import collections

import sklearn
from sklearn.tree import export_graphviz

from feature.Feature_Extractor import Feature_Extractor
from feature.Feature_Loader import load_features
from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from graphviz import Source

twelve_lead_model_filename = '12_lead_model.sav'
six_lead_model_filename = '6_lead_model.sav'
three_lead_model_filename = '3_lead_model.sav'
two_lead_model_filename = '2_lead_model.sav'

################################################################################
#
# Training function
#
################################################################################
class_num = 11
classmap = pd.read_csv('class_label_11.csv','\t')
# classmap.set_index('Abbreviation')
classdict = classmap[['Abbreviation','SNOMEDCTCode']]
classdict = classdict.set_index("SNOMEDCTCode").T
classdict = classdict.to_dict(orient='list')
print(classdict)
idxpath = 'label_feat_list.txt'
labelpath = 'label_name.txt'
featidx = np.loadtxt(idxpath, dtype=float, delimiter=' ')
print(featidx)
label_names = np.loadtxt(labelpath, dtype=str, delimiter=' ')
feat_num = len(featidx)
sum_labels = np.zeros(class_num,dtype=int)

# exit(0)
def labelToStr(arr):
    stra =''
    for ar in arr:

        # print(ar , classdict[int(ar)])
        if ar == '':
            continue
        if int(ar) in classdict:
            stra = stra + str(classdict[int(ar)]) +' , '
        else:
            stra = stra + str(ar) + ' , '
    stra = stra[0:-2]
    return stra


def summer_history(np_summer,np_return,np_counter):
    for i in range (np_summer.shape[0]):
        if isinstance(np_return,int) == True:
            return np_summer,np_counter
        if np.isnan(np_return[i]) == True:
            # np_summer[i] = np_summer[i] + np_return[i]
            pass
        else:
            np_summer[i] = np_summer[i] + np_return[i]
            np_counter[i] += 1
    return np_summer,np_counter

def geteqlabel(strlabel):
    if strlabel == '164909002':
        return '733534002'
    if strlabel == '59118001':
        return '713427006'
    if strlabel == '63593006':
        return '284470004'
    if strlabel == '17338001':
        return '427172004'
    return strlabel

def getratio(A,B,C):
    # print (( B[:,0] / A[:][0] * 100  ,  B[:][1] / A[:][1] * 100) )
    return ( B[:,0] / A[:,0] * 100  ,  B[:,1] / A[:,1] * 100)

def detectValidFeature(args,file_text,pop):
    A,B = args
    for i,a in enumerate(A):
        b = B[i]
        if a is np.NAN or b is np.NAN:
            continue
        if pop[i][0] <5 or pop[i][1] < 5:
            continue
        if abs(a-b) > 60 or ( abs(a-50) > 40 and abs(a-b) > 40) or (abs(b-50) > 40 and abs(a-b) > 40):
            print( i ,'th feature ',file = file_text)
            sum_labels[i] += 1

def ext_treeRatio(strpath, tree):
    with open(strpath, "w") as text_file:

        # for i,tree in enumerate(trees):
        #     print('Enssenble # of tree : ', i, file = text_file)
        stack = collections.deque()
        stack.append(0)  # push tree root to stack
        while stack:
            current_node = stack.popleft()
            # node = tree[current_node]
            # do whatever you want with current node
            # ...
            left_child = tree.children_left[current_node]
            if left_child >= 0:
                stack.append(left_child)
            else:
                continue
            right_child = tree.children_right[current_node]
            if right_child >= 0:
                stack.append(right_child)
            else:
                continue
            print('Current node : ',current_node,label_names[tree.feature[current_node]],tree.threshold[current_node], file=text_file)
            varr = np.array(tree.value[current_node])
            varrL = np.array(tree.value[left_child])
            varrR = np.array(tree.value[right_child])
            print ('left : right side ratio (0 = false label ,1 = true label) ' ,file = text_file)
            print ( getratio(varr,varrL,varrR) ,file = text_file)
            detectValidFeature(getratio(varr,varrL,varrR),text_file, varr)



# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

   # if not num_recordings:
       # raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract classes from dataset.
    print('Extracting classes...')

    classes = set()
    for header_file in header_files:
        header = load_header(header_file)
        classes |= set(get_labels(header))
       # print(classes)
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically otherwise.
    num_classes = len(classes)
    # Extract features and labels from dataset.
    print('Extracting features and labels...')

    data_feat = []
    labels_feat=[]

    for i in range(0,num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))
        label_ = np.zeros( class_num, dtype=np.bool)
        # Load header and recording.
        header = load_header(header_files[i])

        feature_path = header_files[i].replace('hea','fet')

        #loading features from .fet files
        features = load_features(feature_path)
        if features is None:
            continue
        data_feat.append(features)

        current_labels = get_labels(header)
        for label in current_labels:
            if label is '':
                continue
            eqlabel = geteqlabel (label)
            if int(eqlabel) in classdict:
                j = list(classdict.keys()).index(int(eqlabel))
                label_[j] = 1
        labels_feat.append(label_)
        # print(current_labels)
    print (np.sum(labels_feat,axis=0))
    # Define parameters for random forest classifier.
    n_estimators = 3     # Number of trees in the forest.
    max_leaf_nodes = 100 # Maximum number of leaf nodes in each tree.
    random_state = 0     # Random state; set for reproducibility.

    # Train 12-lead ECG model.
    print('Training 12-lead ECG model...')

    leads = twelve_leads
    filename = os.path.join(model_directory, twelve_lead_model_filename)

    # feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
    features =  np.array(data_feat,dtype=float)
    features = np.where(features == np.inf, np.max(features) , features)
    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)
    # print(features)

    classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels_feat)
    save_model(filename, classes, leads, imputer, classifier)

    importances = classifier.feature_importances_
    std = np.std([
        tree.feature_importances_ for tree in classifier.estimators_], axis=0)

    import pandas as pd
    forest_importances = pd.Series(importances,  index=label_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()

    sklearn.tree.plot_tree(classifier[0])

    # graph = Source(export_graphviz(classifier[0], out_file='tree.dot', rounded=True, proportion=False,precision=2, filled=True))
    # graph.format = 'png'
    # graph.render('dtree_render', view=True)
    try:
        os.mkdir('output/')
    except:
        pass
    export_graphviz(classifier[0], out_file='output/tree1.dot',  feature_names=label_names,  rounded=True, proportion=False,  precision=2, filled=True)
    os.system('dot -Tpng output/tree1.dot > output/output1.png')

    export_graphviz(classifier[1], out_file='output/tree2.dot', feature_names=label_names, rounded=True,
                    proportion=False, precision=2, filled=True)
    os.system('dot -Tpng output/tree2.dot > output/output2.png')

    export_graphviz(classifier[2], out_file='output/tree3.dot', feature_names=label_names, rounded=True,
                    proportion=False, precision=2, filled=True)
    os.system('dot -Tpng output/tree3.dot > output/output3.png')

    # samples = collections.defaultdict(list)
    for i in range(n_estimators):
        strpath = 'output/treeratio' + str(i) + '.txt'
        ext_treeRatio(strpath,classifier[i].tree_)
    print(sum_labels)

    # dec_paths = classifier.decision_path(features)

    # for d, dec in enumerate(dec_paths):
    #     print(dec)
    #     for i in range(classifier.tree_.node_count):
    #         print(dec)
    #         if dec.toarray()[0][i] == 1:
    #             samples[i].append(d)

    # Train 6-lead ECG model.
    # print('Training 6-lead ECG model...')
    #
    # leads = six_leads
    # filename = os.path.join(model_directory, six_lead_model_filename)
    #
    # feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
    # features = data[:, feature_indices]
    #
    # imputer = SimpleImputer().fit(features)
    # features = imputer.transform(features)
    # classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)
    # save_model(filename, classes, leads, imputer, classifier)
    #
    # # Train 3-lead ECG model.
    # print('Training 3-lead ECG model...')
    #
    # leads = three_leads
    # filename = os.path.join(model_directory, three_lead_model_filename)
    #
    # feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
    # features = data[:, feature_indices]
    #
    # imputer = SimpleImputer().fit(features)
    # features = imputer.transform(features)
    # classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)
    # save_model(filename, classes, leads, imputer, classifier)
    #
    # # Train 2-lead ECG model.
    # print('Training 2-lead ECG model...')
    #
    # leads = two_leads
    # filename = os.path.join(model_directory, two_lead_model_filename)
    #
    # feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
    # features = data[:, feature_indices]
    #
    # imputer = SimpleImputer().fit(features)
    # features = imputer.transform(features)
    # classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)
    # save_model(filename, classes, leads, imputer, classifier)

def eval_dots(model_directory='model/',output_directory='output/'):
    filename = os.path.join(model_directory, twelve_lead_model_filename)

    classifier = load_model(filename)
    n_estimators = 3  # Number of trees in the forest.
    max_leaf_nodes = 100  # Maximum number of leaf nodes in each tree.
    random_state = 0  # Random state; set for reproducibility.

    # classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes,
    #                                     random_state=random_state)
    # classifier.load
    for i in range(n_estimators):
        strpath = 'output/treeratio' + str(i) + '.txt'
        ext_treeRatio(strpath, classifier[i].tree_)
    print(sum_labels)
################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save_model(filename, classes, leads, imputer, classifier):
    # Construct a data structure for the model and save it.
    # d = {'classes': classes, 'leads': leads, 'imputer': imputer, 'classifier': classifier}
    joblib.dump(classifier, filename, protocol=0)

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, twelve_lead_model_filename)
    return load_model(filename)

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    filename = os.path.join(model_directory, six_lead_model_filename)
    return load_model(filename)

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    filename = os.path.join(model_directory, three_lead_model_filename)
    return load_model(filename)

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, two_lead_model_filename)
    return load_model(filename)

# Generic function for loading a model.
def load_model(filename):
    return joblib.load(filename)

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(model, header, recording, headerpath):
    return run_model(model, header, recording, headerpath)

# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Generic function for running a trained model.
def run_model(model, header, recording, header_path):
    classes = model['classes']
    leads = model['leads']
    imputer = model['imputer']
    classifier = model['classifier']


    # Load features.
    num_leads = len(leads)
    data = np.zeros(num_leads+2, dtype=np.float32)
    age, sex, rms ,_= get_features(header, recording, leads)
    data[0:num_leads] = rms
    data[num_leads] = age
    data[num_leads+1] = sex

    feature_path = header_path.replace('hea','fet')
    features = load_features(feature_path)
    if features is None:
         return
    # data_feat.append(features)

    # Impute missing data.
    features = np.array(features, dtype=float)
    features = np.where(features == np.inf, 0, features)
    features = np.nan_to_num(features, 0)
    features = features.reshape(1, -1)
    print(features.shape)
    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)

    # Predict labels and probabilities.
    labels = classifier.predict(features)
    labels = np.asarray(labels, dtype=np.int)[0]

    probabilities = classifier.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    return classes, labels, probabilities

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adcgains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads, dtype=np.float32)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x**2) / np.size(x))
    rate = get_rate(header)
    return age, sex, rms , rate
