# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:26:49 2019

@author: Athma_000
"""

'''
    Εδώ ξεκινάει η υλοποίηση του αλγορίθμου ID3 στην βάση enron1, όπου εμπεριέχονται αρχεία τα οποία αντιπροσωπεύουν spam email και ham εμαιλ(δηλ. καλά email).
    
    Περιλαμβάνονται όλες οι εντολές για την κατάλληλη προεπεξεργασία και εξαγωγή των χαρακτηριστικών
    Επίσης έχω συμπεριλάβει δύο .spydata αρχεία για οικονομία χρόνου. Το ένα (hamSpamDictionaries.spydata) απλώς περιλαμβάνει τα dictionaries των πιο συνηθισμένων λέξεων για την βάση,
    ενω το άλλο (readyForTreeData.spydata) έχει όλα τα δεδομένα που χρειαζόμαστε για το τρέξιμο ενός δέντρου.

    
'''
import os
import pandas as pd
import numpy as np
eps = np.finfo(float).eps # smallest representative number, to avoid cases where log(0).
from numpy import log2 as log

#####################################################################################################################
# ID3 required fucntions

def find_entropy(df):
    '''
        Finds the entropy of "Class"(target), using S = - Σ pi logpi, for the given dataset.
    '''
    Class = df.keys()[-1]
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy
  

def find_entropy_attribute(df,attribute): 
    '''
        Calculates info gain that would result from splitting the data in chosen attr.
    '''
    Class = df.keys()[-1]

    target_variables = df[Class].unique()  #This gives all '1' and '0'
    variables = df[attribute].unique()    #This gives different features in that attribute, in our case binary ones ('1', '0'). However, it could be anything like 'Humid', 'Windy' etc.
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
            den = len(df[attribute][df[attribute]==variable])
            fraction = num/(den+eps)
            entropy += -fraction*log(fraction+eps)
        fraction2 = den/len(df)
        entropy2 += -fraction2*entropy
    return abs(entropy2) # returns absolute value of a number


def choose_best_attr(df):
    '''
        Chooses the best attribute to split the tree.       
    '''
    IG = []
    for key in df.keys()[:-1]:
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]
  
  
def get_subset(df, node,value):
    '''
        Returns the subset of examples that have value 'value' in node
    '''
    return df[df[node] == value].reset_index(drop=True)

def most_common_class(dataframe):
    '''
        Also kwown as majority(). 
        Returns the most common class in a given dataframe.
    '''
    target = dataframe.keys()[-1]
    return np.unique(dataframe[target])[np.argmax(np.unique(dataframe[target],return_counts=True)[1])]


def build_tree(df, attributes, tree=None): 
    '''
        Constructs an ID3 tree recursively.
    '''
    target = attributes[-1] # or = df.keys()[-1]

    # Case 1 : data is empty
    if len(df) == 0:
        return most_common_class(df)
    
    # Case 2 : all remaining examples belong to one class => perfectly classified
    elif len(np.unique(df[target])) == 1:
        return np.unique(df[target])[0]
    
    # Case 3 : if attributes is empty and only target remains
    elif (len(attributes) -1 ) == 1:
        return most_common_class(df)

    else:
        # Choose the best attribute, based on info gain
        best_attr = choose_best_attr(df) #also can act like a node
        # Create new tree, which controls the best_attr at its root
        if tree is None:
            tree = {}
            tree[best_attr] = {}
            
        # for each distinct value of best attribute
        for value in np.unique( df[best_attr] ):
            
            # the subset of examples that have value v for best attribute
            subset = get_subset(df, best_attr, value ) #examples,i
            
            clValue,counts = np.unique(subset[target],return_counts=True)
            if len(counts) == 1:
                tree[best_attr][value] = clValue[0]                                                    
            else:                  
                # new attributes for the subtree ## ιδιότητες - καλύτερη
                new_attributes = attributes[:]
                new_attributes = new_attributes.drop(best_attr)
                    
                # Create new subtree
                subtree = build_tree(subset, new_attributes)
                    
                # Add the new subtree to the new tree/node which was just created
                tree[best_attr][value] = subtree
                    
    return tree


def predict(inst,tree):
    '''
        Predicts any input variable 
        Recursively we go through the tree
    '''
    for nodes in tree.keys():        #nodes : CommonHamWord 
        value = inst[nodes] # value : 
        tree = tree[nodes][value] # tree :  {'CommonHamWord': {}} ..
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction






#####################################################################################################################
# Preparing data & dataframes for Id3

def check_if_word_exists_in_text(text, word):    
    '''
        Checks if a word exists in a text
        param : text is a string var that contains the full body text of an email in our case
        param : word is the word we are looking for
    '''
    if ( word in text ):
        return 1
    return 0

def read_databases(spam_path = "C:/Users/Athma_000/Desktop/mining/enron1/spam/", ham_path = "C:/Users/Athma_000/Desktop/mining/enron1/ham" ):
    '''
    read data in CSV format according to your PC's address
    '''
    
    # Read spam folder
    df_spam = pd.DataFrame()
    for file in os.listdir(spam_path):
        with open(os.path.join(spam_path, file), encoding="utf-8",errors='ignore') as f:
            text = f.read()
            word_http = check_if_word_exists_in_text(text, 'http')
            word_company = check_if_word_exists_in_text(text, 'company')
            word_e = check_if_word_exists_in_text(text, 'e')
            word_www = check_if_word_exists_in_text(text, 'www')
            word_information = check_if_word_exists_in_text(text,'information')
            current_df = pd.DataFrame({'ContainsWordHttp' : [word_http], 'ContainsWordCompany' : [word_company], 'ContainsWordE' : [word_e], 'ContainsWord_www' : [word_www], 'ContainsWordInformation' : [word_information]})       
            df_spam = df_spam.append(current_df, ignore_index=True)
    
    
    # Read ham folder
    df_ham = pd.DataFrame()
    for file in os.listdir(ham_path):
        with open(os.path.join(ham_path, file), encoding="utf-8",errors='ignore') as f:
            text = f.read()  
            word_http = check_if_word_exists_in_text(text, 'http')
            word_company = check_if_word_exists_in_text(text, 'company')
            word_e = check_if_word_exists_in_text(text, 'e')
            word_www = check_if_word_exists_in_text(text, 'www')
            word_information = check_if_word_exists_in_text(text,'information')
            current_df = pd.DataFrame({'ContainsWordHttp' : [word_http], 'ContainsWordCompany' : [word_company], 'ContainsWordE' : [word_e], 'ContainsWord_www' : [word_www], 'ContainsWordInformation' : [word_information]})       
            df_ham = df_ham.append(current_df, ignore_index=True)

    
    #Add Class column to each dataFrame
    df_spam['Class'] = 1
    df_ham['Class'] = 0
    
    data = df_spam.append(df_ham, ignore_index=True)
    
    return data, df_ham, df_spam


#####################################################################################################################
#main
    

def main():
    
    # Read enron databases
    
    data, df_ham, df_spam = read_databases()

    # Create training and testing set
    from sklearn.utils import shuffle
    data = shuffle(data)
    
    training_part = int(len(data) * .7)
    
    training_set = data[:training_part]
    
    test_set = df_ham[:]
    
    y = test_set.Class
    test_set.drop(['Class'],axis=1,inplace = True) # drop target label from test set
    
    # Start recursively building the tree

    tree = build_tree( training_set  , training_set.keys() )    
    
    '''
    Μπορούμε να δούμε την δομή του δέντρου με το τρέξιμο των παρακάτω γραμμμών και χρήση του pretty_print
    
    import pprint
    pprint.pprint(tree)
    '''
    
    # predict
    
    predictions = []
    for i in range(len(test_set)):
        prediction = predict(test_set.iloc[i], tree)
        predictions.append(prediction)    


    #scores
    
    from sklearn.metrics import recall_score
    recall_score(y, predictions)
        
    from sklearn.metrics import average_precision_score
    average_precision_score(y, predictions)

    from sklearn.metrics import accuracy_score
    accuracy_score(y, predictions)
    
    from sklearn.metrics import f1_score
    f1_score(y, predictions)
    


