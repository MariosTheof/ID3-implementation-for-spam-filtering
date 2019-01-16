# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:46:41 2019

@author: Athma_000
"""

import numpy as np
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from collections import Counter
'''
import nltk
nltk.download('punkt')
'''

from nltk.tokenize import word_tokenize
from sklearn.utils import shuffle
eps = np.finfo(float).eps
from numpy import log2 as log
from sklearn.model_selection import train_test_split

def find_entropy(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy
  

def find_entropy_attribute(df,attribute):
  Class = df.keys()[-1]   #To make the code generic, changing target variable class name
  target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
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
  return abs(entropy2)


def choose_best_attr(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
#         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]
  
  
def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True)



def buildTree(df,tree=None): 
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    
    #Here we build our decision tree
    
    #Get attribute with maximum information gain
    node = choose_best_attr(df)
    ## majority()
     
    #Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])

    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
   #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in attValue:
        
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['Class'],return_counts=True)                        
        
        if len(counts)==1:#Checking purity of subset
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree(subtable) #Calling the function recursively 
            
           
    return tree
        

    

def get_data(examples, attributes, best_attribute,val):
        
        new_examples = [[]]
        index = attributes.index(best_attribute)
        
        for entry in examples:
            if (entry[index] == val):
                new_entry = []
                for i in range(0,len(entry)):
                    if (i != index):
                        new_entry.append(entry[i])
                    new_examples.append(new_entry)
                    
        #new_data.remove([]) ??
        return new_examples 

def get_values(examples, attributes, single_attr):
        
        index = attributes.index(single_attr)
        values = []
        
        for entry in examples:
            if entry[index] not in values:
                values.append(entry[index])

        return values
    






























def predict(inst,tree):
    #This function is used to predict for any input variable 
    
    #Recursively we go through the tree that we built earlier

    for nodes in tree.keys():        
        
        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction
        
           











def make_dictionary(path):
    emails = [os.path.join(path,f) for f in os.listdir(path)]    
    all_words = []       
    for email in emails:
        with open(email, encoding='latin1') as m:
            content = m.read()
            all_words += nltk.word_tokenize(content)
    dictionary = [word for word in all_words if word not in stopwords.words('english')]
    dictionary = [word.lower() for word in dictionary if word.isalpha()]
    dictionary = Counter(dictionary)
    dictionary = dictionary.most_common(10)
    return dictionary


def create_word_features(words):
    my_dict = dict( [ (word, True) for word in words] )
    return my_dict


from nltk.corpus import words
def does_word_exist(word):
    '''
    Checks if a word exists
    @word = word we wish to check if exists
   requires :
        import nltk
        nltk.download('words')    
    '''
    word_exists = word in words.words()
    return word_exists


def check_if_most_words_exist(file_text):
    '''
    checks if most words in a file exist in the english language
    '''
    existing_word_count = 0
    words = word_tokenize(file_text)  
    for word in words:
        if (does_word_exist(word) == True):
            existing_word_count = existing_word_count + 1
    
    if ( existing_word_count >= (0.5 * len(words)) ):
        return True
    else:
        return False
    

def check_most_common_words(text):    
    '''
    '''        
    containsMostCommonSpamWord = 0
    containsMostCommonHamWord = 0
    words = word_tokenize(text)
    for word in words:
        
        for spam_value in spamDictionary:
            if word == spam_value[0]:
                containsMostCommonSpamWord = 1
                
        for ham_value in hamDictionary:
            if word == ham_value[0]:
                containsMostCommonHamWord = 1
              

    return containsMostCommonSpamWord, containsMostCommonHamWord

'''
import collections
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
# Read input file, note the encoding is specified here 
# It may be different in your text file
tempFile = "0160.2004-01-07.GP.spam.txt"
file = open(os.path.join(spam_path, tempFile), encoding="utf8")
a= file.read()
# Stopwords
stop_words = stopwords.words('english')
# Instantiate a dictionary, and for every word in the file, 
# Add to the dictionary if it doesn't exist. If it does, increase the count.
wordcount = {}
# To eliminate duplicates, remember to split by punctuation, and use case demiliters.
for word in a.lower().split():
    word = word.replace(".","")
    word = word.replace(",","")
    word = word.replace(":","")
    word = word.replace("\"","")
    word = word.replace("!","")
    word = word.replace("â€œ","")
    word = word.replace("â€˜","")
    word = word.replace("*","")
    if word not in stop_words:
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1
            
# Print most common word
n_print = int(input("How many most common words to print: "))

print("\nOK. The {} most common words are as follows\n".format(n_print))
word_counter = collections.Counter(wordcount)
for word, count in word_counter.most_common(n_print):
    print(word, ": ", count)
# Close the file
file.close()
# Create a data frame of the most common words 
# Draw a bar chart
lst = word_counter.most_common(n_print)
df = pd.DataFrame(lst, columns = ['Word', 'Count'])
df.plot.bar(x='Word',y='Count')
'''





def read_databases():
    '''
    read data in CSV format according to your PC's address
    '''
    
    #Read spam folder
    df_spam = pd.DataFrame(columns=['CommonHamWord', 'CommonSpamWord'])
    spam_path = "C:/Users/Athma_000/Desktop/mining/enron1/spam/"

    for file in os.listdir(spam_path):
        
        with open(os.path.join(spam_path, file), encoding="utf-8",errors='ignore') as f:
            text = f.read()
            #most_words_exist = check_if_most_words_exist(text)
            
            containsMostCommonSpamWord, containsMostCommonHamWord = check_most_common_words(text)
            
            current_df = pd.DataFrame({'Text': [text], 'CommonSpamWord' : [containsMostCommonSpamWord], 'CommonHamWord' : [containsMostCommonHamWord]})       
            
            df_spam = df_spam.append(current_df, ignore_index=True)

    

    
    #Read ham folder
    df_ham = pd.DataFrame(columns=['CommonHamWord', 'CommonSpamWord'])
    ham_path = "C:/Users/Athma_000/Desktop/mining/enron1/ham"
    

    for file in os.listdir(ham_path):

        with open(os.path.join(ham_path, file), encoding="utf-8",errors='ignore') as f:
            text = f.read()
            #most_words_exist = check_if_most_words_exist(text)
            
            containsMostCommonSpamWord, containsMostCommonHamWord = check_most_common_words(text)
            
            current_df = pd.DataFrame({'Text': [text], 'CommonSpamWord' : [containsMostCommonSpamWord], 'CommonHamWord' : [containsMostCommonHamWord]})       

            df_ham = df_ham.append(current_df, ignore_index=True)

    return df_ham, df_spam



def main():
    
    df_ham, df_spam = read_databases()
    
    spam_path = "C:/Users/Athma_000/Desktop/mining/enron1/spam/"
    spamDictionary = make_dictionary(spam_path)
    
    ham_path = "C:/Users/Athma_000/Desktop/mining/enron1/ham"
    hamDictionary = make_dictionary(ham_path)
    
    
    
    
    '''
    spam_path = "C:/Users/Athma_000/Desktop/mining/enron1/spam"
    spam_dict = make_Dictionary(spam_path)
    ham_path = "C:/Users/Athma_000/Desktop/mining/enron1/ham"
    ham_dict = make_Dictionary(ham_path)
    
    
    df_ham['HamList'] = ham_list
    df_spam['SpamList'] = spam_list
    '''
    
   
    
    
    df_ham.drop(['Text'],axis=1, inplace = True)
    df_spam.drop(['Text'],axis=1, inplace = True)
    
    #Add Class column to each dataFrame
    df_spam['Class'] = 1
    df_ham['Class'] = 0
    
    
    data = df_spam.append(df_ham, ignore_index=True)
        
 
    # Extract target column 'Class'
    y = data.Class
    
    
    training_part = int(len(data) * .1)
    
    from sklearn.utils import shuffle
    data = shuffle(data)
    
    training_set = data[:training_part]
    test_set = data[training_part:]
    

    tree = buildTree(training_set)
    

    
    '''
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(max_depth = 1)
    clf.fit(training_set,training_set.Class)
    clf.score(X_train,y_train)
    
    '''
    test_set.drop(['Class'],axis=1,inplace = True)
    
    predictions = []
    #correct_guess = 0
    for i in range(len(test_set)):
        prediction = predict(test_set.iloc[i], tree)
        predictions.append(prediction)    
       # if (y_test.iloc[i] == prediction):
          #  correct_guess = correct_guess + 1
   
    
    accuracy = float(predictions.count(True))/float(len(predictions))
    
    
    print("Accuracy is : %.4f" % accuracy)
    
    '''
    acc.append(accuracy)

    avg_acc = sum(acc)/len(acc)
    print("Average accuracy: %.4f" % avg_acc)
    '''




















            
            
    