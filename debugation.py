# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 01:13:23 2019

@author: Athma_000
"""




predict(test_set.iloc[6], tree)



def predict(inst,tree):
    #This function is used to predict for any input variable 
    
    #Recursively we go through the tree that we built earlier

    for nodes in tree.keys():        #nodes : CommonHamWord

        value = inst[nodes] # value : 1
        tree = tree[nodes][value] # tree :  {'CommonHamWord': {}}
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction
        










####
xxxxxx ={'CommonHamWord':1, 'CommonSpamWord':0 }
inst = pd.Series(xxxxxx)

print(predict(inst,tree))

####

tree = buildTree(training_set, training_set.keys(), training_set.keys()[-1] )


#Class = df.keys()[-1]
def buildTree(df, attributes, y, tree=None): 
    
    best_attr = choose_best_attr(df) #παίρνει το καλύτερο χαρακτηριστικό
    best_attr_data = df.loc[:, best_attr] # βρίσκει το index της κολόνας του
    
    if tree is None:                    
        tree={}
        tree[best_attr] = {}
    
    for val in np.unique(best_attr_data):
        child_data = get_data(df, attributes[:-1], best_attr, val)
        #print("child_data:")
        #print(child_data)
        subtable = get_subtable(df, best_attr ,val)

        clValue,counts = np.unique(subtable['Class'],return_counts=True) 
        
        
        if len(child_data)==1:#Checking purity of subset
            tree[best_attr][val] = clValue[0]
        else:
            new_attributes = attributes[:] # attributes - best_attribute
            new_attributes.drop(best_attr)
               
            subtree = buildTree(child_data, new_attributes, y)
            tree[best_attr][val] = subtree
        
        
        
        
    '''
    for value in np.unique(best_attr_data):
        
        subtable = get_subtable(df,best_attr,value)
        clValue,counts = np.unique(subtable['Class'],return_counts=True)                        
        
        if len(counts)==1:#Checking purity of subset
            tree[best_attr][value] = clValue[0]                                                    
        else:        
            tree[best_attr][value] = buildTree(subtable) #Calling the function recursively  
    '''
    return tree



def get_data(examples, attributes, best_attribute,val):
        '''
        Παίρνει όλες τις γραμμές, όπου val = π.καλυτερη, και π ε παραδείγματα
        '''
        new_examples = pd.DataFrame()
        #index  = attributes.get_loc(best_attribute)
        i = 0
        for entry in examples[best_attribute]:
            if (entry == val): 
                #print("entry :")
                #print(entry)    
                new_examples = new_examples.append(examples.iloc[i])
                i = i + 1
                
                '''
                new_entry = []
                #for i in range(0,len(index)):
                    #if (i != index):
                new_entry.append(entry)
                
                new_examples.append(new_entry)
                '''
                    
        #new_data.remove([]) ??
        print("new_examples :")
        print(new_examples)
        return new_examples

