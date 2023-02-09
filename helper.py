'''Helper functions.
'''

import pickle
import torch
import csv


coarse_path = "./dataset/coarse_labels.csv"
fine_path = "./dataset/fine_labels.csv"

def unpickle(file):
    '''Unpickle the given file
    '''

    with open(file, 'rb') as f:
        res = pickle.load(f, encoding='bytes')
    return res

def read_meta(metafile):
    '''Read the meta file and return the coarse and fine labels.
    '''
    meta_data = unpickle(metafile)
    fine_label_names = [t.decode('utf8') for t in meta_data[b'fine_label_names']]
    coarse_label_names = [t.decode('utf8') for t in meta_data[b'coarse_label_names']]
    return coarse_label_names, fine_label_names

def csv_to_list_coarselabels():
        '''Reads the path of the file and its corresponding label
        '''
        
        with open(coarse_path, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            data = [data[0][0], data[1][0], data[2][0]]
            
        return data
    
def csv_to_list_finelabels():
        '''Reads the path of the file and its corresponding label
        ''' 
        with open(fine_path, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            data = [data[0][0], data[1][0], data[2][0],data[3][0], data[4][0], data[5][0],
                    data[6][0], data[7][0], data[8][0], data[9][0], data[10][0]
                    ]

        return data

def calculate_accuracy(predictions, labels):
    '''Calculates the accuracy of the prediction.
    '''

    #num_data = labels.size()[0]
    predicted = torch.argmax(predictions, dim=1)
    #correct_pred = torch.sum(predicted == labels)
    #accuracy = correct_pred*(100/labels)
    #return accuracy.item()
    error_rate = abs((predicted - labels))*(100/(predicted))
    accuracy = abs(100 - error_rate)
    
    return accuracy
                
    
