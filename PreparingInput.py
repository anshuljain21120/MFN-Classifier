from datetime import datetime
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

def prepareInput(wv, data):    
    print('{:,} samples loaded.'.format(data.shape[0]))

    #Trimming
    remove = []
    # CSV file loaded should have column names "words" and "mfn" 
    # for strings and labels respectively.
    for word in data['words']:
        try:
            temp = wv[word]
        except:
            remove.append(word)
            data = data.drop(data[data['words']==word].index)
    print('{0:,} Words not in Vocab, preparing remaining {1:,} rows.'.format(len(remove), data.shape[0]))
    del temp, remove

    #Attaching Word Vectors
    vlist = []
    for word in data['words']:
        vlist.append(wv[word])
    del wv
    vlist = np.array(vlist)
    vlist = vlist.transpose()
    for i in range(200):
        data['vector_'+str(i)] = vlist[i]
    print('Samples got word vectors, new size of data is ({0:,} x {1:,}).'.format(data.shape[0], data.shape[1]))
    del vlist

    #Eliminating offset if any
    mSamples = data[data['mfn']== -1]
    fSamples = data[data['mfn']== 1]
    nSamples = data[data['mfn']== 0]
    difference = mSamples.shape[0] - fSamples.shape[0]
    if difference < 0:
        fSamples = fSamples.iloc[:fSamples.shape[0] - abs(difference)]
    else:
        mSamples = mSamples.iloc[:mSamples.shape[0] - abs(difference)]
    print("Trimmed Data now has:\n\t{0:,} Female samples,\n\t{1:,} Male samples,\tand\n\t{2:,} Neutral samples.".format(
        fSamples.shape[0],
        mSamples.shape[0],
        nSamples.shape[0]
    ))
    del difference, data

    #Arranging Samples according to both SVMs
    temp1 = nSamples.drop(columns="mfn")
    temp1.insert(1, "mfn", 1)
    temp2 = fSamples.drop(columns="mfn")
    temp2.insert(1, "mfn", 1)
    mnmSamples = mSamples.append(temp1, ignore_index=True).append(temp2, ignore_index=True)
    temp1 = nSamples.drop(columns="mfn")
    temp1.insert(1, "mfn", -1)
    temp2 = mSamples.drop(columns="mfn")
    temp2.insert(1, "mfn", -1)
    fnfSamples = fSamples.append(temp1, ignore_index=True).append(temp2, ignore_index=True)
    del temp1, temp2, mSamples, fSamples, nSamples

    #Saving processed data and verifying the same
    if mnmSamples.shape[0]==fnfSamples.shape[0]:
        print("mnmSamples and fnfSamples ready. Both have {:,} samples.".format(fnfSamples.shape[0]))
    else:
        print("Fetching SVM data Failed!")
    return mnmSamples, fnfSamples
