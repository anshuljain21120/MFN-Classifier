from datetime import datetime
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, hinge_loss
from sklearn.model_selection import train_test_split 
import joblib 
import pandas as pd
import matplotlib.pyplot as plt
from PreparingInput import prepareInput
from gensim.models import KeyedVectors
import argparse

#argument parser initialization
parser = argparse.ArgumentParser()
parser.add_argument('wvPath')
parser.add_argument('samplesPath')
parser.add_argument('runs', type = int, default = 3)
parser.add_argument('-verbose', action='store_true')
args = parser.parse_args()

#Preprocessing
mnmSamples ,fnfSamples = prepareInput(KeyedVectors.load(args.wvPath), pd.read_csv(args.samplesPath))

#Separating vectors and labels
mlabels = mnmSamples["mfn"].to_numpy()
mvectors = mnmSamples.drop(columns=["words", "mfn"]).to_numpy()
del mnmSamples
flabels = fnfSamples["mfn"].to_numpy()
fvectors = fnfSamples.drop(columns=["words", "mfn"]).to_numpy()
del fnfSamples

#Splitting data for train, test and CV
total_runs = int(args.runs)
mtrainvecs = [None]*total_runs
mCVvecs = [None]*total_runs
mtestvecs = [None]*total_runs
mtrainlabels = [None]*total_runs
mCVlabels = [None]*total_runs
mtestlabels = [None]*total_runs

ftrainvecs = [None]*total_runs
fCVvecs = [None]*total_runs
ftestvecs = [None]*total_runs
ftrainlabels = [None]*total_runs
fCVlabels = [None]*total_runs
ftestlabels = [None]*total_runs

mX, mtestvecs, mY, mtestlabels = train_test_split(mvectors, mlabels, test_size=0.1, random_state=0)
fX, ftestvecs, fY, ftestlabels = train_test_split(fvectors, flabels, test_size=0.1, random_state=0)
for n in range(total_runs):
    mtrainvecs[n], mCVvecs[n], mtrainlabels[n], mCVlabels[n]=train_test_split(mX, mY, test_size=0.30, random_state = None)
    ftrainvecs[n], fCVvecs[n], ftrainlabels[n], fCVlabels[n]=train_test_split(fX, fY, test_size=0.30, random_state = None)
del mvectors, mlabels, fvectors, flabels

#Printing Stats
print("Total runs: {0:}".format(total_runs))
print("Training examples(m, f): {:}, {:}\t".format(len(mtrainlabels[0]), len(ftrainlabels[0])))
print("Cross-Validation examples(m, f): {:}, {:}\t".format(len(mCVlabels[0]), len(fCVlabels[0])))
print("Testing examples(m, f): {:}, {:}\t".format(len(mtestlabels), len(ftestlabels)))

#Training Now
Penaltypowers = [i for i in range(1,3)]
Gpowers = [i for i in range(-9,-7)]
Penalty = [2**i for i in Penaltypowers]
G = [2**i for i in Gpowers]
total_models = len(Penalty)*len(G)

maccuracylist = [None]*total_runs
faccuracylist = [None]*total_runs

mF1list = [None]*total_runs
fF1list = [None]*total_runs

mlosslist = [None]*total_runs
flosslist = [None]*total_runs

for k in range(total_runs):
    print("Run "+str(k+1)+" started...")    
    maccuracylist[k] = []
    faccuracylist[k] = []
    
    mF1list[k] = []
    fF1list[k] = []
    
    mlosslist[k] = []
    flosslist[k] = []
    n=0
    for i in range(len(Penalty)):
        for j in range(len(G)):            
            #BUILDING MODEL STRUCTURE
            mmodel = svm.SVC(kernel='rbf',class_weight='balanced', C=Penalty[i], gamma=G[j], probability=True)
            
            fmodel = svm.SVC(kernel='rbf', class_weight='balanced', C=Penalty[i], gamma=G[j], probability=True)
            
            #TRAINING MODELS
            print(str(k+1)+'.'+str(n+1), end=' ')
            print("Fitting Model ("+str(Penaltypowers[i])+", "+str(Gpowers[j])+") start:"+datetime.now().strftime('%H:%M:%S'))
            
            mmodel.fit(mtrainvecs[k], np.ravel(mtrainlabels[k]))
            if(args.verbose):
                print("\tM-Model trained: OK")
            
            fmodel.fit(ftrainvecs[k], np.ravel(ftrainlabels[k]))
            if(args.verbose):
                print("\tF-Model trained: OK")
            
            #CALCULATING ACCURACY
            if(args.verbose):
                print("\tCalculating Scores...")
            maccuracylist[k].append(accuracy_score(mCVlabels[k], mmodel.predict(mCVvecs[k])))
            
            faccuracylist[k].append(accuracy_score(fCVlabels[k], fmodel.predict(fCVvecs[k])))

            #CALCULATING F1 SCORES
            mF1list[k].append(f1_score(mCVlabels[k], mmodel.predict(mCVvecs[k]), average='macro'))

            fF1list[k].append(f1_score(fCVlabels[k], fmodel.predict(fCVvecs[k]), average='macro'))
            
            #CALCULATING LOSSES
            mlosslist[k].append(hinge_loss(mCVlabels[k], mmodel.predict(mCVvecs[k]))*5)
            
            flosslist[k].append(hinge_loss(fCVlabels[k], fmodel.predict(fCVvecs[k]))*5)
            
            #PRINTING SCORES
            if(args.verbose):
                print("\tMALE:   Accuracy = "+str(maccuracylist[k][n])+"\tF1 score = "+str(mF1list[k][n]))
            
                print("\tFEMALE: Accuracy = "+str(faccuracylist[k][n])+"\tF1 score = "+str(mF1list[k][n])+"\n")
            n += 1
#k-fold Average
mavgaccuracylist = np.mean(maccuracylist, axis=0)
mavglosslist = np.mean(mlosslist, axis=0)

favgaccuracylist = np.mean(faccuracylist, axis=0)
favglosslist = np.mean(flosslist, axis=0)

mavgF1list = np.mean(maccuracylist, axis=0)
mavglosslist = np.mean(mlosslist, axis=0)

favgF1list = np.mean(fF1list, axis=0)
favglosslist = np.mean(flosslist, axis=0)
if(args.verbose):
    ax = []
    f, ax = plt.subplots(nrows=2, ncols=2,figsize=(15,15))

    for k in range(total_runs):
        ax[0][0].plot(range(total_models), maccuracylist[k], label='M_Run '+str(k+1), alpha=0.7)
        ax[0][0].plot(range(total_models), mlosslist[k], label='M_Run '+str(k+1), alpha=0.7)
    ax[0][0].plot(range(total_models), mavgaccuracylist, 'b', label='AVGaccu', linewidth=2)
    ax[0][0].plot(range(total_models), mavgaccuracylist, 'bo', label='AVGaccu')
    ax[0][0].plot(range(total_models), mavglosslist, 'k', label='AVGloss', linewidth=2)
    ax[0][0].plot(range(total_models), mavglosslist, 'ko', label='AVGloss')
    ax[0][0].legend()
    ax[0][0].grid()
    for k in range(total_runs):
        ax[1][0].plot(range(total_models), faccuracylist[k], label='F_Run '+str(k+1), alpha=0.7)
        ax[1][0].plot(range(total_models), flosslist[k], label='F_Run '+str(k+1), alpha=0.7)
    ax[1][0].plot(range(total_models), favgaccuracylist, 'm', label='AVGaccu', linewidth=2)
    ax[1][0].plot(range(total_models), favglosslist, 'k', label='AVGloss', linewidth=2)
    ax[1][0].legend()
    ax[1][0].grid()
    for k in range(total_runs):
        ax[0][1].plot(range(total_models), mF1list[k], label='M_Run '+str(k+1), alpha=0.7)
        ax[0][1].plot(range(total_models), mlosslist[k], label='M_Run '+str(k+1), alpha=0.7)
    ax[0][1].plot(range(total_models), mavgF1list, 'b', label='AVGF1', linewidth=2)
    ax[0][1].plot(range(total_models), mavgF1list, 'bo', label='AVGF1', linewidth=2)
    ax[0][1].plot(range(total_models), mavglosslist, 'k', label='AVGloss', linewidth=2)
    ax[0][1].plot(range(total_models), mavglosslist, 'ko', label='AVGloss', linewidth=2)
    ax[0][1].legend()
    ax[0][1].grid()
    for k in range(total_runs):
        ax[1][1].plot(range(total_models), fF1list[k], label='F_Run '+str(k+1), alpha=0.7)
        ax[1][1].plot(range(total_models), flosslist[k], label='F_Run '+str(k+1), alpha=0.7)
    ax[1][1].plot(range(total_models), favgF1list, 'm', label='AVGF1', linewidth=2)
    ax[1][1].plot(range(total_models), favglosslist, 'k', label='AVGloss', linewidth=2)
    ax[1][1].legend()
    ax[1][1].grid()
    plt.show()

print("For Male case:")
print("\tBest accuracy is at index: {}".format(np.argmax(mavgaccuracylist)), end='\t')
print("\twith accuracy: {}".format(np.max(mavgaccuracylist)))
print("\tand,\n\tBest F1-score is at index: {}".format(np.argmax(mavgF1list)), end='\t')
print("\twith score: {}".format(max(mavgF1list)))


print("For Female case:")
print("\tBest accuracy is at index: {}".format(np.argmax(favgaccuracylist)), end='\t')
print("\twith accuracy: {}".format(np.max(favgaccuracylist)))
print("\tand,\n\tBest F1-score is at index: {}".format(np.argmax(favgF1list)), end='\t')
print("\twith score: {}".format(max(favgF1list)))

if np.argmax(mavgaccuracylist)!=np.argmax(mavgF1list) or np.argmax(favgaccuracylist)!=np.argmax(favgF1list):
    choice = input('Which score you want to go with?(1-accuracy, 2-F1):')
else:
    choice = '1'

if choice =='2':
    mbestPenalty = Penalty[int(np.argmax(mavgF1list)/len(G))]
    mbestGamma   = G[np.argmax(mavgF1list)%len(G)]
    fbestPenalty = Penalty[int(np.argmax(favgF1list)/len(G))]
    fbestGamma   = G[np.argmax(favgF1list)%len(G)]
else:
    mbestPenalty = Penalty[int(np.argmax(mavgaccuracylist)/len(G))]
    mbestGamma   = G[np.argmax(mavgaccuracylist)%len(G)]
    fbestPenalty = Penalty[int(np.argmax(favgaccuracylist)/len(G))]
    fbestGamma   = G[np.argmax(favgaccuracylist)%len(G)]

print("Selected Best model is:")
print("For Male:\tM_SVM_RBF_C"+str(mbestPenalty)+"_Gamma"+str(mbestGamma))
print("For Female:\tF_SVM_RBF_C"+str(fbestPenalty)+"_Gamma"+str(fbestGamma))

#Preparing Models for output
print("\nCalculating Test scores and saving models...")
mtestModel=svm.SVC(C=mbestPenalty, gamma=mbestGamma, kernel='rbf', class_weight='balanced', probability=True).fit(mX, mY)
name = "M_SVM_RBF_C"+str(mbestPenalty)+"_Gamma"+str(mbestGamma)
joblib.dump(mtestModel, "Output/"+name + '.pkl')
mtestAccuracy = accuracy_score(mtestlabels, mtestModel.predict(mtestvecs))
mtestF1 = f1_score(mtestlabels, mtestModel.predict(mtestvecs))
print('For Male:\nAccuracy on test data: ' + str(mtestAccuracy) + "\nF1 score on test data: " + str(mtestF1))

ftestModel=svm.SVC(C=fbestPenalty, gamma=fbestGamma, kernel='rbf', class_weight='balanced', probability=True).fit(fX, fY)
name = "F_SVM_RBF_C"+str(fbestPenalty)+"_Gamma"+str(fbestGamma)
joblib.dump(ftestModel, "Output/"+name + '.pkl')
ftestAccuracy = accuracy_score(ftestlabels, ftestModel.predict(ftestvecs))
ftestF1 = f1_score(ftestlabels, ftestModel.predict(ftestvecs))
print('For Female:\nAccuracy on test data: ' + str(ftestAccuracy) + "\nF1 score on test data: " + str(ftestF1))