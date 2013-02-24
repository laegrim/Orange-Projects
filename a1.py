#! /user/bin/env python

import Orange
import sys
import os

if len(sys.argv) != 2:
    sys.exit('Usage: %s training-data' % sys.argv[0])

    
trainPath = os.getcwd() + '/' + sys.argv[1]

if not os.path.exists(os.getcwd() + '/' + sys.argv[1]):
    sys.exit('Error: training-data was not found in specified location' % sys.argv[1])

#trainFile = open(trainPath, 'r')

outFile = open('output.txt', 'w')

outFile.write(trainPath + '\n')

trainData = Orange.data.Table(trainPath)

if (trainData == False):
    sys.exit('Error: unable to import data from a file')
elif (trainData.has_missing_values() == True):
    sys.exit('Error: some data has missing values')
elif(trainData.has_missing_classes() == True):
    sys.exit('Error: some data is missing class values')

learners = [Orange.classification.knn.kNNLearner(k=1), Orange.classification.knn.kNNLearner(k=5), Orange.classification.knn.kNNLearner(k=10), Orange.classification.tree.SimpleTreeLearner(), Orange.classification.tree.C45Learner()]

cv = Orange.evaluation.testing.cross_validation(learners, trainData, folds=10, stratified=True, store_classifiers=True)

outFile.write('Accuracy on knn (k=1): %.4f\n' % Orange.evaluation.scoring.CA(cv)[0])
outFile.write('Accuracy on knn (k=5): %.4f\n' % Orange.evaluation.scoring.CA(cv)[1])
outFile.write('Accuracy on knn (k=10): %.4f\n' % Orange.evaluation.scoring.CA(cv)[2])
outFile.write('Accuracy on Random Tree: %.4f\n' % Orange.evaluation.scoring.CA(cv)[3])
outFile.write('Accuracy on c4.5 Tree : %.4f\n' % Orange.evaluation.scoring.CA(cv)[4])

#for ex in Orange.evaluation.scoring.CA(cv):
    #outFile.write('Cross-Validation accuracy on Training Data: %.2f \n' % ex)

totAvgHeight = 0.0
localAvgHeight = 0.0
trees = [3, 4]

for x in trees:
    for y in cv.classifiers[x]:
	root = y.tree()
	branchSize = root.branch_sizes()
	for z in branchSize: 
	    localAvgHeight+=z
	localAvgHeight = localAvgHeight / len(branchSize)
	totAvgHeight += localAvgHeight
    totAvgHeight = totAvgHeight / 10
    outFile.write('Average Tree Height: %.4f\n' % totAvgHeight)
    totAvgHeight = 0.0
    localAvgHeight = 0.0
	

outFile.close()

sys.exit()


