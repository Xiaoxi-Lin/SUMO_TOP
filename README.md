# SUMO_TOP
An application od topologiccal data analysis in predicting sumoylation sites

# Requirement
requirement.txt

# Data
1. dataset1.csv: Sumoylation sites with corresponding uniprot entry in dataset1
2. dataset2.csv：Positive samples of dataset2
3. DATASET1_10fold_crossvalidation.csv: The information of labels and features of dataset1 for 10-fold cross validation
4. DATASET2_independentTest_trainingset.csv : The training set with labels and features of dataset2 for independent set test with test-size 0.3
5. DATASET2_independentTest_testingset.csv : The testing set with labels and features of dataset2 for independent set test with test-size 0.3

# Scripts
1. binary_classifiers.py: Binary classifiers 
2. extractTopologicalFeatures.py: Features construted from topological data analysis for predicting sumoylation sites
3. reproduceResults.py: Functions to reproduce the results, including pairwise sequence identity, undersampling and validation strategies
4. interpretabilityOfFeatures.py: Functions including f-score computation, pie graph, ROC curves and bar graph

# Contact 
Please feel free to contact us if you need any help: lxx0217@mail.dlut.edu.cn
