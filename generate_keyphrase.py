'''
This file comprises of some functions to generate the keyphrase from the system.
'''

#%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import generate_candidate
from pandas import DataFrame as df
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

def get_tf_keyphrase(candidates_list, number_keyphrases, csv_name):

	'''
	get_top_candidates(candidates_list = name of file,
						number_keyphrase = number of keyphrase,
						csv_name = name of file)
						
	this function is built to extract keyphrase from tfidf extraction
	'''
	
	best_candidates = []
	for doc in candidates_list:
        
		#sort candidates by tf-idf value
		sorted_candidates = sorted(doc, key = lambda x: x[1], reverse = True)[:number_keyphrases]
        
		#best_candidates.append(sorted_candidates)
		best_candidates.append([x for x,_ in sorted_candidates])
		
	#store extracted keyphrase into excel
	header = ['keyphrase %d' %x for x in range(1, number_keyphrases+1)]	
	predicted_keyphrases = df.from_records(best_candidates, columns = header)
	doc_id = ['doc %d' %x for x in range(1, len(best_candidates)+1)]
	predicted_keyphrases.insert(loc=0, column='document', value = doc_id)
	csv = predicted_keyphrases.to_csv('%s_data.csv' % csv_name, encoding = 'utf-8')

	return best_candidates



def get_top_candidates(candidates_list, number_keyphrases):

	'''
	get_top_candidates(candidates_list = name of file,
						number_keyphrase = )
	
	Extract top candidates based on number of keywords
	'''
	
	best_candidates = []
	for doc in candidates_list:
        
		#sort candidates by tf-idf value
		sorted_candidates = sorted(doc, key = lambda x: x[1], reverse = True)[:number_keyphrases]
        
		#best_candidates.append(sorted_candidates)
		best_candidates.append([x for x,_ in sorted_candidates])

	return best_candidates
	

def calculate_fmeasure(candidates_list, gold_data, number):

	'''
	calculate_fmeasure(candidates_list = name of candidate,
						gold_data = name of label,
						number = number of keyphrase)
	
	This function calculates and returns precision, recall, fmeasure.
	'''

    
	all_matches = []
	for n_doc in range(len(candidates_list)):
	
		#store all measure per document in dic
		value = {'tp': None, 'fp': None, 'fn': None, 'gold': None}
		
		#calculate how many gold standard per document
		value['gold'] = len(gold_data[n_doc])

		#counter true positive per document
		true_positive = 0
		for element_candidate in candidates_list[n_doc]:                    
			for element_goldkeyphrase in gold_data[n_doc]:
			
				#matched predicted keyword in gold keyphrase
				if element_candidate == element_goldkeyphrase:
					true_positive += 1
					
			#calculate true positive, false positive, false negative per document
			value['tp'] = int(true_positive) #matched pair
			value['fp'] = int(number - true_positive) #depend how many keyword should we use
			value['fn'] = int(value['gold'] - value['tp'])
			
		#return all metrics per document
		all_matches.append(value)

	#micro averaged -> all tp, fp, fn must be summed, not in average
	true_positive = sum(doc['tp'] for doc in all_matches)
	false_positive = sum(doc['fp'] for doc in all_matches)
	false_negative = sum(doc['fn'] for doc in all_matches)
    
    #matched / total number extracted keywords
	precision = float("{0:.2F}".format(true_positive / (false_positive + true_positive) * 100))
    
	#matched / total gold standard
	recall = float("{0:.2F}".format(true_positive / (false_negative + true_positive) * 100))
    
	# calculate with beta micro averaged precision, 
	# f beta = ((beta^2) + 1) * precision * recall / b^2 * precision + recall, because beta = 1
	# it can be calculated like f beta = (2 * precision * recall) / (precision + recall)
	f_measure = float("{0:.2F}".format(2 * (precision * recall) / (precision + recall)))
	
	#print("precision: {}, recall: {}, fmeasure: {}".format(precision, recall, f_measure))
	return precision, recall, f_measure
	

def probability_to_fmeasure(predict_proba, candidates, labels, models, number):

	'''
	probability_to_fmeasure(predict_proba = name of predict, 
							candidates = name of candidates, 
							labels = name of label, 
							models = name of models, 
							number = number of keyphrase)
							
	This function calculates the probability from machine learning into fmeasure
	'''

	#mapping predict probabilty value into candidates
	for model in range(0, len(predict_proba)):
		probability = []
		counter = 0
		for n_doc in range(len(candidates)):
			doc = []
			
			#add candidates and its predic probability into list
			for n_cand in range(len(candidates[n_doc])):
				doc.append((candidates[n_doc][n_cand][0], predict_proba[model][counter]))
				counter += 1
			probability.append(doc)
		
		#calculate fmeasure		
		fmeasure = calculate_fmeasure(get_top_candidates(probability, number), labels, number)
		print("Model %s: %.3f" % (models[model][0], fmeasure))

	return 'finish'
	

def predict_data(candidates, labels, train_data, test_data, n_keyphrase):
		
	'''
	predict_data(candidates = name of test candidates,
				labels = name of test label,
				train_data = name of train_data, 
				test_data = name of test_data, 
				n_keyphrase = number of keyphrase)
	'''
   
	#to skip candidates and label column in excel
	columns_to_skip = ['candidates', 'label']
	
	#load training and testing data into the memory
	x_train = pd.read_csv('%s_data.csv' % train_data, index_col=0, usecols = lambda x: x not in columns_to_skip).fillna(value = 0).values
	y_train = pd.read_csv('%s_data.csv' % train_data)['label'].fillna(value = 0).values
    
	x_test = pd.read_csv('%s_data.csv' % test_data, index_col=0, usecols = lambda x: x not in columns_to_skip).fillna(value = 0).values
	y_test = pd.read_csv('%s_data.csv' % test_data)['label'].fillna(value = 0).values

	#standard scaler feature scaling
	x_train = StandardScaler().fit_transform(x_train)
	x_test = StandardScaler().fit_transform(x_test)
	
	#Min max feature scaling
	#x_train = MinMaxScaler().fit_transform(x_train)
	#x_test = MinMaxScaler().fit_transform(x_test)
	
	#normalize feature scaling
	#x_train = Normalizer().fit_transform(x_train)
	#x_test = Normalizer().fit_transform(x_test)
	
	'''
	This system consists from 5 machine learning algorithms.
	However, the model is used Logistic Regression (default). 
	The other 4 algorithms are used for benchmark the algorithms.
	
	'''
	
	#please uncomment if want to test all the classifiers
	models  =  []
	models.append(('LR', LogisticRegression(C = 1)))
	models.append(('NB', GaussianNB()))
	models.append(('RF', RF(n_estimators = 20, max_depth = 11)))
	models.append(('AdaBoost', AdaBoostClassifier(n_estimators = 70, learning_rate = 1.0)))
	models.append(('Bagging', BaggingClassifier(n_estimators = 30)))
   
	
	#predict probability candidates is being selected as keyphrase (predict probability on class 1)
	all_predict_proba = []
	for name, model in models:
		all_predict_proba.append(model.fit(x_train, y_train).predict_proba(x_test)[:,1]) #only store probability on label '1'
    
	#calculate fmeasure 
	all_fmeasure = []
	for model in range(0, len(all_predict_proba)):
		probability = []
		counter = 0
		for n_doc in range(len(candidates)):
			doc = []
			
			#map candidates and their predict probability
			for n_cand in range(len(candidates[n_doc])):
				doc.append((candidates[n_doc][n_cand][0], all_predict_proba[model][counter]))
				counter += 1
			probability.append(doc)
			
		#extract n top keyphrase
		top_candidates = get_top_candidates(probability, n_keyphrase)
		
		#calculate the fmeasure
		fmeasure = calculate_fmeasure(top_candidates, labels, n_keyphrase)
		all_fmeasure.append((models[model][0], fmeasure))
	return all_fmeasure	


def feature_importance(train_data, test_data, name):
	
	'''
	feature_importance(train_data = name of train data excel,
						name = '' name of dataset, it must be strin)
	This fuction is intended to calculate which feature is important by Logistic Regression.
	'''
	
	#list all features
	features = ['tf-idf', 'length', 'supervised_key', 'distance', 
            'back_distance','spread', 'topic_rank', 'tf', 
            'DPM-index', 'GDC']
	
	#load the training and testing data to the memory
	x_train = pd.read_csv(train_data)[features].fillna(value = 0).values
	y_train = pd.read_csv(train_data)['label'].fillna(value = 0).values
	
	x_test = pd.read_csv(test_data)[features].fillna(value = 0).values
	y_test = pd.read_csv(test_data)['label'].fillna(value = 0).values

	#feature scaling 
	x_train = StandardScaler().fit_transform(x_train)
	x_test = StandardScaler().fit_transform(x_test)
	
	#initialise the Logistic Regression
	model = LogisticRegression(C = 1).fit(x_train, y_train)
	model.score(x_test, y_test)
		
	#store the coefficient from the classifier
	feature_importance = model.coef_[0]
	
	#map with the feature names
	values = sorted(zip(features, feature_importance), key=lambda x: x[1] * -1)

	#generate the plot 
	plt.figure()
	plt.title("Feature importance on %s" %name)
	plt.bar(range(len(values)), [val[1] for val in values],
       color="b", align="center")
	plt.xticks(range(len(values)), [val[0] for val in values])
	plt.xticks(rotation=70)
	plt.savefig('./figure/Feature importances (testing) on %s.png' %name)
	plt.show()
	
	return
	

def cross_validation(train_data):
#def cross_validation(labels, train_data):
	'''
	cross_validation(
				labels = name of train label,
				train_data = name of train_data
				)
				
	to check if the model is overfitting or no
	'''
   
	#to skip candidates and label column in excel
	columns_to_skip = ['candidates', 'label']
	
	#load train dataset into memory
	x_train = pd.read_csv('%s_data.csv' % train_data, index_col=0, usecols = lambda x: x not in columns_to_skip).fillna(value = 0).values
	y_train = pd.read_csv('%s_data.csv' % train_data)['label'].fillna(value = 0).values
	
	x_train = StandardScaler().fit_transform(x_train)
	#selection of machine learning algorithms
	
	models  =  []
	models.append(('LR', LogisticRegression(C = 1)))
	models.append(('NB', GaussianNB()))
	models.append(('RF', RF(n_estimators = 20, max_depth = 11)))
	models.append(('AdaBoost', AdaBoostClassifier(n_estimators = 70, learning_rate = 1.0)))
	models.append(('Bagging', BaggingClassifier(n_estimators = 30)))

	
	#print average accuracy from cross validation per each model
	scoring = 'accuracy'
	results = []
	names = []
	###measure accuracy with k-fold
	print("Accuracy on training data with Cross-validation:")
	for name, model in models:
	
		#split training data into 10 fold, and calculate the score by accuracy 
		cv_results = model_selection.cross_val_score(model, x_train, y_train, cv = StratifiedKFold(n_splits=10, random_state = None), scoring = scoring)
		results.append(cv_results)
		names.append(name)
		
		#print the result, its mean and standard deviation
		msg = "%s: %.3f (%.3f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)
	
	return 'done'

	
def feature_selection(train_data):

	'''
	This function is intended to calculate the optimum number of features
	'''

	#to skip candidates and label column in excel
	columns_to_skip = ['candidates', 'label']
	
	#create dense matrix
	x_train = pd.read_csv('%s_data.csv' % train_data, index_col=0, usecols = lambda x: x not in columns_to_skip).fillna(value = 0).values
	y_train = pd.read_csv('%s_data.csv' % train_data)['label'].fillna(value = 0).values

	#feature scaling
	x_train = StandardScaler().fit_transform(x_train)

	#initialise the classifier, RFECV and fit the train data into the package
	model = LogisticRegression(C = 1)
	rfecv = RFECV(estimator = model, step = 1, cv = StratifiedKFold(10), scoring = 'accuracy')
	rfecv.fit(x_train, y_train)
	print("Optimal number of features : %d" % rfecv.n_features_)

	# Plot number of features VS. cross-validation scores
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
	plt.show()
	return plt
	
	
def get_predicted_keyphrases(candidates, train_data, test_data, csv_name, n_keyphrase):
   
	'''
	predict_data(candidates = name of candidates,
				train_data = name of train data,
				test_data = name of test data,
				csv_name = name of csv, 
				n_keyphrase = number of keyphrase)
	
	To save predicted keywords on excel.
	'''
	
	#to extract the content without candidates and label
	columns_to_skip = ['candidates', 'label']
	
	#read csv of training and testing data without its index, and fill null column with 0
	x_train = pd.read_csv('%s_data.csv' % train_data, usecols = lambda x: x not in columns_to_skip).fillna(value = 0).values
	y_train = pd.read_csv('%s_data.csv' % train_data)['label'].fillna(value = 0).values
    
	x_test = pd.read_csv('%s_data.csv' % test_data, usecols = lambda x: x not in columns_to_skip).fillna(value = 0).values
	y_test = pd.read_csv('%s_data.csv' % test_data)['label'].fillna(value = 0).values
	
	#feature scaling
	x_train = StandardScaler().fit_transform(x_train)
	x_test = StandardScaler().fit_transform(x_test)
	
	#define classifier for the model
	predict_proba = LogisticRegression(C = 1).fit(x_train, y_train).predict_proba(x_test)[:,1]

	#mapping predict probability into candidates
	probability = []
	counter = 0
	for n_doc in range(len(candidates)):
		doc = []
		for n_cand in range(len(candidates[n_doc])):
			doc.append((candidates[n_doc][n_cand][0], predict_proba[counter]))
			counter += 1
		probability.append(doc)
		
	#get top candidates, and store into csv
	top_candidates = get_top_candidates(probability, n_keyphrase)
	
	#create column in excel
	labels = ['keyphrase %d' %x for x in range(1, n_keyphrase+1)]			
	
	#convert top candidates into a data frame
	predicted_keyphrases = df.from_records(top_candidates, columns=labels)
	
	#create doc index
	doc_id = ['doc %d' %x for x in range(1, len(top_candidates)+1)]
	predicted_keyphrases.insert(loc=0, column='document', value=doc_id)
	
	#store the result into a excel
	csv = predicted_keyphrases.to_csv('%s_data.csv' % csv_name, encoding = 'utf-8')
			
	return 'done'