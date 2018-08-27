
This project aims to create a generic pipeline for automatically extracting keywords in scientific papers, abstracts and news articles.

There are 4 folders on this project. 
   figure - storing the result of generated figure.
   csv - storing the training dataset, testing dataset, and result of predicted keywords
   pickle - storing the processed result from certain result on this system

1. There are 3 datasets which involved in this project. 
   Semeval text file version is available on https://github.com/snkim/AutomaticKeyphraseExtraction
   Semeval xml version is available on https://github.com/snkim/AutomaticKeyphraseExtraction
   Inspec xml is available on https://github.com/snkim/AutomaticKeyphraseExtraction
   500N-KPCrowd is available on https://github.com/boudinfl/ake-datasets/tree/master/datasets/500N-KPCrowd
   Before running the system, if the pickle of those datasets have not been available, 
	please download the dataset first, then run the certain part of code in each file of model. 
   Moreover, if the pickle for topicRank has not been available too, please follow the next step.

2. This project implemented TopicRank as one of features, however there is several modification on it. 
   Therefore, the first step is to install pytopicrank from https://github.com/smirnov-am/pytopicrank. 
   Replace the original 'topicrank.py' to 'topicrank.py' from this project. 
   To obtain keywords from topicRank, please run 'generate_topic_rank.py'. 
   This code will produce topic rank keywords and store it to pickle.

3.  There are several jupyter notebooks on the system.
    1. TF-IDF text file extraction.ipynb: keyphrase extraction using TF-IDF on Semeval txt version.
    2. TF-IDF XML extraction.ipynb: keyphrase extraction using TF-IDF on Semeval xml version.  
    3. Machine learning - Text file.ipynb: keyphrase extraction on Semeval text file version using machine learning. 
       This model employes 2 linguistic filters, noun phrase and n-gram with combined label. 
    4. Machine learning - XML.ipynb: keyphrase extraction on Semeval XML version using machine learning. 
       This model employes 2 linguistic filters, noun phrase and n-gram with combined label.
    5. Evaluation on Semeval.ipynb: the selected pipeline (Semeval XML noun phrase) version with author, reader, combined label. 
       It is intended to measure performance of the model on all evaluation.
    6. Evaluation on Inspec.ipynb: the pipeline with noun phrase and ngram filter. 
       The pipeline using noun phrase, but ngrams is included to evaluate the performance of noun phrase.
    7. Evaluation on 500N-KPCrowd.ipynb: the pipeline with noun phrase and ngram filter. 
       The pipeline using noun phrase, but ngrams is included to evaluate the performance of noun phrase.
    
