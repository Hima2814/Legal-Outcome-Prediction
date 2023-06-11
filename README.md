# Legal-Outcome-Prediction
A Deep Learning based Approach for Legal Outcome Prediction
Legal Outcome Prediction or Legal Judgement Prediction is the process of making predictions of the possible outcome of a legal proceeding based on its case file. The models proposed in this work is built over the concepts of natural language processing and deep learning. 
There were about thirteen models under different categories that were implemented. This work discusses the implementation and the results obtained for each of the models. The models ranged from simple perceptron model to convolutional network models to encoder-based models. 
The models like Multi-layer Perceptron (MLP), Convolutional Neural Network (CNN) model, BERT, LegalBERT, DistilBERT, RoBERTa and the different combination of these encoder-based models was implemented. The input to these models is the case files and their corresponding judgement status. There were four different classes of petitions and appeals that were submitted to the court. 
The efficiency of these models was tested in terms of various performance metrics like training accuracy, testing accuracy, training loss and validation loss.

# Dataset
The legal outcome prediction models were built over the data that was extracted from the Indian Kanoon website.This website is a search engine for Indian law and all the case files and court proceedings in India are available on this portal.

The dataset was manually prepared by gathering various case files and their corresponding judgement status’ and currently has 600 records.

The judgement status has four different categories namely, appeal allowed, appeal dismissed, petition dismissed and petition allowed and all these four categories were encoded to a numerical value of 0, 1, 2, 3 respectively.
