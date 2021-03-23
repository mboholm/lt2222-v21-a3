import os
import sys
import train #i.e. the python file used for training the model
import numpy as np
import argparse
import torch

def unknown(features_for_instance,vocy_list):
	"""
	Takes a list of features and replaces those features that are not recognized by the model (as defined by vocy_list) with the integer 0.
	"""

	replace_unknown=[]
	for f in features_for_instance:
		if f in vocy_list:
			replace_unknown.append(f)
		else:
			replace_unknown.append(0) #the integer 0 is used for unknowns

	return replace_unknown

def g_two(x, voc):
    """
    A version of g(), from train.py, which return "no-hot" vectors for unknown features (i.e. features not encountered when training). 

    Takes a list of features (x) and a "vocabulary" (voc), and returns a collapsed one-hot representation of those symbols with regard to the vocabulary.
    """

    z = np.zeros(len(voc))
    if x!=0:
	    z[voc.index(x)] = 1
    return z

def b_two(text, voc):
    """
    A version of b(), from train.py, which handles unknown features (i.e. features not encountered when training). 

    Takes a text and a vocabulary (voc) and returns features (test_X) and true classes (truth)
    """

    gt = [] #this is the class
    gr = [] #this it the features
    for v in range(len(text) - 4):
        if text[v+2] not in vowels: #first two tokens are startsymbols ("<s>") 
            continue
        
        h2 = vowels.index(text[v+2])
        gt.append(h2)
        r = np.concatenate([g_two(x, voc) for x in unknown([text[v], text[v+1], text[v+3], text[v+4]], voc)]) 
        gr.append(r) 

    return np.array(gr), np.array(gt) #features, truth

def accuracy(truth, predictions):
	"""
	Calculates the accuracy of the classifier and prints to terminal.
	"""

	tp=0 #a counter for true positives
	for t, p in zip(truth, predictions):
		if t==p:
			tp+=1

	accuracy=tp/len(truth)

	print("Accuracy: {}.".format(round(accuracy, 3)))

def transformer(text, predictions):
	"""
	Takes a text and replaces its vowles with a list of predicted ones.
	"""

	storage=predictions[:] #since pop is used we need a clone not to "empty" the predictions variable

	optimus_prime = []

	for c in text:
		if c in vowels:
			x=storage.pop(0)
			optimus_prime.append(vowels[x])
		else:
			optimus_prime.append(c)

	return optimus_prime

def list_to_file(c_list, path):
	"""
	Takes a list of characters (c_list) and write that to a file (path). 
	"""

	with open(path, mode="w", encoding="utf-8") as file:
		for c in c_list:
			file.write(c)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Takes a PyTorch model and a text for testing and outputs 1. a text transformed by the model and 2. the accuracy of the model on that text (printed to the terminal).")
	parser.add_argument("model", type=str, help="The model.")
	parser.add_argument("test_text", type=str, help="The test file.")
	parser.add_argument("output_txt", type=str, help="File (path) for the transformed test file, where vowels have been changed according to model.")
	args = parser.parse_args()

	vowels=train.vowels

	model = torch.load(args.model)
	model.eval()

	vocy = model.vocab #to build features and true class for evaluation, the vocabulary used for building the model is used (not the vocabulary of the test text, per se)

	text = train.a(args.test_text)[0] #uses a() of train.py
	test_X, truth = b_two(text, vocy)

	predictions=[torch.topk(x, 1)[1] for x in model(torch.FloatTensor(test_X))]
	#there is a lot going on here, which can be clarified as follows: 
	#the features (test_X, as tensor) are processed through the model
	#the output O of this process is an iterable of tensors, 
	#where every values is the output of LogSoftmax.
	#O is traversed and for every tensor T,
	#the index of the maximum value of T (top k=1)
	#is compiled in the list comprehension named "predictions"
	#(note: topk() returns a tuple: the maximum value and the index of tha value)

	transformation = transformer(text, predictions)
	list_to_file(transformation, args.output_txt)

	accuracy(truth, predictions)
