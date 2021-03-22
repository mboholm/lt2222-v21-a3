import train #i.e. the python file used for training the model
import numpy as np
import argparse
import torch

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

	with open(path, mode="w") as file:
		for c in c_list:
			file.write(c)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Takes a PyTorch model and a text for testing and outputs 1. a text transformed by the model and 2. the accuracy of the model on that text (printed to the terminal). For 3. perplexity, provide (optional) argument p.")
	parser.add_argument("model", type=str, help="The model.")
	parser.add_argument("test_text", type=str, help="The test file.")
	parser.add_argument("output_txt", type=str, help="File (path) for the transformed test file, where vowels have been changed according to model.")
	args = parser.parse_args()

	vowels=train.vowels
	text, vocy = train.a(args.test_text)
	test_X, truth = train.b(text, vocy)

	model = torch.load(args.model)
	model.eval()
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
