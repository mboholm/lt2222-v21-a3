# LT2222 V21 Assignment 3

Your name: Max Boholm (gusbohom)

## Part 1
The functions `a()`, `b()`, and `g()` can be explained as follows:

`a()` is for loading a file and for pre-procesing. It takes a file *f* and returns:
1. *f* represented as a list of characters (including newline and space); two start symbols (`<s>`) and two end symbols (`<e>`) are added to this list.
2. a "vocabulary", i.e. the set of character *types* in *f*

`b()` is for producing a traingin set. It takes a text *t* (pre -processed and represented as a list of characters) and a vocabulary *v* (the types of charcters in the text) and returns a training set: two numpy arrays, one for the class *y* and one for the features *X*. More precisely, `b()` traverses through the characters (c1, ... cn)in *t* and when ci is an element of `vowels` (i.e. 'y', 'é', 'ö', 'a', 'i', 'å', 'u', 'ä', 'e', 'o'), 
1. the index of ci in `vowles` is put in list `gt` (later made into a numpy array; the class *y*)
2. the two characters prior ci and the two characters after ci are represented as a collapsed one-hot representation of those characters with regard to *v*
3. this collapsed one-hot representation of features (characters in context of ci) is added to list `gr`  (later made into a numpy array; the features *X*)

`g()` is a function for pairing characters of a vocabulary *v* with a (constant) index (or "key") with respect to a list of *v* and based on those indecies producing collapsed one-hot representations for a list of characters (c1, ... cn). `g()`is implemented in `b()`. 

The commmand line arguments of `argparse` are as follows:

*   `m` is the file used for training the model.
*   `h` is the file for saving the model (PATH object).
*   `r` (optional; default=100) defines the epochs.
*   `k` (optional; default=200) defines the size of hidden layer.

## Part 2
For `eval.py` the following design choices are the ones most in need of commenting: 

1. The functions of `a()` and `b()` of `train.py` are used for preprocessing the data in `eval.py`; thus, `train` is a module of `eval.py`.
2. Through `argparse` there are three obligatory arguments to run the file: 
    i. the model (as saved by `train.py`); 
    ii. the text fiel to be processed for evaluation; and 
    iii. a path for the output text file where vowles have been replaced, as suggested by the model.
3. For the variable `predictions` this is defined by a list comprehension (`[torch.topk(x, 1)[1] for x in model(torch.FloatTensor(test_X))]`), where:
    - predictions of the model based on the test features (`test_X`) are traversed,
    - so that every prediction (which is a tensor of log probabilities) is processed through `tourch.topk()`, 
    - which returns the top *k* values of a tensor and its indecies as tuples (thus the index `[1]` on that function output for getting the index of the maximum value)
    - (note: when top *k* = 1, *k* is the maximum value)

## Part 3

for test.txt
EVAL
k50 (hidden layer)
Accuracy: 0.18.
k100 (hidden layer)
Accuracy: 0.06.
k150 (hidden layer)
Accuracy: 0.09.
k200 (hidden layer)
Accuracy: 0.09.
k250 (hidden layer)
Accuracy: 0.09.
k300 (hidden layer)
Accuracy: 0.08.

r50 (epochs)
Accuracy: 0.09.
r100 (epochs)
Accuracy: 0.04.
r150 (epochs)
Accuracy: 0.06.
r200 (epochs)
Accuracy: 0.17.
r250 (epochs)
Accuracy: 0.06.
r300 (epochs)
Accuracy: 0.05.


## Bonuses

## Other notes
