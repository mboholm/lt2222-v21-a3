# LT2222 V21 Assignment 3

Your name: Max Boholm (gusbohom)

## Part 1
The functions `a()`, `b()`, and `g()` can be explained as follows:

`a()` is for loading a file and for pre-procesing it. `a()` takes a file *f* and returns:
1. *f* represented as a list of characters (including newline and space). Two start symbols (`<s>`) and two end symbols (`<e>`) are added to this list.
2. a "vocabulary", i.e. the set of character *types* in *f*.

`b()` is for producing a training set. It takes a text *t* (pre-processed and represented as a list of characters by `a()`) and a vocabulary *v* (the types of characters in the text) and returns a training set: two numpy arrays, one for the classes *y* and one for the features *X*. More precisely, `b()` traverses through the characters (c1, ... cn) in *t* and if a character (ci) is an element of `vowels` (i.e. 'y', 'é', 'ö', 'a', 'i', 'å', 'u', 'ä', 'e', 'o'), then: 
1. the index of ci in `vowles` is put in the list `gt` (later made into a numpy array; the class *y*)
2. the two characters prior ci and the two characters after ci are represented as concatenated one-hot representation of those characters with regard to *v*
3. this concatenated one-hot representation of features (characters in the context of ci) is added to the list `gr` (later made into a numpy array; the features *X*)

`g()` is a function for pairing characters with their index (or "key") with respect to a list (vocabulary), represented as one-hot representations. `g()`is implemented in `b()`. 

The commmand line arguments of `argparse` are as follows:

*   `m` is the file used for training the model. That is the training data.
*   `h` is the file for saving the model (PATH object).
*   `r` (optional; default=100) defines the epochs.
*   `k` (optional; default=200) defines the size of the hidden layer.

## Part 2
For `eval.py` the following design choices are the ones most in need of commenting: 

1. The function `a()` of `train.py` is used for loading and preprocessing the testing (evaluation) data. Thus, `train` is a module of `eval.py`.
2. Versions of `b()` and `g()` from `train.py` (i.e. `b_two()` and `g_two()` in `eval.py`) are adapted for the evaluation such that they handle new symbols of the test file which are not "recognized" by the model, i.e not part of the training features. Previously unseen chracters are ignored. In the concatenated vectors for features, the features not part of the training are treated as "all zero vectors". 
3. Through `argparse` there are three obligatory arguments for running the file `eval.py`: 
    1. the model (as saved by `train.py`). (Note: by convention, the file endings `.pt` or `.pth` can be used to save - and load - models; https://pytorch.org/tutorials/beginner/saving_loading_models.htm)
    2. the text file to be processed in evaluation; and 
    3. a path for the output text file where vowels have been replaced, as suggested by the model.
4. For the variable `predictions` in `eval.py` this is defined by a list comprehension (`[torch.topk(x, 1)[1] for x in model(torch.FloatTensor(test_X))]`), where:
    - predictions of the model based on the test features (`test_X`) are traversed,
    - so that every prediction (which is a tensor of log probabilities) is processed through `tourch.topk()`, 
    - which returns the top *k* values of a tensor and its indicies as tuples (thus the index `[1]` on that function output for getting the index of the maximum value)
    - (Note: when top *k* = 1, *k* is the maximum value)

## Part 3
I have trained models with the following parameters (a bash script for this procedure is found in the respository: `part3.sh`): 

|Parameter   |Value|Accuracy|
|------------|-----|--------|
|Hidden layer| k50 |   0.376|
|            | k100|   0.453|
|            | k150|   0.061|
|            | k200|   0.258|
|            | k250|   0.171|
|            | k300|   0.398|
|Epochs      | r50 |   0.238|
|            | r100|   0.371|
|            | r150|   0.404|
|            | r200|   0.182|
|            | r250|   0.236|
|            | r300|   0.097|

*Note:* default values for *k* and *r*: *k* (default) = 200; *r* (default) = 100.

Of these, the best model in terms of accuracy is the *k*=100 model (where *k* is the size of the hidden layer). This model is part of the respository as `best_model.pt`. The text "produced by" the model is called `text_best_model.txt` in respository.  

Furthermore, for Part 3, we are asked to:

> Describe any patterns you see, if there are any. Look at the output texts and make qualitative comments on the performances of the model.

*    *Observation 1:* The effects of altering *k* and *r* seems to be non-linear. For example, considering *k*, moving from *k*=50 to *k*=100, improves accuracy, while for *k*=150, the accuracy drops radically, recovering somewhat for *k*=200, and so on.  
*    *Observation 2:* The output text from "the best model" is in large parts incomprehensible, even for a native speaker of Swedish. Indeed, the original 19th centery text requires some extra effort for reading, but most of it is possible to grasp. The output text, on the other hand, mostly looks like "toy Scandinvian". As such, this text is illustrative of what accuracy=0.453 result in for a task like this one.
*    *Observation 3:* Vowels in (short) function words (e.g. *och*, *att*, *för*, *än*, *som*, and *till*) seems to be the ones best predicted by the model. The explanation for this is probably the commonality of these words in corpora, which result in high frequencies of class-features associations in training; in turn, providing better data for learning these associations.     

## Bonuses
I have not done any bonuses.

## Other notes
When running `train.py` and then using the model it produces in `eval.py`, I did have problems with an error that ended by:
	_pickle.UnpicklingError: invalid loan key ...
After some investigations on this error on the web, I came to the conclusion that this error could result from a incomplete saving by `train.py`. The problem was solved by producing a new model from `train.py`. 
