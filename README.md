# LT2222 V21 Assignment 3

Your name: Max Boholm (gusbohom)

## Part 1
The functions `a()`, `b()`, and `g()` can be explained as follows:

`a()` is for loading a file and for pre-procesing it. `a()` takes a file *f* and returns:
1. *f* represented as a list of characters (including newline and space). Two start symbols (`<s>`) and two end symbols (`<e>`) are added to this list.
2. a "vocabulary", i.e. the set of character *types* in *f*.

`b()` is for producing a training set. It takes a text *t* (pre-processed and represented as a list of characters by `a()`) and a vocabulary *v* (the types of charcters in the text) and returns a training set: two numpy arrays, one for the classes *y* and one for the features *X*. More precisely, `b()` traverses through the characters (c1, ... cn) in *t* and the caracter (ci) is an element of `vowels` (i.e. 'y', 'é', 'ö', 'a', 'i', 'å', 'u', 'ä', 'e', 'o'), 
1. the index of ci in `vowles` is put in list `gt` (later made into a numpy array; the class *y*)
2. the two characters prior ci and the two characters after ci are represented as concatenated one-hot representation of those characters with regard to *v*
3. this concatenated one-hot representation of features (characters in the context of ci) is added to list `gr` (later made into a numpy array; the features *X*)

`g()` is a function for pairing characters of a vocabulary *v* with their index (or "key") with respect to a list (for *v*). Based on the indecies of the caracters/features, a concatenated one-hot representations for a list of characters (c1, ... cn) is produced. `g()`is implemented in `b()`. 

The commmand line arguments of `argparse` are as follows:

*   `m` is the file used for training the model. That is the training data.
*   `h` is the file for saving the model (PATH object).
*   `r` (optional; default=100) defines the epochs.
*   `k` (optional; default=200) defines the size of the hidden layer.

## Part 2
For `eval.py` the following design choices are the ones most in need of commenting: 

1. The function `a()` of `train.py` is used for loading and preprocessing the testing (evaluation) data. Thus, `train` is a module of `eval.py`.
2. Versions of `b()` and `g()` from `train.py` are adapted for the evaluation, such that they handle new symbols of the test file which are not "recognized" by the model, i.e not part of the traingin features. Previously unseen chracters are ignored. In the concatenated vectors for features, chracters not part of the traingin are treated as "all zero vectors". 
3. Through `argparse` there are three obligatory arguments to run the file `eval.py`: 
    i. the model (as saved by `train.py`); by convention, the file endings `.pt` or `.pth` can be used to save models (https://pytorch.org/tutorials/beginner/saving_loading_models.htm)
    ii. the text file to be processed in evaluation; and 
    iii. a path for the output text file where vowles have been replaced, as suggested by the model.
4. For the variable `predictions`in `eval.py` this is defined by a list comprehension (`[torch.topk(x, 1)[1] for x in model(torch.FloatTensor(test_X))]`), where:
    - predictions of the model based on the test features (`test_X`) are traversed,
    - so that every prediction (which is a tensor of log probabilities) is processed through `tourch.topk()`, 
    - which returns the top *k* values of a tensor and its indecies as tuples (thus the index `[1]` on that function output for getting the index of the maximum value)
    - (note: when top *k* = 1, *k* is the maximum value)

## Part 3
I have trained models with the following parameters (there is a bash script for this uploded in the respository): 

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


Of these the best model in terms of accuracy is the k=100 model (where k is size of hidden layer).
 
Note default values for k and r:
k(default=200
r(default)=100

## Bonuses
I have not done any bonuses.

## Other notes
When running `train.py` and then using the model it prodces in `eval.py` I did have problems with an error which ended by:
    _pickle.UnpicklingError: invalid loan key ...
After some investigations on the web, I came to the conclusion that this error could result from a incomplete saving by `train.py`; thus, problems with loading a corrupt file. 
