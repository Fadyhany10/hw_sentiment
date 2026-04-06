__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2022"

from collections import Counter
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch.nn as nn

from torch_rnn_classifier import TorchRNNClassifier
from torch_tree_nn import TorchTreeNN
import sst
import utils
%load_ext autoreload
%autoreload 2

SST_HOME = os.path.join('data', 'sentiment')

sst_train = sst.train_reader(SST_HOME)

sst_train.shape[0]

sst_dev = sst.dev_reader(SST_HOME)

bakeoff_dev = sst.bakeoff_dev_reader(SST_HOME)

bakeoff_dev.sample(3, random_state=1).to_dict(orient='records')

bakeoff_dev.label.value_counts()

def unigrams_phi(text):
    return Counter(text.split())

def fit_softmax_classifier(X, y):
    mod = LogisticRegression(
        fit_intercept=True,
        solver='liblinear',
        multi_class='ovr')
    mod.fit(X, y)
    return mod

softmax_experiment = sst.experiment(
    sst.train_reader(SST_HOME),   # Train on any data you like except SST-3 test!
    unigrams_phi,                 # Free to write your own!
    fit_softmax_classifier,       # Free to write your own!
    assess_dataframes=[sst_dev, bakeoff_dev]) # Free to change this during development!

def rnn_phi(text):
    return text.split()


def fit_rnn_classifier(X, y):
    sst_glove_vocab = utils.get_vocab(X, mincount=2)
    mod = TorchRNNClassifier(
        sst_glove_vocab,
        early_stopping=True)
    mod.fit(X, y)
    return mod

rnn_experiment = sst.experiment(
    sst.train_reader(SST_HOME),
    rnn_phi,
    fit_rnn_classifier,
    vectorize=False,  # For deep learning, use `vectorize=False`.
    assess_dataframes=[sst_dev, bakeoff_dev])

def find_errors(experiment):
    """Find mistaken predictions.

    Parameters
    ----------
    experiment : dict
        As returned by `sst.experiment`.

    Returns
    -------
    pd.DataFrame

    """
    dfs = []
    for i, dataset in enumerate(experiment['assess_datasets']):
        df = pd.DataFrame({
            'raw_examples': dataset['raw_examples'],
            'predicted': experiment['predictions'][i],
            'gold': dataset['y']})
        df['correct'] = df['predicted'] == df['gold']
        df['dataset'] = i
        dfs.append(df)
    return pd.concat(dfs)

softmax_analysis = find_errors(softmax_experiment)

rnn_analysis = find_errors(rnn_experiment)

# Examples where the softmax model is correct, the RNN is not,
# and the gold label is 'positive'

error_group = analysis[
    (analysis['predicted_x'] == analysis['gold'])
    &
    (analysis['predicted_y'] != analysis['gold'])
    &
    (analysis['gold'] == 'positive')
]

error_group.shape[0]

for ex in error_group['raw_examples'].sample(5, random_state=1):
    print("="*70)
    print(ex)


# Token-level differences [1 point]*************

df = pd.DataFrame([
        {'sentence': 'a a b'},
        {'sentence': 'a b a'},
        {'sentence': 'a a a b.'}])

sum([Counter(item.split(" ")) for item in df["sentence"].values], Counter())

def get_token_counts(df):

    counts = sum([Counter(item.split(" ")) for item in df["sentence"].values], Counter())
    return pd.Series(counts).sort_values(ascending=False)


def test_get_token_counts(func):
    df = pd.DataFrame([
        {'sentence': 'a a b'},
        {'sentence': 'a b a'},
        {'sentence': 'a a a b.'}])
    result = func(df)
    for token, expected in (('a', 7), ('b', 2), ('b.', 1)):
        actual = result.loc[token]
        assert actual == expected, \
            "For token {}, expected {}; got {}".format(
            token, expected, actual)

if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_get_token_counts(get_token_counts)

# Training on some of the bakeoff data [1 point]******************

def run_mixed_training_experiment(wrapper_func, bakeoff_train_size):

    bakeoff_dev_train, bakeoff_dev_eval = bakeoff_dev[:bakeoff_train_size], bakeoff_dev[bakeoff_train_size:]

    experiment = sst.experiment(
    [sst_train, bakeoff_dev_train],
    unigrams_phi,
    wrapper_func,
    vectorize=True,  # For deep learning, use `vectorize=False`.
    assess_dataframes=[sst_dev, bakeoff_dev_eval])

    return experiment

def test_run_mixed_training_experiment(func):
    bakeoff_train_size = 1000
    experiment = func(fit_softmax_classifier, bakeoff_train_size)

    assess_size = len(experiment['assess_datasets'])
    assert len(experiment['assess_datasets']) == 2, \
        ("The evaluation should be done on two datasets: "
         "SST3 and part of the bakeoff dev set. "
         "You have {} datasets.".format(assess_size))

    bakeoff_test_size = bakeoff_dev.shape[0] - bakeoff_train_size
    expected_eval_examples = bakeoff_test_size + sst_dev.shape[0]
    eval_examples = sum(len(d['raw_examples']) for d in experiment['assess_datasets'])
    assert expected_eval_examples == eval_examples, \
        "Expected {} evaluation examples; got {}".format(
        expected_eval_examples, eval_examples)

if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_run_mixed_training_experiment(run_mixed_training_experiment)


# A more powerful vector-averaging baseline [2 points]****************

from torch_shallow_neural_classifier import TorchShallowNeuralClassifier

def fit_shallow_neural_classifier_with_hyperparameter_search(X, y):

    basemod = TorchShallowNeuralClassifier(
        early_stopping=True
    )
    cv = 3
    param_grid = {
        'hidden_dim': [50, 100, 200],
        'hidden_activation': [nn.Tanh(), nn.ReLU()]
    }
    bestmod = utils.fit_classifier_with_hyperparameter_search(
        X, y, basemod, cv, param_grid)

    return bestmod

# Testing the implementation
DATA_HOME = 'data'

GLOVE_HOME = os.path.join(DATA_HOME, 'glove.6B')

glove_lookup = utils.glove2dict(
    os.path.join(GLOVE_HOME, 'glove.6B.300d.txt'))

def vsm_phi(text, lookup, np_func=np.mean):
    """Represent `text` as a combination of the vector of its words.

    Parameters
    ----------
    text : str

    lookup : dict
        From words to vectors.

    np_func : function (default: np.sum)
        A numpy matrix operation that can be applied columnwise,
        like `np.mean`, `np.sum`, or `np.prod`. The requirement is that
        the function take `axis=0` as one of its arguments (to ensure
        columnwise combination) and that it return a vector of a
        fixed length, no matter what the size of the text is.

    Returns
    -------
    np.array, dimension `X.shape[1]`

    """
    allvecs = np.array([lookup[w] for w in text.split() if w in lookup])
    if len(allvecs) == 0:
        dim = len(next(iter(lookup.values())))
        feats = np.zeros(dim)
    else:
        feats = np_func(allvecs, axis=0)
    return feats

def glove_phi(text, np_func=np.mean):
    return vsm_phi(text, glove_lookup, np_func=np_func)



bestmod = sst.experiment(
    sst_train,
    glove_phi,
    fit_shallow_neural_classifier_with_hyperparameter_search,
    assess_dataframes=[sst_dev],
    vectorize=False
)


# BERT encoding [2 points]**********************

from transformers import BertModel, BertTokenizer
import vsm

# Instantiate a Bert model and tokenizer based on `bert_weights_name`:
bert_weights_name = 'bert-base-uncased'

bert_tokenizer = BertTokenizer.from_pretrained(bert_weights_name)
bert_model = BertModel.from_pretrained(bert_weights_name)

def hf_cls_phi(text):
    # Get the ids. `vsm.hf_encode` will help; be sure to
    # set `add_special_tokens=True`.
 
    encoding = vsm.hf_encode(text=text, tokenizer=bert_tokenizer, add_special_tokens=True)

    # Get the BERT representations. `vsm.hf_represent` will help:

    reps = vsm.hf_represent(batch_ids=encoding, model=bert_model, layer=-1)


    # Index into `reps` to get the representation above [CLS].
    # The shape of `reps` should be (1, n, 768), where n is the
    # number of tokens. You need the 0th element of the 2nd dim:

    cls_rep = reps[:,0,:].squeeze()

    # These conversions should ensure that you can work with the
    # representations flexibly. Feel free to change the variable
    # name:
    return cls_rep.cpu().numpy()

def test_hf_cls_phi(func):
    rep = func("Just testing!")

    expected_shape = (768,)
    result_shape = rep.shape
    assert rep.shape == (768,), \
        "Expected shape {}; got {}".format(
        expected_shape, result_shape)

    # String conversion to avoid precision errors:
    expected_first_val = str(0.1709)
    result_first_val = "{0:.04f}".format(rep[0])

    assert expected_first_val == result_first_val, \
        ("Unexpected representation values. Expected the "
        "first value to be {}; got {}".format(
            expected_first_val, result_first_val))

if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_hf_cls_phi(hf_cls_phi)


bestmod = sst.experiment(
    sst_train,
    hf_cls_phi,
    fit_shallow_neural_classifier_with_hyperparameter_search,
    assess_dataframes=[sst_dev],
    vectorize=False
)

# Your original system [3 points]*****************

# PLEASE MAKE SURE TO INCLUDE THE FOLLOWING BETWEEN THE START AND STOP COMMENTS:
#   1) Textual description of your system.
#   2) The code for your original system.
#   3) The score achieved by your system in place of MY_NUMBER.
#        With no other changes to that line.
#        You should report your score as a decimal value <=1.0
# PLEASE MAKE SURE NOT TO DELETE OR EDIT THE START AND STOP COMMENTS

# NOTE: MODULES, CODE AND DATASETS REQUIRED FOR YOUR ORIGINAL SYSTEM
# SHOULD BE ADDED BELOW THE 'IS_GRADESCOPE_ENV' CHECK CONDITION. DOING
# SO ABOVE THE CHECK MAY CAUSE THE AUTOGRADER TO FAIL.

# START COMMENT: Enter your system description in this cell.
# My peak score was: 0.619
if 'IS_GRADESCOPE_ENV' not in os.environ:
    """This solution experiments with using TorchRNNClassifier with pretrained contexual word representations
    of text tokens based on BERT base model.

    The SST3 and Bakeoff dev datasets are used for training the system. A hyperparameter search is performed using
    SST3-train and part of bakeoff data and assessment of the best model is performed by using SST3-dev and rest
    of the bakeoff data. The following parameters are considered for the best hyperparameter search:
    - hidden_dim: Dimensionality of the RNN hidden layer
    - bidirectional: Unidirectional / bidirectional nature of the RNN model
    - num_layers: Number of layers in the RNN model
    - classifier_activation: Activation function of the classifier model

    The following are the best hyperparameters after doing the cross validation with three stratified shuffle
    splits:
    {'bidirectional': True, 'classifier_activation': ReLU(), 'hidden_dim': 200, 'num_layers': 2}.

    The final model is trained using all the data, i.e., SST3-train, SST3-dev and Bakeoff dev.
    """
     # Imports
    from torch_rnn_classifier import TorchRNNClassifier
    from transformers import BertModel, BertTokenizer
    import vsm
    import sst
    import utils

    # Datasets
    bakeoff_dev = sst.bakeoff_dev_reader(SST_HOME) # Sentences from resturant reviews
    sst_train = sst.train_reader(SST_HOME) # Sentences from movie reviews
    sst_dev = sst.dev_reader(SST_HOME)


    # Best RNN classifier using bert embeddings
    bert_weights_name = 'bert-base-uncased'

    bert_tokenizer = BertTokenizer.from_pretrained(bert_weights_name)
    bert_model = BertModel.from_pretrained(bert_weights_name)

    def bert_phi(text):
        """Get the BERT last hidden state embeddings text tokens.
        """
        encoding = vsm.hf_encode(text=text, tokenizer=bert_tokenizer, add_special_tokens=True)
        reps = vsm.hf_represent(batch_ids=encoding, model=bert_model, layer=-1)
        return reps.squeeze(0).numpy()


    def system_RNN_bert(X, y):
        """Gridsearch on different hyperparameters of an RNN model using token embeddings based on BERT base
        model.
        """
        basemod = TorchRNNClassifier(
            vocab = [],
            early_stopping=True,
            use_embedding=False
        )
        cv = 3
        ## Add the bidirectional and num_layers in the grid search parameters in
        ## compariosn to typically done in course
        param_grid = {
            'hidden_dim': [100, 200],
            'classifier_activation': [nn.Tanh(), nn.ReLU()],
            'bidirectional': [True, False],
            'num_layers': [1, 2]
        }

        bestmod = utils.fit_classifier_with_hyperparameter_search(
            X, y, basemod, cv, param_grid)

        return bestmod

    # Use part of the bake off dev set for training so that model can learn the trends specific to bakeoff dataset
    bakeoff_train_size = bakeoff_dev.shape[0] // 2
    bakeoff_dev_train, bakeoff_dev_eval = bakeoff_dev[:bakeoff_train_size], bakeoff_dev[bakeoff_train_size:]

    bestmod_bert = sst.experiment(
        [sst_train, bakeoff_dev_train],
        bert_phi,
        system_RNN_bert,
        assess_dataframes=[sst_dev, bakeoff_dev_eval],
        vectorize=False
    )

    # Train the best model on all the data.

    # Train the best model on complete dataset
    train = sst.build_dataset(
            [sst_train, sst_dev, bakeoff_dev],
            bert_phi,
            vectorizer=None,
            vectorize=False)

    X_train = train['X']
    y_train = train['y']

    bestmod_bert["model"] = bestmod_bert["model"].fit(X_train, y_train)

# STOP COMMENT: Please do not remove this comment.

# Bakeoff [1 point]****************

def predict_one_softmax(text):
    # Singleton list of feature dicts:
    feats = [softmax_experiment['phi'](text)]
    # Vectorize to get a feature matrix:
    X = softmax_experiment['train_dataset']['vectorizer'].transform(feats)
    # Standard sklearn `predict` step:
    preds = softmax_experiment['model'].predict(X)
    # Be sure to return the only member of the predictions,
    # rather than the singleton list:
    return preds[0]

def predict_one_rnn(text):
    # List of tokenized examples:
    X = [bestmod_bert['phi'](text)]
    # Standard `predict` step on a list of lists of str:
    preds = bestmod_bert['model'].predict(X)
    # Be sure to return the only member of the predictions,
    # rather than the singleton list:
    return preds[0]

def create_bakeoff_submission(
        predict_one_func,
        output_filename='cs224u-sentiment-bakeoff-entry.csv'):

    bakeoff_test = sst.bakeoff_test_reader(SST_HOME)
    sst_test = sst.test_reader(SST_HOME)
    bakeoff_test['dataset'] = 'bakeoff'
    sst_test['dataset'] = 'sst3'
    df = pd.concat((bakeoff_test, sst_test))

    df['prediction'] = df['sentence'].apply(predict_one_func)

    df.to_csv(output_filename, index=None)

# This check ensure that the following code only runs on the local environment only.
# The following call will not be run on the autograder environment.
if 'IS_GRADESCOPE_ENV' not in os.environ:
    create_bakeoff_submission(predict_one_rnn)





