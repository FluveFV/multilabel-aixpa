# Predict

The product is ready for downstream tasks. A simple example for using the predictive capacity is displayed in `predict.py`.

The script should be ran from terminal in the same location as train.py. Two arguments can be specified:

-   `-k` (int, the number of labels to predict)
-   `-map` (bool, whether to map the main label to macrocategoria and campo or not).

The main elements of said scripts are divided in:

-   loading the model weights and architecture for prediction from `bert_wrapper.py` and the model config in directory `tuned_model`

``` python
model = BertForSentenceClassification.from_pretrained(
    model_path,
    config=config,
    model_name="dbmdz/bert-base-italian-xxl-cased",
    num_labels=config.num_labels
)
```

-   loading the label mapping from dense to sparse

<!-- -->

-   predicting mechanism

``` python
def one_label(inputs):
        ...
        return result
def multiple_labels(inputs):
        ...
        return result
```

-   simple CLI interface

``` python
testing_model = correct_input()

while testing_model == 'y':
    ...
```

In case one would want to test the predicting mechanism, there is a whole frozen section below the CLI interface that can be ran on 100 examples and save the prediction. It needs to be adapted manually if one wants multiple labels or just one.

All other functions in `predict.py` help the interface being smoother.

### Loading model weights, architecture

It can be done by using hugginface's AutoTokenizer, as the tokenizer was not further trained, but the custom BERT architecture has to be loaded from `bert_wrapper.py`

### Label mapping, additional information regarding labels

Since the BERT architecture in use was trained with specific attributes, the predicted labels were mapped in two ways:

-   $f(y) = y-1$ so that they start from 0 since labels originally start from 1
-   reindexing $f(y)$ so that $y$ is contiguous (using reindexing)

Hence, the model learns and evaluates labels specifically moved towards 0 by 1 and mapped to a dense array. For this reason, any downstream task requires to store the mapping back to the original distribution using the inverted indexing. The operation of summing of 1 each label can be done with no further explanation, while `predict.py` uses the label indexing used in training and saved in `tuned_model/label_mapping.json` to revert the predictions back to original labelling.

Additional information includes:

-   macroambito
-   campo

and the relative descriptions. Those are domain-specific information that is stored in [`correspondences.csv`](https://github.com/FluveFV/faudit-classifier/blob/main/src/correspondence.csv)

## Predicting mechanism and CLI usage

After loading the pretrained model, `predict.py` given a text will output its predicted label, map it back to the original distribution, then display the additional information. The user can test more texts. There is no lower or upper limit to the size of the input, but the BERT will only use the first 512 tokenized elements of the text. From the terminal, the user can insert the text they want to test.
