# Predict

The product is ready for downstream tasks. A simple example for using the predictive capacity is displayed in [predictor.py](https://github.com/FluveFV/multilabel-aixpa/blob/main/src/howto/predictor.py) .

Two arguments can be specified:

The main elements of said scripts are divided in:

-   loading the model weights and architecture for prediction from `train.py` and the model + tokenizer config in directory `multi_label_classification` or `single_label_classification`, depending on what your data looks like.
-   predicting a random example's observation

Important notice: to connect to a different model for appropriate data, you need to modify the path connecting the model and tokenizer configuration. For example,

``` python
t_file_path = "multi_label_classification/tokenizer_config"  #leads to organizations trained tokenizer
t_file_path = "single_label_classification/tokenizer_config"  #leads to municipalities trained model
```

After loading the pretrained model, `predictor.py` given a text will output its predicted label, map it back to the original distribution, then display the additional information, such as the probabilities of all the classes for predictions.

Any model trained will predict a number of observations depending on the threshold of the activation function:

``` python
threshold = float(input("Enter problem activation threshold (close to 0.1 for organization data, close to 0.7 for municipalities data):"))
```

The reason for this behavior is unknown, but:

-   The model trained on municipalities data, the single-label classifier, usually predicts more classes above a threshold of $0.7$.
-   The model trained on organizations, the multi-label classifier, data usually predicts more classes only if the threshold is much lower, such as $0.1$.

To avoid overconfidence, one may modify the script and set a scaling parameter called temperature, and modify the logits **before** the activation function:

``` python
temperature = float(input("Enter temperature (higher values soften predictions, lower values make them sharper):"))

with torch.no_grad():
    logits = model(**inputs).logits.numpy() / temperature
```

For values of `temperature` greater than $1$, the predictions will be softened, while smaller than $1$ they will be sharpened. The user can test more texts. There is no lower or upper limit to the size of the input, but the BERT will only use the first 512 tokenized elements of the text.
