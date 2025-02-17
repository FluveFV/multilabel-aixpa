# Training FamilyAudit-Classifier

Italian 🇮🇹 version below.

In case of availability of new data it is advised to first check its compatibility through the specifications of [data preprocessing](https://github.com/FluveFV/multilabel-aixpa/blob/main/src/howto/preprocess.ipynb).

Simply running the default training mode can be done from command line using Docker. For example

``` bash
docker run --gpus '"device=*"' --rm -ti --shm-size=32gb \
    -v $PWD:/src \
    --workdir /src \
    dockerimagename \
    python train.py
```

In the command above, the `\*` was replaced with the computing unit's ID and `dockerimagename` was also replaced with the docker image made for the purpose of training.

Otherwise, the content of the script `train.py` can be ran anywhere with Python. Check for the requirements in your machine in [requirements.txt](https://github.com/FluveFV/multilabel-aixpa/blob/main/requirements.txt)

The training was executed on one GPU that exists in a cluster. Only one was specified, as the model does not require exceptional computational power.

## Model hyperparameters

Moving on, all parameters for training can be modified from the training script `train.py`

For example, to train and augment the patience for a later automated stopping of the model, one can modify:

``` python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)
```

Or similarly, one can modify the epochs in the training arguments within the script, or other arguments.

``` python
training_args = TrainingArguments(
    output_dir=model_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=25,
    #max_steps=1, #D
    weight_decay=0.005,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    #report_to=None,  #Wandb and other are excluded with this setting
    fp16=True,
    optim="adamw_torch_fused",
    # parallel computing parameters (GPU)
    #dataloader_num_workers = 4,
    #ddp_find_unused_parameters=False,
    #ddp_backend='nccl',
    resume_from_checkpoint = False
)
```

Since the nature of the taxonomy of the companies data includes many labels, during training the model can tend to focus on more frequent classes. To avoid this, the training is carried on with the objective to focus on all classes, and the evaluation is weighted on the inverse frequency of the classes with weighted Cross Entropy Loss.

On the other hand, almost absent classes shouldn't hinder the overall evaluation on the test test, as it is expected for them to not appear as much in future input data. For this reason, while training is done to try and learn for all classes, the final evaluation on test set is done considering a weighted per-class average of correctly predicted observations - the micro F1 weighted measure. For short, the training Loss forces the model to be more influenced by smaller classes, while the performance is more influenced by the bigger classes.

In the custom data loader for multiple labels a specific feature turns a structured datalake into a *datasets* object with one column containing all the labels in a hot encoded format.  In this format, each observation is associated with a vector of length $m$ where

-   the classes (unique labels) are ordered from $0$ to $m$
-   $m$ is the number of classes found in the data
-   each element of the vector is either a 0 or 1
-   each element is at a position i from 0 to m
-   an element is 1 if there is a correspondence between the element at position $i$ and the presence of the class, otherwise 0

For example, the following observation "Dog" is labelled "Mammal" and "Pet" and is not "Reptile". 

The classes are ordered
$$l_{labels} = ['Mammal', 'Pet', 'Reptile'] \rightarrow l_{classes} = [0, 1, 2]$$

A vector for that observation is created:
$$v = [0, 0, 0] $$

The vector gets updated in correspondence to the presence of classes.
$$ v \leftarrow [1, 1, 0] $$

The hot-encoded label vector requires a special loss. In the model a Binary Cross Entropy loss is used. Binary does not refer to two classes.

$\ell_{BCE}=-\frac{1}{N}\sum^N_{i=1}[y_i ~ log ~ \sigma(x_i) + (1-y_i)log(1-\sigma(x_i)]$

Where $x_i$ is one raw logit output of the model, $y_i$ is the true label (one hot encoded). The raw outputs get transformed to probabilities using sigmoid activation $\sigma$.

The metric of evaluation for the performance is micro F1 score, weighted on the frequency of classes. 

$$
\text{Micro F1} = \frac{2 \times \text{Micro Precision} \times \text{Micro Recall}}{\text{Micro Precision} + \text{Micro Recall}}
$$

where Micro Precision and Micro Recall are defined as:

$$\text{Micro Precision} = \frac{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n}{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n + \text{FP}_1 + \text{FP}_2 + \cdots + \text{FP}_n}$$

$$\text{Micro Recall} = \frac{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n}{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n + \text{FN}_1 + \text{FN}_2 + \cdots + \text{FN}_n}$$ 

-   $\text{TP}_i$: True Positive for class $i$
-   $\text{FP}_i$: False Positive for class $i$
-   $\text{FN}_i$: False Negative for class $i$

Since the output of the test set prediction is saved from the `train.py` in a csv file, the ultimate tests can be chosen ad hoc. For multilabel problems, I use sci-kit learn's [special metric module](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) specified for multilabel problems.

At each epoch of the training, the performance is evaluated on the validation set. Each observation is a list of metrics. It is also saved as a csv file.

## End of training

The F1 score on test data is printed out in the terminal at the end of the process.

If the data is compatible and the choice of parameters does not raise any errors, the training will come to an end, and train.py will automatically save the model configuration (model architecture, weights, etc.).

# Addestrare il classificatore per FamilyAudit

In caso di nuovi dati si consiglia di leggere le specifiche della [preparazione dati](https://github.com/FluveFV/multilabel-aixpa/blob/main/src/howto/preprocess.ipynb).

``` bash
docker run --gpus '"device=*"' --rm -ti --shm-size=32gb \
    -v $PWD:/src \
    --workdir /src \
    dockerimagename \
    python train.py
```

Nel comando qui sopra, `\*` è stato rimpiazzato con il nome dell'unità di computazione, e `dockerimagename` con il nome dell'immagine Docker creata per addestrare il modello.

Altrimenti, è possibile eseguiree `train.py` ovunque con Python. È necessario avere i requisiti installati nella macchina che si possono trovare su [requirements.txt](https://github.com/FluveFV/multilabel-aixpa/blob/main/requirements.txt)

### Iperparametri del modello

Passando oltre, gli iperparametri possono essere modificati all'interno di `train.py`.

Esempio di modifica: cambiare la pazienza per un termine dell'addestramento successivo

``` python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]  #Qui
)
```

Alternativamente, è possibile modificare le ere di addestramento negli argomenti all'interno di `training_args`, o gli altri parametri se necessario.

``` python
training_args = TrainingArguments(
    output_dir=model_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=25,
    #max_steps=1, #D
    weight_decay=0.005,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    #report_to=None,  #Wandb and other are excluded with this setting
    fp16=True,
    optim="adamw_torch_fused",
    # parallel computing parameters (GPU)
    #dataloader_num_workers = 4,
    #ddp_find_unused_parameters=False,
    #ddp_backend='nccl',
    resume_from_checkpoint = False
)
```

A causa della natura della tassonomia che include molte voci l'addestramento tende a focalizzarsi di più sulle classi più frequenti. Per evitare che questo succeda, l'addestramento è eseguito con l'obiettivo di apprendere tutte le classi, e la valutazione dell'apprendimento è quindi soppesata dall'inverso delle frequenze delle classi con la weighted Cross Entropy Loss.

D'altro canto, le classi che sono quasi inesistenti nel campione non dovrebbero peggiorare vistosamente la performance, poiché ci si aspetta che esse non compariranno altrettanto in futuro. Per questo motivo, sebbene l'addestramento sia fatto con l'obiettivo di apprendere tutte le classi, la valutazione finale della performance sul test set è fatta considerando soltanto una media soppesata dalla frequenza delle classi per le osservazioni correttamente predette: la weighted micro F1.

In breve, la Loss dovrebbe forzare l'apprendimento ad essere più influenzato dalle classi più infrequenti, mentre la valutazione della performance generale è più influenzata dalle classi più frequenti.

Nel dataloader fatto ad hoc per molteplici etichette, una specifica funzione rende il *lago dati strutturato* in un oggetto *datasets*, con una colonna che contiene tutte le etichette in un formato binario. In questo formato ogni osservazione è associata con un vettore di lunghezza $m$ dove

-   le classi (etichette uniche) sono ordinate da $0$ a $m$
-   $m$ è il numero totale di classi trovate nei dati
-   ogni elemento del vettore si trova alla posizione i da $0$ a $m$
-   ogni elemento è rappresentato da $1$ o $0$.
-   un elemento è $1$ se c'è corrispondenza tra l'elemento alla posizione $i$ e la presenza di una classe nella posizione $i$, altrimenti è $0$.

Ad esempio, la seguente osservazione "Fido" appartiene alla classe "Mammifero" e "Animale da compagnia" e non è "Rettile". 

Viene definito l'ordine delle classi: 
$$l_{etichette} = ['Mammifero', 'Animale da compagnia', 'Rettile'] \rightarrow l_{classi} = [0, 1, 2]$$

Viene creato un vettore per l'osservazione:
$$v = [0, 0, 0] $$

Il vettore viene aggiornato controllando quali classi sono davvero presenti per l'osservazione:
$$ v \leftarrow [1, 1, 0] $$

Su una matrice di outputs è necessaria una loss speciale. Per questo problema è stata usata la Binary Cross Entropy. Binary non si riferisce a sole due classi. 

$$\ell_{BCE}=-\frac{1}{N}\sum^N_{i=1}[y_i ~ log ~ \sigma(x_i) + (1-y_i)log(1-\sigma(x_i)]$$

Dove $x_i$ è un logit output del modello; $y_i$ sono le vere etichette (dummy). Il logit output è trasformato in probabilità usando l'attivazione sigmoidale $\sigma$. 

La metrica di valutazione della performance è la micro F1 score, soppesata dalla frequenza delle classi. È definita come

$$\text{Micro F1} = \frac{2 \times \text{Micro Precision} \times \text{Micro Recall}}{\text{Micro Precision} + \text{Micro Recall}}$$

dove la MicroPrecision e la MicroRecall sono

$$\text{Micro Precision} = \frac{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n}{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n + \text{FP}_1 + \text{FP}_2 + \cdots + \text{FP}_n}$$

$$\text{Micro Recall} = \frac{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n}{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n + \text{FN}_1 + \text{FN}_2 + \cdots + \text{FN}_n}$$ 

-   $\text{TP}_i$: True Positive per la classe $i$
-   $\text{FP}_i$: False Positive per la classe $i$
-   $\text{FN}_i$: False Negative per la classe $i$


L'output delle predizioni sul test set è salvato automaticamente in un file .csv. Ogni osservazione di quel file è una lista di metriche. 

# Termine addestramento

La misura F1 è stampata nel terminale al termine del processo.

Se i dati sono compatibili e la scelta dei parametri non porta a nessun errore, l'addestramento sarà portato a termine e `train_py` salva automaticamente la configurazione del modello per altre attività downstream (architettura del modello, pesi, etc.).

I risultati dell'addestramento possono essere analizzati (magari sotto altre metriche) nel file di output che contiene le predizioni del modello sul test set e la ground truth, assieme agli indici delle osservazioni del test set che rappresentano la loro posizione nel dataset di input.
