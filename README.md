# multilabel-aixpa

🇮🇹 ITALIAN

Classificatore multietichetta per la PA e organizzazioni private. 
Il classificatore è addestrato su dati comunali e separatamente su dati _Family in Italy_ delle organizzazioni che sono coinvolte nel marchio _Family Audit_. 

È stato implementato in maniera da riconoscere automaticamente se nei dati vi sono molteplici etichette per osservazione (quindi un problema di classificazione _multietichetta_) o una singola etichetta per osservazione (problema di classificazione _multiclasse_)

#### Specifiche e contesto
- `tipologia`: product-template
- `ai`: NLP
- `dominio`: PA

Questo strumento può essere usato per ogni task di classificazione di testo. Lo script ```training.py``` si occuperà di ogni numero di etichette sui nuovi dati, solo se i dati sono formattati in un modo specifico. Una volta addestrato e data una nuova osservazione, il modello addestrato può inferire le etichette che appartengono ad essa. I dettagli per preparare i dati sono spiegati nella sezione ```How To```, incluso l'addestramento per il classificatore su nuovi dati.

Al momento della stesura, questo ```README``` è strutturato su strumenti simili come [Faudit Classifier](https://github.com/FluveFV/faudit-classifier), anche se quest'ultimo può solo essere addestrato con un'etichetta per osservazione.


### How To

-   [Processare i testi per addestramento](./src/howto/preprocess.ipynb)
-   [Addestrare il modello classificatore](./src/howto/train.md)
-   [Predire una nuova osservazione](./src/howto/predict.md)
-   [Parametri ottimali per dati comunali](./src/m_parameters.yaml)
-   [Parametri ottimali per dati organizzativi](./src/o_parameters.yaml)

#### Dati di addestramento
La versione integrale dei dati è disponibile su richiesta. I dati delle organizzazioni private non sono di dominio pubblico, mentre per i dati dei Comuni lo sono a fini di trasparenza amministrativa. 

### Comportamento conosciuto nella predizione
Modelli addestrati differentemente hanno comportamenti dipendenti dal tipo di dati sui quali sono addestrati. È provato che:

- se un modello BERT-base italian xxl uncased è stato addestrato su un dataset multiclasse con una singola etichetta per osservazione, le sue predizioni sono più confidenti del solito, per cui è necessario impostare una soglia di attivazione maggiore di $0.5$
- se un modello BERT-base italian xxl uncased è stato addestrato su dati multietichetta, le sue predizioni sono meno confidenti del solito, rendendo necessario impostare una soglia di attivazione minore di $0.5$. 

##### Raccomandazione: 
Per una **precisione del 75%**, si raccomanda che:
- il classificatore multiclasse singola etichetta può predire osservazioni usando una soglia nella funzione (e.g. sigmoidea) di $0.97$
- il classificatore multietichetta può predire osservazioni usando una soglia di $0.44$

Altrimenti, è possibile scalare le osservazioni usando il ridimensionamento tramite temperatura e usare una tipica soglia come $0.5$. 

---
🇺🇸-🏴󠁧󠁢󠁥󠁮󠁧󠁿 ENGLISH

Document classifier for Public Administration and private organizations. 
The classifier is trained on Municipalities' data for the _Family in Italy_ mark and separately on organizations' data that have applied for _Family Audit_ mark. 

It has been deployed to automatically detect if the data has multiple labels per observation (_multi-label_ classification problem) or only a single label per observation (_multi-class_).


#### Specifics and context
-   `kind`: product-template
-   `ai`: NLP
-   `domain`: PA, enterprise

This tool can be used for any text classification task. The ```training.py``` script will handle any number of labels on the new data, given that data is formatted in a standard way. Once trained and given a new observation, the trained model can infer the labels it belongs to. To preprocess data more details are explained in the ```How To``` section, included fine tuning the classifier on new data.

At the time of writing, this ```README``` is structured on similar tools such as [Faudit Classifier](https://github.com/FluveFV/faudit-classifier), even though the latter can only be trained on one label per observation.

### How To

-   [Preprocess corpora for training](./src/howto/preprocess.ipynb)
-   [Train the classifier model](./src/howto/train.md)
-   [Predict a new observation](./src/howto/predict.md)
-   [Optimal parameters for Municipalities data](./src/m_parameters.yaml)
-   [Optimal parameters for Organizations data](./src/o_parameters.yaml)

#### Training data
All of the data is available on request. However, organizations' data is private, while Municipalities data is of public domain. 


### Known behavior for prediction

Trained models have different behavior depending on the type of data they were trained on. It is known that 

- if a BERT-base italian xxl uncased model has been trained on a singlelabel multiclass dataset, its predictions are more confident than usual, therefore making it necessary to set an activation threshold higher than $0.5$
- if a BERT-base italian xxl uncased model has been trained on multilabel data, its predictions are less confident than usual, making it necessary to set an activation threshold lower than $0.5$

##### Recommendation: 
For a **Precision of 75%**, we advise that:
- the single label multiclass classifier predicts observations using an activation function (e.g. sigmoid) with a threshold of $0.97$
- the multilabel classifier predicts observations using an activation function with a threshold of $0.44$

Otherwise, it is possible to just use temperature scaling for the predictions with a typical threshold, such as $0.5$

