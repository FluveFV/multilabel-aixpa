# multilabel-aixpa
🇺🇸-🏴󠁧󠁢󠁥󠁮󠁧󠁿

Document classifier. It has been deployed to automatically detect if the data has multiple labels per observation (_multi-label_ classification problem) or only a single label per observation (_multi-class_).


#### Specifics and context
-   `kind`: product-template
-   `ai`: NLP
-   `domain`: PA

This classifier has been developed as an updated version that can handle multilabel data, specifically trained on data from companies participating in Family Audit project. 

This tool can be used for any classification task. The ```training.py``` script will handle any number of labels on the new data, given that data is formatted in a standard way. Once trained and given a new observation, the trained model can infer the labels it belongs to. To preprocess data more details are explained in the ```How To``` section, included fine tuning the classifier on new data.

At the time of writing, this ```README``` is structured on similar tools such as [Faudit Classifier](https://github.com/FluveFV/faudit-classifier), even though the latter can only be trained on one label per observation.

### How To

-   [Preprocess corpora for training](./src/howto/preprocess.ipynb)
-   [Train the classifier model](./src/howto/train.md)

---
🇮🇹

Classificatore multietichetta. Multietichetta indica che il modello può essere addestrato su dati con molteplici etichette per osservazione, come "cane" può essere "mammifero" e "animale di compagnia". 

#### Specifiche e contesto
- `tipologia`: product-template
- `ai`: NLP
- `dominio`: PA

Questo classificatore è stato sviluppato come una versione aggiornata capace di gestire dati multietichetta, specificamente addestrato su dati di organizzazioni che partecipano al progetto Family Audit.

Questo strumento può essere usato per ogni task di classificazione. Lo script ```training.py``` si occuperà di ogni numero di etichette sui nuovi dati, solo se i dati sono formattati in un modo specifico. Una volta addestrato e data una nuova osservazione, il modello addestrato può inferire le etichette che appartengono ad essa. I dettagli per preparare i dati sono spiegati nella sezione ```How To```, incluso l'addestramento per il classificatore su nuovi dati.

Al momento della stesura, questo ```README``` è strutturato su strumenti simili come [Faudit Classifier](https://github.com/FluveFV/faudit-classifier), anche se quest'ultimo può solo essere addestrato con un'etichetta per osservazione.
