# MSThesis

## DESCRIPTION

This repo contains the code implementing the Byzantine resilience investigation under federated learning setup for language modeling performed in the context of my Master Thesis.

The general idea is to pretrain a general model and then feed it in a federated setup and see whether the presence of byzantine nodes impacts performances and to what extend.

The projects implements Federated AVG as well as LICCHAVI.

## Usage

### Data Preparation

The data is stored in the */data* folder. To generate the data for training, run

```
python src/data_processing.py <data_name> <id>
``` 

Where <data_name> can either be *tweet* or *wiki*, which are the two data types available. The first used the torchtext wikitext dataset and the second used tweets aggregated during the 2016 USA presidential election.
*id* must be 2 for *tweet* and None for *wiki*

This will gererate the train-val-test splits for models training as well as train-val-test splits sets for every node.

### Model Pretraining

The used tensor library is PyTorch. The model architecture is as follows:

- Embeddings layer
- RNN layer(s)
- Decoder (fully connected)

The embedding dimension, RNN type and dimension and number can be modified on the configuration files.
The configuration files are in the *config_files* folder. There is one per dataset and one per federated as well as pretraining for each.

The model can then be trained calling the Pipeline class

```
from src.models import Pipeline

pipeline = Pipeline(config_file_name.json, load_model_data = True)

pipeline.train(num_max_epochs)
```

### Federated Learning

The model being pretrained, the federated model can be called for federated training.

```
from src.federated_pipeline import Federated_LICCHAVI, Federated_AVG

federated = Federated_LICCHAVI(
    CONFIG_MODEL_TWEETS.json,
    CONFIG_FEDERATED_TWEETS.json
)

federated = Federated_AVG(
    CONFIG_MODEL_TWEETS.json,
    CONFIG_FEDERATED_TWEETS.json
)

federated.train(<num_epochs>)
```
Results are then saved in the */results* or */attack_results* folder depending on the config files.

## OUTLINE

- config_files/
  - CONFIG_FEDERATED_\<data name\>.json
  - CONFIG_MODEL_\<data name\>.json
- data/
- models/
  - \<data name\>/
    - base_model.pth
    - attack_model.pth
- nodes_data/
  - nodes_data_\<data name\>/
    - node data 1
    - node data 2
    - ...
- src/
    - data_processing.py
    - federated_pipeline.py
    - models.py
    - nodes.py
    - utils.py
- vocabs/
    - vocab_\<data name\>.pickle
- weigths/
    - embeddings/
    - linear/
    - rnn/

## Packages Installed

- python = 3.9.5
- pytorch = 1.6.0
- torchtext = 0.7.0
- nltk = 3.6.5
- pandas = 1.3.4
- numpy = 1.21.2
- matplotlib = 3.5.0
- nltk = 3.6.5
- tqdm = 4.62.3
- apex (https://github.com/NVIDIA/apex, for GPU fp16)

## PAPERS

- google next word prediciton:

https://arxiv.org/pdf/1811.03604.pdf

https://ai.googleblog.com/2017/04/federated-learning-collaborative.html

- Federated Learning

https://arxiv.org/pdf/1602.05629.pdf

- Federarge AVG

http://arxiv.org/abs/1602.05629

- gradient/data poisoning attack

https://openreview.net/forum?id=7pZiaojaVGU

- LICCHAVI

https://arxiv.org/abs/2106.02398