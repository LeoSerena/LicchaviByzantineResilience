# MSThesis

## DESCRIPTION

Byzantine resilience investigation under federated learning setup for language modeling.
The first step is to pretrain a general model and then feed it in a federated setup and see whether the presence of byzantine nodes impacts performances and to what extend.

The data is stored in the */data* folder. To generate the data for training, run

```
python src/data_processing.py *data_name* *id*
``` 
Where *data_name* can either be *tweet*, *wiki* and *id* must be 2 for *tweet* and None for *wiki*. To use wikitext-2 or wikitext103, change the config files accordingly.

This will gererate the train-val-test splits for models training as well as train-val-test splits sets for every node.

The used tensor library is PyTorch. The model architecture is as follows:

- Embeddings layer
- RNN layer(s)
- Decoder (fully connected)

The embedding dimension, RNN type and dimension and number can be modified on the configuration files.

The model can then be trained calling the Pipeline class

```
from src.models import Pipeline

pipeline = Pipeline(*config_file_name.json*, load_model_data = True)

pipeline.train(*num_max_epochs*)
```

The model being pretrained, the federated model can be called for federated training.

```
from src.federated_pipeline import Federated_LICCHAVI, Federated_SGD, Federated_AVG

federated = Federated_LICCHAVI(
    "CONFIG_MODEL_TWEETS.json",
    "CONFIG_FEDERATED_TWEETS.json"
)
# or
federated = Federated_SGD(
    "CONFIG_MODEL_TWEETS.json",
    "CONFIG_FEDERATED_TWEETS.json"
)
# or 
federated = Federated_AVG(
    "CONFIG_MODEL_TWEETS.json",
    "CONFIG_FEDERATED_TWEETS.json"
)

federated.train(*num_epochs*)
```
retults are then saved in the */results* folder.

## OUTLINE

- config_files: files to change the parameters
- data: where to store the data
  - WikiText-2
  - WikiText103
  - tweets
- logs: logging
- models: where the models are stored
- nodes_data: data for every node
- results: results storage
- src: code
  - data_processing.py: data preprocessing
  - federated_pipeline.py: Federated class implementation
  - models.py: NextWordPredictor and Pipeline implementation for pretraining
  - utils.py: general methods
- vocabs: vocabularies specific to datasets
- weights: temporary weights for nodes for every layer

## PAPERS

- Federated Learning
https://arxiv.org/pdf/1602.05629.pdf

- google next word prediciton:
https://arxiv.org/pdf/1811.03604.pdf

- FedAtt
https://arxiv.org/abs/1812.07108