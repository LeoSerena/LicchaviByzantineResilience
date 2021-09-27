from abc import abstractclassmethod
import json
import sys
import os
import logging
import pickle

import pandas as pd
from torch.nn.modules import rnn

from src.data_processing import  SequenceDataset
from src.models import Pipeline
from src.utils import make_dir_if_not_exists

import torch

class Node():
    def __init__(
        self,
        id_ : int
    ):
        if self.__class__ ==  Node:
            raise NotImplementedError("""This is an abstract class""")
        self.id_ = id_

class UserNode(Node):
    def __init__(
        self,
        id_,
        datafolder,
        vocabulary,
        min_seq_length,
        max_seq_length,
        device
    ):
        super(UserNode, self).__init__(id_)
        for file in os.listdir(datafolder):
            if f'node_{id_}' in file:
                self.file = file

        with open(os.path.join(datafolder, self.file), 'rb') as f:
            data = pickle.load(f)
            
        self.data = SequenceDataset(
            vocabulary = vocabulary,
            text = data,
            min_seq_length = min_seq_length,
            max_seq_length = max_seq_length,
            device = device
        )

class ByzantineNode(Node):
    def __init__():
        pass

class NullByzantineNode(ByzantineNode):
    def __init__():
        pass

class RandomByzantineNode(ByzantineNode):
    def __init__():
        pass

class StrategicalByzantineNode(ByzantineNode):
    def __init__():
        pass


class Federated():
    def __init__(
        self,
        pipeline_args : str,
        federated_args : str,
        load_model : bool = False,
        load_model_data : bool = False
    ):
        self.pipeline = Pipeline(pipeline_args, load_model_data = load_model_data)
        with open(federated_args, 'r') as f:
            self.federated_args = json.load(f)

        self.num_nodes = self.federated_args['num_nodes']
        self.prepare_directories()
        if load_model:
            self.load_embeddings()
            self.load_weights(0)

        self.build_nodes()

    def prepare_directories(self):
        self.model_dir = self.federated_args['model_dir']
        make_dir_if_not_exists(self.model_dir)
        self.embeddings_path = os.path.join(self.federated_args['embeddings_folder'], 'embeddings.pth')
        make_dir_if_not_exists(self.federated_args['embeddings_folder'])
        self.rnn_folder = self.federated_args['rnn_folder']
        make_dir_if_not_exists(self.rnn_folder)
        self.linear_folder = self.federated_args['linear_folder']
        make_dir_if_not_exists(self.linear_folder)

    def build_nodes(self):
        num_nodes = self.federated_args['num_training_nodes']
        num_bysantine = self.federated_args['num_byzantine']
        if num_nodes < num_bysantine:
            logging.error("The number of byzantine nodes can't be superior to the total number of nodes")
            sys.exit(1)
        self.nodes = {}
        for node_id in range(1, num_nodes+1):
            if node_id < num_bysantine:
                self.nodes[node_id] = ByzantineNode(node_id)
            else:
                self.nodes[node_id] = UserNode(
                    id_ = node_id,
                    datafolder = self.federated_args['nodes_data_folder'],
                    vocabulary = self.pipeline.vocabulary,
                    min_seq_length = self.federated_args['min_seq_length'],
                    max_seq_length = self.federated_args['max_seq_length'],
                    device = self.federated_args['DEVICE']
                )
                
        
    def pretrain_model(self):
        self.pipeline.train_model()
        self.pipeline.model.freeze_embeddings()
        self.save_embeddings()
        self.save_weights(0)

    def save_embeddings(self):
        embeddings_state_dict = self.pipeline.model.embedding_layer.state_dict()
        torch.save(embeddings_state_dict, self.embeddings_path)

    def load_embeddings(self):
        with torch.no_grad():
            weights = torch.load(self.embeddings_path)['weight']
            self.pipeline.model.embedding_layer.weight.copy_(weights)

    def save_weights(self, node_id : int):
        rnn_state_dict = self.pipeline.model.rnn.state_dict()
        rnn_path = os.path.join(self.rnn_folder, 'rnn_general.pth' if node_id == 0 else f"rnn_{node_id}.pth")
        torch.save(rnn_state_dict, rnn_path)

        linear_state_dict = self.pipeline.model.linear.state_dict()
        linear_path = os.path.join(self.linear_folder, 'linear_general.pth' if node_id == 0 else f"linear_{node_id}.pth")
        torch.save(linear_state_dict, linear_path)

    def load_weights(self, node_id : int):
        rnn_path = os.path.join(self.rnn_folder, 'rnn_general.pth' if node_id == 0 else f"rnn_{node_id}.pth")
        linear_path = os.path.join(self.linear_folder, 'linear_general.pth' if node_id == 0 else f"linear_{node_id}.pth")
        with torch.no_grad():
            rnn_state_dict = torch.load(rnn_path)
            self.pipeline.model.rnn.load_state_dict(rnn_state_dict)

            linear_state_dict = torch.load(linear_path)
            self.pipeline.model.linear.load_state_dict(linear_state_dict)

    def load_model(self, path = None):
        self.pipeline.load_model(path)

    def licchavi_loss_general(self):
        pass

    def licchavi_loss_node(self, model):
        pass

    def epoch_step(self, epoch):
        for node_id in range(1, self.num_nodes+1):
            # At the first epoch all nodes start from the init model
            if epoch == 0:
                self.load_weights(0)
            else:
                self.load_weights(node_id)




if __name__ == '__main__':
    logging.basicConfig(filename = 'federated.log', encoding = 'utf-8', level=logging.DEBUG)
    pass