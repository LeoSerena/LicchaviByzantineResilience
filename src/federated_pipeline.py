from abc import abstractclassmethod
import json
import sys
import os
import logging
import pickle
import gc

import pandas as pd
from torch.nn.modules import rnn

from src.data_processing import  SequenceDataset
from src.models import NextWordPredictorModel, Pipeline, init_model
from src.utils import make_dir_if_not_exists

import torch

class Node():
    def __init__(
        self,
        id_ : int,
        lambda_ : float,
        p : int
    ):
        if self.__class__ ==  Node:
            raise NotImplementedError("""This is an abstract class""")
        self.id_ = id_
        self.lambda_ = lambda_
        self.p = p

class UserNode(Node):
    def __init__(
        self,
        id_,
        lambda_,
        p,
        datafolder,
        vocabulary,
        min_seq_length,
        max_seq_length,
        device
    ):
        super(UserNode, self).__init__(id_, lambda_, p)
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
    def __init__(self):
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
                self.nodes[node_id] = ByzantineNode(node_id, lambda_ = 1e-3, p = 1)
            else:
                self.nodes[node_id] = UserNode(
                    id_ = node_id,
                    lambda_ = 1e-3,
                    p = 1,
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

    def load_embeddings(self, model : NextWordPredictorModel = None):
        if model is None:
            model = self.pipeline.model
        with torch.no_grad():
            weights = torch.load(self.embeddings_path)['weight']
            model.embedding_layer.weight.copy_(weights)

    def save_weights(self, node_id : int = 0):
        # by default saves the Pipeline
        if node_id == 0:
            model = self.pipeline.model
        else:
            model = self.user_model
        rnn_state_dict = model.rnn.state_dict()
        rnn_path = os.path.join(self.rnn_folder, 'rnn_general.pth' if node_id == 0 else f"rnn_{node_id}.pth")
        torch.save(rnn_state_dict, rnn_path)

        linear_state_dict = model.linear.state_dict()
        linear_path = os.path.join(self.linear_folder, 'linear_general.pth' if node_id == 0 else f"linear_{node_id}.pth")
        torch.save(linear_state_dict, linear_path)

    def load_weights(self, node_id : int = 0, model : NextWordPredictorModel = None):
        if model is None:
            model = self.pipeline.model
        rnn_path = os.path.join(self.rnn_folder, 'rnn_general.pth' if node_id == 0 else f"rnn_{node_id}.pth")
        linear_path = os.path.join(self.linear_folder, 'linear_general.pth' if node_id == 0 else f"linear_{node_id}.pth")
        with torch.no_grad():
            rnn_state_dict = torch.load(rnn_path)
            model.rnn.load_state_dict(rnn_state_dict)

            linear_state_dict = torch.load(linear_path)
            model.linear.load_state_dict(linear_state_dict)

    def load_model(self, path = None):
        self.pipeline.load_model(path)


    def models_difference(self, model2 : NextWordPredictorModel, node : Node):
        reg = torch.FloatTensor([0]).to(self.pipeline.model.device)
        if node.lambda_ == 0:
            return reg
        else:
            reg.requires_grad = True
            for (name, w1) in self.pipeline.model.named_parameters():            
                if w1.requires_grad and 'bias' not in name:
                    w2 = model2.state_dict()[name]
                    reg = reg + node.lambda_ * torch.dist(w1, w2, node.p)
            return reg

    def epoch_step(self, epoch):
        for node_id in range(1, self.federated_args['num_training_nodes'] + 1):
            # At the first epoch all nodes start from the init model
            node = self.nodes[node_id]
            if epoch == 0:
                self.load_weights(0, self.user_model)
            else:
                self.load_weights(node_id, self.user_model)

            node_dataloader = torch.utils.data.DataLoader(
                node.data,
                batch_size = 16,
                shuffle = True,
                drop_last = True
            )
            self.user_model.regularizer = self.models_difference
            self.user_model.epoch_step(node_dataloader, node)

            self.save_weights(node_id)


    def train(self, epochs):
        self.pipeline.model.train()
        self.pipeline.model.freeze_embeddings()
        # optimizer for the general model in the Pipeline class
        self.general_optimizer = self.pipeline.model.optimizer
        
        # Initialize the model for the users
        model_parameters = self.pipeline.parameters['MODEL_PARAMETERS']
        model_parameters['device'] = self.pipeline.parameters['DEVICE']
        model_parameters['vocab_size'] = self.pipeline.vocabulary.get_vocab_size()
        self.user_model = init_model(vocabulary = None, **model_parameters)
        self.user_model.load_model()
        self.user_model.train()
        self.user_model.fp16 = True
        del self.user_model.embedding_layer
        gc.collect()
        self.user_model.embedding_layer = self.pipeline.model.embedding_layer # This way they share the embeddings layer to save memory
        
        # general model regularizer
        self.pipeline.lambda_ = self.federated_args['lambda_0']
        self.pipeline.p = 1

        for epoch in range(epochs+1):
            self.epoch_step(epoch)
            self.general_optimizer.step()

if __name__ == '__main__':
    logging.basicConfig(filename = 'federated.log', encoding = 'utf-8', level=logging.DEBUG)
    pass