from abc import abstractclassmethod
import json
import sys
import os
import logging
import pickle
import gc

import numpy as np
from tqdm import tqdm

from src.data_processing import  SequenceDataset
from src.models import NextWordPredictorModel, init_model
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
    def __init__(self, id_):
        super().__init__(id_)

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
        load_model_from = None
    ):
        self.load_model_from = load_model_from
        # READ CONFIG FILEs
        with open(federated_args, 'r') as f:
            self.federated_args = json.load(f)
            logging.info('federated arguments loaded')
        with open(pipeline_args, 'r') as f:
            self.pipeline_args = json.load(f)
            logging.info('pipeline arguments loaded')
        # LOAD VOCAB
        with open(self.pipeline_args['DATA_PARAMETERS']['vocab_file'], 'rb') as f:
            self.vocabulary = pickle.load(f)
            logging.info('vocabulary loaded')
        self.prepare_directories()

        # INIT GENERAL MODEL
        self.model_parameters = self.pipeline_args['MODEL_PARAMETERS']
        self.model_parameters['device'] = self.pipeline_args['DEVICE']
        self.model_parameters['vocab_size'] = self.vocabulary.get_vocab_size()
        self.general_model = init_model(None, **self.model_parameters)
        self.load_model(path = load_model_from)

        # INIT NODES
        self.num_nodes = self.federated_args['num_nodes']
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

    def init_lambdas(self, num_nodes : int):
        self.lambdas = {}
        if self.federated_args['lambdas'] == 'uniform':
            for node_id in range(1, num_nodes+1):
                self.lambdas[node_id] = self.federated_args['lambda_n']

    def build_nodes(self):
        num_nodes = self.federated_args['num_training_nodes']
        num_bysantine = self.federated_args['num_byzantine']
        self.init_lambdas(num_nodes)
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
                    lambda_ = self.lambdas[node_id],
                    p = self.federated_args['p_0'],
                    datafolder = self.federated_args['nodes_data_folder'],
                    vocabulary = self.vocabulary,
                    min_seq_length = self.federated_args['min_seq_length'],
                    max_seq_length = self.federated_args['max_seq_length'],
                    device = self.federated_args['DEVICE']
                )
        logging.info(f'generated {num_nodes} nodes with {num_bysantine} byzantine')

    def save_embeddings(self):
        embeddings_state_dict = self.general_model.embedding_layer.state_dict()
        torch.save(embeddings_state_dict, self.embeddings_path)

    def load_embeddings(self, model : NextWordPredictorModel = None):
        if model is None:
            model = self.general_model
        with torch.no_grad():
            weights = torch.load(self.embeddings_path)['weight']
            model.embedding_layer.weight.copy_(weights)

    def save_weights(self, node_id : int = 0):
        # by default saves the Pipeline
        if node_id == 0:
            model = self.general_model
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
            model = self.general_model
        rnn_path = os.path.join(self.rnn_folder, 'rnn_general.pth' if node_id == 0 else f"rnn_{node_id}.pth")
        linear_path = os.path.join(self.linear_folder, 'linear_general.pth' if node_id == 0 else f"linear_{node_id}.pth")
        with torch.no_grad():
            rnn_state_dict = torch.load(rnn_path)
            model.rnn.load_state_dict(rnn_state_dict)

            linear_state_dict = torch.load(linear_path)
            model.linear.load_state_dict(linear_state_dict)

    def load_model(self, path = None):
        self.general_model.load_model(path)

    def prepare_models_for_training(self):
        self.general_model.train()
        self.general_model.freeze_embeddings()        
        # Initialize the model for the users
        self.user_model = init_model(None, **self.model_parameters)
        self.user_model.load_model(path = self.load_model_from)
        self.user_model.train()
        del self.user_model.embedding_layer
        gc.collect()
        # This way they share the embeddings layer to save memory
        self.user_model.embedding_layer = self.general_model.embedding_layer 
        self.general_model.lambda_ = self.federated_args['lambda_0']
        self.general_model.p = self.federated_args['p_0']

    def models_difference(self, model2 : NextWordPredictorModel, node : Node):
        """
        Computes the p normed difference between the general model and another
        for the parameters that require gradient excepting biases.
        """
        reg = torch.FloatTensor([0]).to(self.general_model.device)
        if node.lambda_ == 0:
            return reg
        else:
            reg.requires_grad = True
            for (name, w1) in self.general_model.named_parameters():            
                if w1.requires_grad and 'bias' not in name:
                    w2 = model2.state_dict()[name]
                    reg = reg + node.lambda_ * torch.dist(w1, w2, node.p)
            return reg

    def epoch_step(self, epoch):
        # PASS THROUGHT THE NODES
        users_losses = {}
        for node_id in tqdm(range(1, self.federated_args['num_training_nodes'] + 1)):
            # At the first epoch all nodes start from the init model
            node = self.nodes[node_id]
            if epoch == 0:
                self.load_weights(0, self.user_model)
            else:
                self.load_weights(node_id, self.user_model)

                node_dataloader = torch.utils.data.DataLoader(
                    node.data,
                    batch_size = 8,
                    shuffle = True,
                    drop_last = True
                )
                self.user_model.regularizer = self.models_difference
                user_loss = self.user_model.epoch_step(node_dataloader, node, with_tqdm = False)
                users_losses[node_id] = np.mean(user_loss)

                # UPDATE OF THE GENERAL MODEL
                # adds the general model regularization loss and its gradient
                if self.general_model.lambda_ != 0:
                    general_model_reg_loss = self.general_model.regularizer()
                    general_model_reg_loss.backward()
                # performs the general model optimization step
                self.general_model.optimizer.step()
            
            self.save_weights(node_id)        

        return users_losses

    def train(self, num_max_epochs):
        self.prepare_models_for_training()
        node_losses = {}
        general_model_val_losses = {}
        for epoch in range(num_max_epochs+1):
            # Performs the full pass trough the data for every node
            losses = self.epoch_step(epoch)
            if epoch > 0:
                node_losses[epoch] = losses
        return node_losses

    def generate_general(self, start_text : str, num_words : int = 100):
        return self.general_model.generate(start_text=start_text, vocabulary = self.vocabulary, num_words=num_words)

    def generate_node(self, start_text : str, node_id : int, num_words : int = 100):
        self.load_weights(node_id, self.user_model)
        return self.user_model.generate(start_text=start_text, vocabulary = self.vocabulary, num_words=num_words)

if __name__ == '__main__':
    logging.basicConfig(filename = 'federated.log', encoding = 'utf-8', level=logging.DEBUG)
    pass