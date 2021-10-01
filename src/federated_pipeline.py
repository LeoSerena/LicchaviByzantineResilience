import json
import sys
import os
import logging
import pickle
import gc

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        self.losses = {
            'total_loss' : [],
            'loss' : [],
            'reg_loss' : []
        }


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
    def __init__(
        self,
        id_,
        lambda_,
        p
    ):
        super(Node, self).__init__(id_, lambda_, p)
        if self.__class__ ==  ByzantineNode:
            raise NotImplementedError("""This is an abstract class""")

class NullByzantineNode(ByzantineNode):
    def __init__(
        self,
        id_,
        lambda_,
        p,
        vocabulary,
        min_seq_length,
        max_seq_length,
        device
    ):
        super(ByzantineNode, self).__init__(id_,lambda_,p)
        padd_token = 'trump'
        length = 1000
        data = [' '.join([padd_token] * max_seq_length)] * length
        self.data = SequenceDataset(
            vocabulary = vocabulary,
            text = data,
            max_seq_length = max_seq_length,
            min_seq_length = min_seq_length,
            device = device
        )


class RandomByzantineNode(ByzantineNode):
    def __init__(
            self,
            id_,
            lambda_,
            p,
            vocabulary,
            min_seq_length,
            max_seq_length,
            device 
        ):
        super(ByzantineNode, self).__init__(id_, lambda_, p)
        length = 1000
        random_tokens = np.random.randint(low = 1, high = vocabulary.get_vocab_size(), size = int(length * max_seq_length))
        random_data = np.array([vocabulary.idx_to_word[x] for x in random_tokens]).reshape((length, max_seq_length))
        self.data = SequenceDataset(
            vocabulary = vocabulary,
            text = random_data,
            max_seq_length = max_seq_length,
            min_seq_length = min_seq_length,
            device = device
        )

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
        self.load_val_test_set()

        # INIT GENERAL MODEL
        self.model_parameters = self.pipeline_args['MODEL_PARAMETERS']
        self.model_parameters['device'] = self.pipeline_args['DEVICE']
        self.model_parameters['vocab_size'] = self.vocabulary.get_vocab_size()
        self.general_model = init_model(None, **self.model_parameters)
        self.load_model(path = load_model_from)
        self.save_embeddings()
        self.save_weights()

        # INIT NODES
        self.num_nodes = self.federated_args['num_nodes']
        self.build_nodes()

    def prepare_directories(self):
        self.weights_dir = self.federated_args['weights_dir']
        make_dir_if_not_exists(self.weights_dir)
        self.embeddings_path = os.path.join(self.weights_dir, self.federated_args['embeddings_folder'], 'embeddings.pth')
        make_dir_if_not_exists(os.path.join(self.weights_dir, self.federated_args['embeddings_folder']))
        self.rnn_folder = os.path.join(self.weights_dir, self.federated_args['rnn_folder'])
        make_dir_if_not_exists(self.rnn_folder)
        self.linear_folder = os.path.join(self.weights_dir, self.federated_args['linear_folder'])
        make_dir_if_not_exists(self.linear_folder)
        self.optim_folder = os.path.join(self.weights_dir, self.federated_args['optim_folder'])
        make_dir_if_not_exists(self.optim_folder)

    def load_val_test_set(self):
        test_set_file = os.path.join('data', 'test_2.pickle')
        with open(test_set_file, 'rb') as f:
            test_set = pickle.load(f)
        sep = int(len(test_set) / 100)
        val_set = test_set[:sep]
        test_set = test_set[-sep:]
        params = self.pipeline_args['DATA_PARAMETERS']
        self.val_dataset = SequenceDataset(
            vocabulary = self.vocabulary,
            text = val_set,
            min_seq_length = params['min_seq_length'],
            max_seq_length = params['max_seq_length'],
            device = self.pipeline_args['DEVICE'],
        )
        self.test_dataset = SequenceDataset(
            vocabulary = self.vocabulary,
            text = test_set,
            min_seq_length = params['min_seq_length'],
            max_seq_length = params['max_seq_length'],
            device = self.pipeline_args['DEVICE'],
        )

    def init_lambdas(self, num_nodes : int):
        self.lambdas = {}
        if self.federated_args['lambdas'] == 'uniform':
            for node_id in range(1, num_nodes+1):
                self.lambdas[node_id] = self.federated_args['lambda_n']

    def build_nodes(self):
        """
        Builds N nodes with f byzantine. The type of byzantine nodes can be 
        """
        self.num_nodes = self.federated_args['num_training_nodes']
        self.num_bysantine = self.federated_args['num_byzantine']
        self.byzantine_type = self.federated_args['byzantine_type']
        self.init_lambdas(self.num_nodes)
        if self.num_nodes < self.num_bysantine:
            logging.error("The number of byzantine nodes can't be superior to the total number of users")
            sys.exit(1)
        self.nodes = {}
        for node_id in range(1, self.num_nodes+1):
            parameters = {
                'id_' : node_id,
                'lambda_' : self.lambdas[node_id],
                'p' : self.federated_args['p_n'],
                'vocabulary' : self.vocabulary,
                'min_seq_length' : self.federated_args['min_seq_length'],
                'max_seq_length' : self.federated_args['max_seq_length'],
                'device' : self.federated_args['DEVICE']
            }
            if node_id < self.num_bysantine + 1:
                if self.byzantine_type == 'null':
                    self.nodes[node_id] = NullByzantineNode(**parameters)
                elif self.byzantine_type == 'random':
                    self.nodes[node_id] = RandomByzantineNode(**parameters)
                elif self.byzantine_type == 'strategic':
                    self.nodes[node_id] = StrategicalByzantineNode(**parameters)
            else:
                self.nodes[node_id] = UserNode(
                    datafolder = self.federated_args['nodes_data_folder'],
                    **parameters
                )
        logging.info(f'generated {self.num_nodes} nodes with {self.num_bysantine} byzantine')

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
        """
        Given a node id, saves the rnn and linear weights of the current node model. If the id is 0, will save 
        the general model.

        Parameters
        ----------
        - node_id : int
            The node to save to.
        """
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

        optim_stat_dict = model.optimizer.state_dict()
        optim_path = os.path.join(self.optim_folder, 'optim_general.pth' if node_id == 0 else f"optim_{node_id}.pth")
        torch.save(optim_stat_dict, optim_path)

    def load_weights(self, node_id : int = 0, model : NextWordPredictorModel = None):
        """
        Given a node id and a NextWordPredictoModel, loads the rnn and linear weights from the 
        corresponding node in the model. If the id is 0, will load to the general model

        Parameters
        ----------
        - node_id : int
            The id of the node
        - model : NextWordPredictorModel
            The model to load the weigths in
        """
        if model is None:
            model = self.general_model
        rnn_path = os.path.join(self.rnn_folder, 'rnn_general.pth' if node_id == 0 else f"rnn_{node_id}.pth")
        linear_path = os.path.join(self.linear_folder, 'linear_general.pth' if node_id == 0 else f"linear_{node_id}.pth")
        optim_path = os.path.join(self.optim_folder, 'optim_general.pth' if node_id == 0 else f"optim_{node_id}.pth")
        with torch.no_grad():
            rnn_state_dict = torch.load(rnn_path)
            model.rnn.load_state_dict(rnn_state_dict)

            linear_state_dict = torch.load(linear_path)
            model.linear.load_state_dict(linear_state_dict)

            optim_stat_dict = torch.load(optim_path)
            model.optimizer.load_state_dict(optim_stat_dict)


    def load_model(self, path = None):
        self.general_model.load_model(path)

    def prepare_models_for_training(self):
        """
        Prepares the general model and the node model for training. Inintializes
        a second model, shares the embeddings and freezes them and setup the lamdba_0
        and p_0 paramters for the general model.
        """
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

    def nodes_epoch_step(self, epoch):
        """
        Epoch step passing trough every node. For every node it loads the corresponding
        weights and dataloader and performs an epoch step with the given regularizer. Then
        the backward propagation is also performed on the general model at the end.

        Parameters
        ----------
        - epoch : int
            The current epoch
        
        Returns
        -------
        - users_losses : dict
            The nodes training losses
        """
        # PASS THROUGHT THE NODES
        for node_id in tqdm(range(1, self.federated_args['num_training_nodes'] + 1)):
            # At the first epoch all nodes start from the init model
            node = self.nodes[node_id]
            if epoch == 0:
                self.load_weights(0, self.user_model)
            else:
                self.load_weights(node_id, self.user_model)

                node_dataloader = torch.utils.data.DataLoader(
                    node.data,
                    batch_size = 32,
                    shuffle = True,
                    drop_last = True
                )
                self.user_model.regularizer = self.models_difference
                user_total_losses, user_losses, user_reg_losses = self.user_model.epoch_step(
                    node_dataloader,
                    node,
                    with_tqdm = False,
                    sep_losses=True
                )
                node.losses['total_loss'].append(np.mean(user_total_losses))
                node.losses['loss'].append(np.mean(user_losses))
                node.losses['reg_loss'].append(np.mean(user_reg_losses))
            
            self.save_weights(node_id) 

    def train(self, num_max_epochs):
        """
        Trains the nodes and the general model alternatively. The pass is done through
        all the nodes with their respective loss and regularizer depeneding on the general
        model and then the update is made on the regularizer.

        Parameters
        ----------
        - num_max_epochs : int
            maximum number of epochs
        
        Returns
        -------
            node_losses : dict
                train loss at every epoch of every node
            general_model_val_losses : dict
                validation loss of the general model at every epoch
        """
        self.prepare_models_for_training()
        self.general_model_val_losses = {}
        for epoch in range(num_max_epochs+1):
            # Performs the full pass trough the data for every node
            for param in self.general_model.parameters():
                param.grad = None
            self.nodes_epoch_step(epoch)
            # UPDATE OF THE GENERAL MODEL GRADIENT
            # adds the general model regularization loss and its gradient
            general_model_reg_loss = self.general_model.regularizer()
            general_model_reg_loss.backward()
            # performs the general model optimization step
            self.general_model.optimizer.step()
            # evaluates the general model performances
            val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size = 32, drop_last = True)
            general_model_val_loss = self.general_model.evaluate(val_dataloader)
            self.general_model.train()
            self.general_model_val_losses[epoch] = general_model_val_loss

        self.plot_training_history()

    def generate_general(self, start_text : str, num_words : int = 100):
        return self.general_model.generate(start_text=start_text, vocabulary = self.vocabulary, num_words=num_words)

    def generate_node(self, start_text : str, node_id : int, num_words : int = 100):
        self.load_weights(node_id, self.user_model)
        return self.user_model.generate(start_text=start_text, vocabulary = self.vocabulary, num_words=num_words)

    def plot_training_history(self):
        if self.num_bysantine > 0:
            num_rows = 2
        else:
            num_rows = 1
        losses_types = self.nodes[1].losses.keys()
        num_cols = len(losses_types)
        axs = plt.figure(figsize = (16,8)).subplots(num_rows, num_cols)
        for i, (ax, loss_type) in enumerate(zip(axs[0],losses_types)):
            for node in self.nodes.values():
                if isinstance(node, UserNode):
                    if i == 0:
                        ax.set_ylabel('UserNode loss')
                    ax.set_title(loss_type)
                    ax.plot(node.losses[loss_type])
        for j, (ax, loss_type) in enumerate(zip(axs[1], losses_types)):
            for node in self.nodes.values():
                if not isinstance(node, UserNode):
                    if j == 0:
                        ax.set_ylabel('Byzantine loss')
                    ax.plot(node.losses[loss_type])
        plt.show()

if __name__ == '__main__':
    logging.basicConfig(filename = 'federated.log', encoding = 'utf-8', level=logging.DEBUG)
    pass