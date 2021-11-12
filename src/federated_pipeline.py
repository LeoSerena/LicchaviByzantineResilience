from abc import abstractclassmethod, abstractmethod
import json
import sys
import os
import logging
import pickle
import gc
import re
from datetime import date

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('.')
from src.data_processing import  SequenceDataset
from src.models import NextWordPredictorModel, init_model
from src.utils import make_dir_if_not_exists, split_data, update_json

import torch

class Node():
    def __init__(
        self,
        id_ : int,
        lambda_ : float,
        p : int
    ):
        if self.__class__ == Node:
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
            if re.match(r'node_'+str(id_)+r'_.*\.pickle', file):
                self.file = file

        self.num_bodies = int(re.sub('\.pickle', '', self.file.split('_')[2]))

        with open(os.path.join(datafolder, self.file), 'rb') as f:
            data = pickle.load(f)

        train_set, val_set, test_set = split_data(data)

        self.data = SequenceDataset(
            vocabulary = vocabulary,
            text = train_set,
            min_seq_length = min_seq_length,
            max_seq_length = max_seq_length,
            device = device,
            with_tqdm=False
        )

        self.val = SequenceDataset(
            vocabulary = vocabulary,
            text = val_set,
            min_seq_length = min_seq_length,
            max_seq_length = max_seq_length,
            device = device,
            with_tqdm=False
        )

        self.test = SequenceDataset(
            vocabulary = vocabulary,
            text = test_set,
            min_seq_length = min_seq_length,
            max_seq_length = max_seq_length,
            device = device,
            with_tqdm=False
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
        N,
        vocabulary,
        min_seq_length,
        max_seq_length,
        device
    ):
        super(ByzantineNode, self).__init__(id_,lambda_,p)
        token = 'day'
        data = [' '.join([token] * max_seq_length)] * N
        self.data = SequenceDataset(
            vocabulary = vocabulary,
            text = data,
            max_seq_length = max_seq_length,
            min_seq_length = min_seq_length,
            device = device,
            with_tqdm=False
        )


class RandomByzantineNode(ByzantineNode):
    def __init__(
            self,
            id_,
            lambda_,
            p,
            N,
            vocabulary,
            min_seq_length,
            max_seq_length,
            device 
        ):
        super(ByzantineNode, self).__init__(id_, lambda_, p)
        random_tokens = np.random.randint(low = 1, high = vocabulary.get_vocab_size(), size = int(N * max_seq_length))
        random_data = np.array([vocabulary.idx_to_word[x] for x in random_tokens]).reshape((N, max_seq_length)).tolist()
        random_data = [' '.join(x) for x in random_data]
        self.data = SequenceDataset(
            vocabulary = vocabulary,
            text = random_data,
            max_seq_length = max_seq_length,
            min_seq_length = min_seq_length,
            device = device
        )

class StrategicalByzantineNode(ByzantineNode):
    def __init__(
            self,
            id_,
            lambda_,
            p,
            N,
            vocabulary,
            min_seq_length,
            max_seq_length,
            device 
        ):
        super(ByzantineNode, self).__init__(id_, lambda_, p)
        sentence = 'all work and no play makes jack a dull boy'.split(' ')
        


class Federated():
    def __init__(
        self,
        pipeline_args,
        federated_args,
        load_model_from,
        testing = False
    ):
        if self.__class__ == Federated:
            raise NotImplementedError("""This is an abstract class""")
        self.testing = testing
        # READ CONFIG FILES
        federated_args = os.path.join('config_files', federated_args)
        with open(federated_args, 'r') as f:
            self.federated_args = json.load(f)
            logging.info('federated arguments loaded')
        pipeline_args = os.path.join('config_files', pipeline_args)
        with open(pipeline_args, 'r') as f:
            self.pipeline_args = json.load(f)
            logging.info('pipeline arguments loaded')

        
        # setting seeds
        torch.manual_seed(self.pipeline_args['TORCH_SEED'])
        np.random.seed(self.pipeline_args['NUMPY_SEED'])
        # LOAD VOCAB
        vocab_path = os.path.join('vocabs', self.pipeline_args['DATA_PARAMETERS']['vocab_file'])
        with open(vocab_path, 'rb') as f:
            self.vocabulary = pickle.load(f)
            logging.info('vocabulary loaded')
        self.prepare_directories()
        self.load_val_test_set()


        # INIT GENERAL MODEL
        self.model_parameters = self.pipeline_args['MODEL_PARAMETERS']
        self.model_parameters['device'] = self.pipeline_args['DEVICE']
        self.model_parameters['vocab_size'] = self.vocabulary.get_vocab_size()
        self.model_parameters['LEARNING_RATE'] = self.federated_args['general_model_lr']
        self.general_model = init_model(None,  **self.model_parameters)
        
        if load_model_from is None:
            self.load_model_from = os.path.join(
                self.pipeline_args['TRAINING_PARAMETERS']['model_path'],
                self.pipeline_args['TRAINING_PARAMETERS']['model_name']
            )
        else:
            self.load_model_from = load_model_from
        self.load_model(path = self.load_model_from)
        self.save_embeddings()
        self.save_weights()

        # INIT NODES
        self.num_nodes = self.federated_args['num_nodes']
        self.build_nodes()
        
        self.prepare_models_for_training()

    
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

        self.results_folder = self.federated_args['results_folder']
        make_dir_if_not_exists(self.results_folder)
    
    def load_val_test_set(self):
        data_params = self.pipeline_args['DATA_PARAMETERS']
        if data_params['data_name'] == 'tweets':
            test_set_file = os.path.join(data_params['data_folder'], 'test_2.pickle')
            val_set_file = os.path.join(data_params['data_folder'], 'val_2.pickle')
        elif data_params['data_name'] == 'WikiText-2':
            test_set_file = os.path.join(data_params['data_folder'], 'wikitext-2', 'test_2.pickle')
            val_set_file = os.path.join(data_params['data_folder'], 'wikitext-2', 'val_2.pickle')
        elif data_params['data_name'] == 'WikiText103':
            test_set_file = os.path.join(data_params['data_folder'], 'wikitext-3', 'test_103.pickle')
            val_set_file = os.path.join(data_params['data_folder'], 'wikitext-3', 'val_103.pickle')

        with open(test_set_file, 'rb') as f:
            test_set = pickle.load(f)
        with open(val_set_file, 'rb') as f:
            val_set = pickle.load(f)

        self.val_dataset = SequenceDataset(
            vocabulary = self.vocabulary,
            text = val_set[:5000] if self.testing else val_set[:50000],
            min_seq_length = data_params['min_seq_length'],
            max_seq_length = data_params['max_seq_length'],
            device = self.pipeline_args['DEVICE'],
        )
        self.test_dataset = SequenceDataset(
            vocabulary = self.vocabulary,
            text = test_set[:5000] if self.testing else test_set,
            min_seq_length = data_params['min_seq_length'],
            max_seq_length = data_params['max_seq_length'],
            device = self.pipeline_args['DEVICE'],
        )
        logging.info(f"""
        loaded validation set ({len(self.val_dataset)}) and test set ({self.test_dataset})
        """)

    def build_nodes(self):
        """
        Builds N nodes with f byzantine. The type of byzantine nodes can be 
        """
        self.num_nodes = self.federated_args['num_training_nodes']
        self.num_bysantine = self.federated_args['num_byzantine']
        self.byzantine_type = self.federated_args['byzantine_type']
        self.byzantine_datasize = self.federated_args['byzantine_datasize']
        self.init_lambdas(self.num_nodes)
        if self.num_nodes < self.num_bysantine:
            logging.error("The number of byzantine nodes can't be superior to the total number of users")
            sys.exit(1)
        self.nodes = {}
        nodes_path = os.path.join('nodes_data', self.federated_args['nodes_data_folder'])
        for node_id in tqdm(range(1, self.num_nodes+1)):
            parameters = {
                'id_' : node_id,
                'lambda_' : self.lambdas[node_id],
                'p' : self.federated_args['p_n'],
                'vocabulary' : self.vocabulary,
                'min_seq_length' : self.federated_args['min_seq_length'],
                'max_seq_length' : self.federated_args['max_seq_length'],
                'device' : self.federated_args['DEVICE']
            }
            if node_id < self.num_nodes - self.num_bysantine + 1:
                self.nodes[node_id] = UserNode(
                    datafolder = nodes_path,
                    **parameters
                )
            else:
                parameters['N'] = self.byzantine_datasize
                if self.byzantine_type == 'null':
                    self.nodes[node_id] = NullByzantineNode(**parameters)
                elif self.byzantine_type == 'random':
                    self.nodes[node_id] = RandomByzantineNode(**parameters)
                elif self.byzantine_type == 'strategic':
                    self.nodes[node_id] = StrategicalByzantineNode(**parameters)

        logging.info(f'generated {self.num_nodes} nodes with {self.num_bysantine} byzantine')

    def init_lambdas(self, num_nodes : int):
        self.lambdas = {}
        if self.federated_args['lambdas'] == 'uniform':
            for node_id in range(1, num_nodes+1):
                self.lambdas[node_id] = self.federated_args['lambda_n']

    def load_model(self, path = None):
        self.general_model.load_model(path)

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
        corresponding node in the model. If the id is 0, will load to the general model. It also
        loads the optimizer specific to the id.

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

    def generate_general(self, start_text : str, num_words : int = 100):
        return self.general_model.generate(start_text=start_text, vocabulary = self.vocabulary, num_words=num_words)

    def generate_node(self, start_text : str, node_id : int, num_words : int = 100):
        self.load_weights(node_id, self.user_model)
        return self.user_model.generate(start_text=start_text, vocabulary = self.vocabulary, num_words=num_words)

    def plot_training_history(self, save_results, plt_name):
        if self.num_bysantine > 0:
            num_rows = 2
        else:
            num_rows = 1
        losses_types = self.nodes[1].losses.keys()
        num_cols = len(losses_types)
        axs = plt.figure(figsize = (16,8)).subplots(num_rows, num_cols)
        plt.suptitle(f"""$\lambda_0 = {self.general_model.gamma}$  $p_0 = {self.general_model.q}$  $lr_0 = {self.federated_args['general_model_lr']}$
$\lambda_n = {self.nodes[1].lambda_}$  $p_n = {self.nodes[1].p}$ $lr_n = {self.federated_args['node_model_lr']}$""")
        row_1 = axs[0] if num_rows > 1 else axs
        for i, (ax, loss_type) in enumerate(zip(row_1,losses_types)):
            for node in self.nodes.values():
                if isinstance(node, UserNode):
                    if i == 0:
                        ax.set_ylabel(f'UserNode loss ({self.num_nodes - self.num_bysantine})')
                    ax.set_title(loss_type)
                    ax.plot(node.losses[loss_type])
        if num_rows > 1:
            for j, (ax, loss_type) in enumerate(zip(axs[1], losses_types)):
                for node in self.nodes.values():
                    if not isinstance(node, UserNode):
                        if j == 0:
                            ax.set_ylabel(f'{self.byzantine_type} Byzantine loss ({self.num_bysantine})')
                        ax.plot(node.losses[loss_type])

        today = date.today().strftime("%m-%d-%y")
        filename_nodes = today + f"_nodes_{plt_name}.jpg"

        if save_results:
            plot_folder = self.save_results()
            plt.savefig(os.path.join(plot_folder, filename_nodes))
        else:
            plt.show()
        plt.close()

        plt.figure()
        plt.plot(self.general_model_val_losses.values())
        plt.title('Generale Model Validation loss')

        filename_general = today + f"_general_{plt_name}.jpg"
        if save_results:
            plt.savefig(os.path.join(plot_folder, filename_general))
            
        else:
            plt.show()
        plt.close()

    @abstractclassmethod
    def prepare_models_for_training(self):
        raise NotImplementedError

    def train(self, num_max_epochs, save_results = False, plt_name = ''):
        self.general_model_val_losses = {}
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size = 32, drop_last = True, shuffle = False)
        for epoch in range(num_max_epochs+1):
            self.general_model.train()
            for param in self.general_model.parameters():
                param.grad = None
            self.nodes_epoch_step(epoch)
            self.general_model_update()
            general_model_val_loss = self.general_model.evaluate(val_dataloader)
            self.general_model_val_losses[epoch] = general_model_val_loss
        self.plot_training_history(save_results, plt_name)

    @abstractclassmethod
    def nodes_epoch_step(self, epoch):
        raise NotImplementedError

    @abstractclassmethod
    def general_model_update(self):
        raise NotImplementedError

    def test_loss_perplexity_recall(self):
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size = 1,
            drop_last = True,
            shuffle = False
        )
        # metrics:
        results = {}
        results['val_loss'] = self.general_model_val_losses
        results['test_perplexity'], results['test_loss'], results['f1_recall'], results['f3_recall'] = self.general_model.perplexity(
            test_dataloader,
            with_tqdm = True,
            with_recall = True
        )
        # data info:
        results['val_len'] = len(self.val_dataset)
        results['val_tokens'] = self.val_dataset.token_len()
        results['test_len'] = len(self.test_dataset)
        results['test_tokens'] = self.test_dataset.token_len()
        if isinstance(self, Federated_ATT) or isinstance(self, Federated_LICCHAVI):
            for node_id, node in self.nodes.items():
                if isinstance(node, UserNode):
                    self.load_weights(node_id, self.general_model)
                    test_dataloader = torch.utils.data.DataLoader(
                        node.test,
                        batch_size = 1,
                        drop_last = True,
                        shuffle = False
                    )
                    # metrics:
                    (
                        results[f'test_perplexity_{node_id}'],  results[f'test_loss_{node_id}'], 
                        results[f'f1_recall_{node_id}'], results[f'f3_recall_{node_id}']
                    ) = self.general_model.perplexity(
                        test_dataloader,
                        with_tqdm = False,
                        with_recall=True
                    )
                    results[f'val_loss_{node_id}'] = node.losses
                    # data info:
                    results[f'train_len_{node_id}'] = len(node.data)
                    results[f'train_tokens_{node_id}'] = len(node.data)
                    results[f'val_len_{node_id}'] = len(node.val)
                    results[f'val_tokens_{node_id}'] = len(node.val)
                    results[f'test_len_{node_id}'] = len(node.test)
                    results[f'test_tokens_{node_id}'] = len(node.test)

        return results


    def get_name(self):
        if isinstance(self, Federated_SGD):
            return 'FedSGD'
        elif isinstance(self, Federated_AVG):
            return 'FedAVG'
        elif isinstance(self, Federated_ATT):
            return 'FedATT'
        elif isinstance(self, Federated_LICCHAVI):
            return 'LICCHAVI'
        else:
            return 'Fed'

    def save_results(self):
        results_path = self.federated_args['results_folder']
        path = os.path.join(results_path, self.pipeline_args['DATA_PARAMETERS']['data_name'])
        make_dir_if_not_exists(path)
        path = os.path.join(path, self.get_name())
        make_dir_if_not_exists(path)
        # path is of format *resultsFolder*/*dataType*/*FedAlg*/*date*_*id*/

        id_ = len(os.listdir(path))
        path = os.path.join(path, f"{str(date.today())}_{id_}")
        make_dir_if_not_exists(path)

        # Hyperparameters saving
        hyperparameters_path = os.path.join(path, "hyperparams.pickle")
        hyperparameters = self.pipeline_args
        hyperparameters['FEDERATED_ARGS'] = self.federated_args
        with open(hyperparameters_path, 'wb') as f:
            pickle.dump(hyperparameters, f)

        # history and results saving
        metrics_path = os.path.join(path, "metrics.pickle")
        metrics = self.test_loss_perplexity_recall()
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)

        return path


class Federated_SGD(Federated):
    def __init__(
        self,
        pipeline_args : str,
        federated_args : str,
        load_model_from = None,
        testing = False
    ):
        super(Federated_SGD, self).__init__(
            pipeline_args,
            federated_args,
            load_model_from,
            testing = testing
        )
    
    def prepare_models_for_training(self):
        self.general_model.train()
        self.general_model.freeze_embeddings()  
        # Current general model is stored in state dict
        self.current_state_dict = self.general_model.state_dict()
        # save an optimizer to reinitialize for every node
        self.optim_stat_dict = self.general_model.optimizer.state_dict()

    def weigthed_avg(self, N, total_data):
        # perform node weighted average
        node_state_dict = self.general_model.state_dict()
        if self.agg_state_dict is None:
            self.agg_state_dict = node_state_dict
            for key in self.agg_state_dict:
                self.agg_state_dict[key] = self.agg_state_dict[key] * N / total_data
        else:
            for key in self.agg_state_dict:
                if self.agg_state_dict[key].requires_grad:
                    self.agg_state_dict[key] = self.agg_state_dict[key] + self.node_state_dict[key] * N / total_data

    def nodes_epoch_step(self, epoch):
        """[summary]

        :param epoch: [description]
        :type epoch: [type]
        """
        total_data = 0
        self.agg_state_dict = None
        for node_id in tqdm(range(1, self.federated_args['num_training_nodes'] + 1)):
            # At the first epoch all nodes start from the init model
            node = self.nodes[node_id]

            N = len(node.data)
            total_data += N

            node_dataloader = torch.utils.data.DataLoader(
                node.data,
                batch_size = 32,
                shuffle = True,
                drop_last = True
            )
            # loads general model
            self.general_model.load_state_dict(self.current_state_dict)
            self.general_model.optimizer.load_state_dict(self.optim_stat_dict)
            if epoch == 0:
                user_total_losses, user_losses, user_reg_losses = self.general_model.evaluate(
                    dataloader = node_dataloader, 
                    node = node, 
                    eval_mode = False,
                    with_tqdm = False,
                    sep_losses = True
                )
            else:
                user_total_losses, user_losses, user_reg_losses = self.general_model.epoch_step(
                    node_dataloader,
                    node,
                    with_tqdm = False,
                    sep_losses=True
                )
            node.losses['total_loss'].append(np.mean(user_total_losses))
            node.losses['loss'].append(np.mean(user_losses))
            node.losses['reg_loss'].append(np.mean(user_reg_losses))

            self.weigthed_avg(N, total_data)

    def general_model_update(self):
        self.current_state_dict = self.agg_state_dict

class Federated_AVG(Federated):
    def __init__(
        self,
        pipeline_args : str,
        federated_args : str,
        load_model_from = None
    ):
        super(Federated_AVG, self).__init__(
            pipeline_args,
            federated_args,
            load_model_from
        )

    def prepare_models_for_training(self):
        self.general_model.train()
        self.general_model.freeze_embeddings()  
        # Current general model is stored in state dict
        self.current_state_dict = self.model.state_dict()
        self.general_model.lambda_ = self.federated_args['lambda_n']
        self.general_model.p = self.federated_args['p_n']

class Federated_ATT(Federated):
    pass

class Federated_LICCHAVI(Federated):
    def __init__(
        self,
        pipeline_args : str,
        federated_args : str,
        load_model_from = None,
        testing = False
    ):
        super(Federated_LICCHAVI, self).__init__(
            pipeline_args,
            federated_args,
            load_model_from,
            testing
        )

    def prepare_models_for_training(self, share_embeddings = True):
        """
        Prepares the general model and the node model for training. Inintializes
        a second model, shares the embeddings and freezes them and setup the lamdba_0
        and p_0 paramters for the general model.
        """
        self.general_model.train()
        self.general_model.freeze_embeddings()        
        # Initialize the model for the users
        self.model_parameters['LEARNING_RATE'] = self.federated_args['node_model_lr']

        self.user_model = init_model(None, **self.model_parameters)
        self.user_model.lambda_ = self.federated_args['lambda_n']
        self.user_model.p = self.federated_args['p_n']
        self.user_model.load_model(path = self.load_model_from)
        self.user_model.train()

        if share_embeddings:
            del self.user_model.embedding_layer
            gc.collect()
            # This way they share the embeddings layer to save memory
            self.user_model.embedding_layer = self.general_model.embedding_layer
        self.general_model.gamma = self.federated_args['lambda_0']
        self.general_model.q = self.federated_args['p_0']

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
            node_dataloader = torch.utils.data.DataLoader(
                node.data,
                batch_size = 32,
                shuffle = True,
                drop_last = True
            )
            self.user_model.general_regularizer = self.models_difference
            if epoch == 0:
                self.load_weights(0, self.user_model)
                user_total_losses, user_losses, user_reg_losses = self.user_model.evaluate(
                    dataloader = node_dataloader, 
                    node = node,
                    eval_mode = False,
                    sep_losses = True,
                    with_tqdm = False
                )
            else:
                self.load_weights(node_id, self.user_model)
                user_total_losses, user_losses, user_reg_losses = self.user_model.epoch_step(
                    node_dataloader,
                    node,
                    with_tqdm = False,
                    sep_losses = True
                )
            node.losses['total_loss'].append(np.mean(user_total_losses))
            node.losses['loss'].append(np.mean(user_losses))
            node.losses['reg_loss'].append(np.mean(user_reg_losses))
            
            self.save_weights(node_id) 

    def general_model_update(self):
        # UPDATE OF THE GENERAL MODEL GRADIENT
        # adds the general model regularization loss and its gradient
        general_model_reg_loss = self.general_model.regularizer()
        general_model_reg_loss.backward()
        # performs the general model optimization step
        self.general_model.optimizer.step()


def grid_search(federated_alg, dataType):

    if dataType == 'tweet':
        model_file = "CONFIG_MODEL_TWEETS.json"
        fed_file = "CONFIG_FEDERATED_TWEETS.json"
    else:
        model_file = "CONFIG_MODEL_WIKI.json"
        fed_file = "CONFIG_FEDERATED_WIKI.json"

    if federated_alg == 'FedSGD':
        # For FedSGD, we only need to grid search learning rates
        federated = Federated_SGD
        for lr in [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
            update_json(
                os.path.join('.','config_files', fed_file), node_model_lr = lr )
            federated = federated_alg(model_file, fed_file)
            federated.train(10, save_results = True, plt_name = f'type:{dataType}|alg:{federated.get_name()} || lr:{lr}|')
    elif arguments[1] == 'FedAVG':
        federated = Federated_AVG
    elif arguments[1] == 'FedATT':
        federated = Federated_ATT
    else:
        federated = Federated_LICCHAVI


    for lr in [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
        for lambda_0 in [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
            update_json(
                os.path.join('.','config_files', fed_file),
                general_model_lr = lr,
                node_model_lr = lr,
                lambda_0 = lambda_0
            )
            federated = federated_alg(model_file, fed_file)
            federated.train(
                10,
                save_results = True, 
                plt_name = f'type:{dataType}|alg:{federated.get_name()} || lr:{lr}|lamda_0:{lambda_0}'
            )


if __name__ == '__main__':
    """[summary]
    """
    arguments = sys.argv

    if len(arguments) != 4:
        print('invalid arguments')
        sys.exit(1)
    elif arguments[1] in ['FedSGD', 'FedAVG', 'FedATT', 'LICCHAVI']:
        if arguments[3] not in ['tweet', 'wiki']:
            print('invalid arguments')
            sys.exit(1)

        if arguments[2] == 'grid':
            grid_search(
                arguments[1],
                arguments[3]
            )
        elif arguments[2] == 'attack':
            pass
        else:
            print('invalid arguments')
            sys.exit(1)


    else:
        print('invalid arguments')
        sys.exit(1)