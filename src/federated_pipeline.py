import json
import os
import logging

from src.models import Pipeline
from src.utils import make_dir_if_not_exists

import torch

class Federated():
    def __init__(
        self,
        pipeline_args : str,
        federated_args : str,
        load_model : bool = False
    ):
        self.pipeline = Pipeline(pipeline_args)
        with open(federated_args, 'r') as f:
            self.federated_args = json.load(f)
        self.num_nodes = self.federated_args['num_nodes']
        self.prepare_directories()
        if load_model:
            self.load_embeddings()
            self.load_weights(0)

    def prepare_directories(self):
        self.model_dir = self.federated_args['model_dir']
        make_dir_if_not_exists(self.model_dir)
        self.embeddings_path = os.path.join(self.federated_args['embeddings_folder'], 'embeddings.pth')
        make_dir_if_not_exists(self.federated_args['embeddings_folder'])
        self.rnn_folder = self.federated_args['rnn_folder']
        make_dir_if_not_exists(self.rnn_folder)
        self.linear_folder = self.federated_args['linear_folder']
        make_dir_if_not_exists(self.linear_folder)
        
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
            for weights_type, weights in rnn_state_dict.items():
                self.pipeline.model.rnn[weights_type].copy_(weights)

            linear_state_dict = torch.load(linear_path)
            for weights_type, weights in linear_state_dict.items():
                self.pipeline.model.linear[weights_type].copy_(weights)

    def licchavi_loss_general(self):
        pass

    def licchavi_loss_node(self, model):
        pass

    def epoch_step(self, epoch):
        for node_id in range(1, self.num_nodes+1):
            if epoch == 0:
                pass




if __name__ == '__main__':
    logging.basicConfig(filename = 'federated.log', encoding = 'utf-8', level=logging.DEBUG)
    pass