import os
import math
import json
import gc
import logging
import pickle
import sys
import time
import re
from typing import Union, Tuple

sys.path.append('.')
from src.utils import make_dir_if_not_exists, update_json
from src.data_processing import FromTweetsVocabulary, FromRawTextVocabulary, \
    Vocabulary, SequenceDataset, text_cleaner_raw
from src.nodes import Node

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import TweetTokenizer, word_tokenize

from apex import amp
import torch
from torch import Tensor


class PositionalEncoding(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        """torch module implementing positionial encoding

        :param emb_dim: embedding dimention
        :type emb_dim: int
        :param dropout: dropout, defaults to 0.1
        :type dropout: float, optional
        :param max_len: maximum length of sequence, defaults to 5000
        :type max_len: int, optional
        """    
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(max_len, 1, emb_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        - x : torch.Tensor
            input tensor of shape [batch_size, seq_len, embedding_dim]
        """
        x = x + torch.transpose(self.pe[:x.size(1)], 0,1)
        return self.dropout(x)

class NextWordPredictorModel(torch.nn.Module):
    def __init__(
        self,
        type_of_rnn : str,
        emb_dim : int,
        vocab_size : int,
        num_rnn_hidden_layers : int,
        hidden_state_size : int,
        dropout : float,
        device : str,
        weight : list = None,
        positional_encoding : bool = False,
        tied_embeddings : bool = False,
        q : int = 2,
        gamma : float = 1e-3
    ):
        """Torch.nn.MModule for next word prediction using RNNs.

        :param type_of_rnn: The type of RNN used. Can be ['LSTM', 'GRU']. Default: RNN
        :type type_of_rnn: str
        :param emb_dim: Dimension of the embeddings vectors
        :type emb_dim: int
        :param vocab_size: The number of tokens in the vocabulary
        :type vocab_size: int
        :param num_rnn_hidden_layers: The number of rnn layers
        :type num_rnn_hidden_layers: int
        :param hidden_state_size: The hidden state size (and for LSTM the cell state size)
        :type hidden_state_size: int
        :param dropout: The dropout to be used on positional ecoding (if used) and RNN layer
        :type dropout: float
        :param device: The device used for training
        :type device: str
        :param weight: The addaptive weights to be used on the tokens fo the loss. 
        Must be of the same size as the number of tokens, aka vocab_size., defaults to None
        :type weight: list, optional
        :param positional_encoding: Whether to use the positional encoding on top of the embeddings, defaults to False
        :type positional_encoding: bool, optional
        :param tied_embeddings: Whether to tied embeddings encoder weigths with decoder weights, defaults to False
        :type tied_embeddings: bool, optional
        :param q: The norm to use for regularization, defaults to 2
        :type q: int, optional
        :param gamma: The weight of the regularizer, defaults to 1e-3
        :type gamma: float, optional
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.num_rnn_hidden_layers = num_rnn_hidden_layers
        self.hidden_state_size = hidden_state_size
        self.device = device
        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.type_of_rnn = type_of_rnn
        self.tied_embeddings = tied_embeddings
        self.q = q
        self.gamma = gamma
        if tied_embeddings:
            print('tied_embeddings not yet implemented')
        
        # Embedding layer
        self.embedding_layer = torch.nn.Embedding(
            self.vocab_size,
            emb_dim,
            padding_idx = 0
        ).to(device)
        # positional encoder
        if positional_encoding:
            self.positional_encoder = PositionalEncoding(
                emb_dim,
                dropout
            )
        inputs = {
            'input_size' : emb_dim,
            'hidden_size' : hidden_state_size,
            'num_layers' : num_rnn_hidden_layers,
            'dropout' : dropout,
            'batch_first' : True # -> input of the shape (bath size, seq length, emb length)
        }
        self.has_cell_state = False
        if type_of_rnn == 'LSTM':
            self.rnn = torch.nn.LSTM(**inputs).to(device)
            self.has_cell_state = True
        elif type_of_rnn == 'GRU':
            self.rnn = torch.nn.GRU(**inputs).to(device)
        else:
            print('Default torch.nn.RNN used')
            self.rnn = torch.nn.RNN(**inputs).to(device)

        self.linear = torch.nn.Linear(
            hidden_state_size,
            self.vocab_size
        ).to(device)
        
        self.criterion = torch.nn.CrossEntropyLoss(
            weight = torch.FloatTensor([weight]).to(self.device) if weight is not None else None,
            ignore_index = 0,
            reduction = 'mean'
        ).to(device) # may use the weight as prior n_occ / num_words
        
        
        self.init_weights()

        
    def forward(
        self, 
        inputs : torch.Tensor, 
        hidden : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        :param inputs: The input Tensor of shape [batch size, sequence length, embedding length]
        :type inputs: torch.Tensor
        :param hidden: The hidden state of length self.hidden_state_size. 
        For LSTM the hidden state and the cell state.
        :type hidden: Union[torch.Tensor, Tuple(torch.Tensor, torch.Tensor)]
        :return: The scores for all tokens, of size of the vocabulary and the new hidden (and cell state for LSTM)
        :rtype: Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
        """        
        embeddings = self.embedding_layer(inputs)
        if self.positional_encoding:
            embeddings = self.positional_encoder(embeddings)
        output, hidden = self.rnn(embeddings, hidden)
        output = self.linear(output)
        return output, hidden
    
    def init_weights(self) -> None:
        """
        Initialize the weights for the embeddings and the linear layers.
        """
        initrange = 0.1
        self.embedding_layer.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
    
    def save_model(self, path = None):
        """
        Saves the model in the given path. If no path is given, automatically saved
        in the model_path specified at training.
        :param path: The path to save the model to, defaults to None
        :type path: str, optional
        """
        if path is not None:
            torch.save(self.state_dict(), path)
        else:
            if hasattr(self, 'model_path'):
                torch.save(self.state_dict(), os.path.join(self.model_path, 'model.pth'))
            else:
                print('No path given, please enter a path to save the model.')
    
    def load_model(self, path : str = None):
        """
        Loads the model from the given path. If no path is given, automatically loaded
        from the model_path specified at training.
        :param path: The path to load the model from, defaults to None
        :type path: str, optional
        """        
        if path is not None:
            self.load_state_dict(torch.load(path), strict = False)
        else:
            try:
                self.load_state_dict(torch.load(os.path.join('models', 'model.pth')))
            except FileNotFoundError:
                if hasattr(self, 'model_path'):
                    self.load_state_dict(torch.load(os.path.join(self.model_path, 'model.pth')))
                else:
                    print('No path given, please enter a path to load the model.')
    
    def init_hidden(self, batch_size : int) -> torch.Tensor:
        """Initializes the hidden states (ans cell states if LSTM).

        :param batch_size: batch size
        :type batch_size: int
        :return: The initialized tensor
        :rtype: torch.Tensor
        """        
        if self.has_cell_state:
            return (
                torch.zeros(
                    self.num_rnn_hidden_layers, batch_size, self.hidden_state_size
                ).to(self.device).detach(),
                torch.zeros(
                    self.num_rnn_hidden_layers, batch_size, self.hidden_state_size
                ).to(self.device).detach()
            )
        else:
            return torch.zeros(
                self.num_rnn_hidden_layers, batch_size, self.hidden_state_size
            ).to(self.device).detach()

    def perplexity(
        self, 
        dataloader : torch.utils.data.DataLoader, 
        with_recall = False, 
        with_tqdm = False
    ) -> Union[Tuple[float, float], Tuple[float, float, float, float]]:
        """Computes the perplexity and loss of the given data loader.
        Returns either (perp, loss) or (perp, loss, R1, R3).

        :param dataloader: data loader to evaluate
        :type dataloader: torch.utils.data.DataLoader
        :param with_recall: Whether to also compute $R_1$ & $R_3$, defaults to False
        :type with_recall: bool, optional
        :param with_tqdm: Whether to display process evolution, defaults to False
        :type with_tqdm: bool, optional
        :return: [description]
        :rtype: [type]
        """        

        """
        Computes the perplexity of the model on the given dataset

        Parameters
        ----------
        - dataloader : torch.utils.data.DataLoader
            The data to be tested
        """
        self.eval()
        m = torch.nn.Softmax(dim = 0)
        total_losses = []
        with torch.no_grad():
            total_tokens = 0
            probabilities = []
            if with_recall:
                total, top1hit, top3hit = 0,0,0
            for batch in tqdm(dataloader) if with_tqdm else dataloader:
                hidden = self.init_hidden(dataloader.batch_size)
                outputs, _ = self.forward(batch[:,:-1], hidden)
                outputs = torch.transpose(outputs, 1,2)
                labels = batch[:,1:]
                
                for i in range(dataloader.batch_size):
                    seq = outputs[i]
                    lab = labels[i]

                    logits = m(seq)

                    
                    if with_recall:
                        for top3, label in zip(torch.topk(logits.T,3)[1].cpu().numpy(), lab.cpu().numpy()):
                            if label not in [0,1]:
                                if label in top3:
                                    top3hit += 1
                                if label == top3[0]:
                                    top1hit += 1
                                total += 1

                    logits = logits[lab,range(len(lab))]
                    logits = logits[lab != 0].cpu().numpy()
                    # logits = np.where(logits == 0, 1e-50, logits) # adds a regularizer to avoid explosion of the log
                    probabilities.append(sum(np.log(logits)))
                    total_tokens += len(logits)

                total_losses.append(self.criterion(outputs, labels).item())

        perplexity = np.exp(- np.sum(probabilities) / total_tokens)
        test_loss = np.mean(total_losses)
        if with_recall:
            f1_recall = top1hit / total
            f3_recall = top3hit / total
            return perplexity, test_loss, f1_recall, f3_recall
        else:
            return perplexity, test_loss
    
    def evaluate(
        self, 
        dataloader : torch.utils.data.DataLoader, 
        sep_losses : bool = False, 
        eval_mode : bool = True, 
        node : Node = None, 
        with_tqdm : bool = True
    ) -> Union[float, Tuple[float, float, float]]:
        """Evaluates the given data loader by computing the loss.
        Either returns the total loss or the regularization and the prediction
        loss separetly (total_loss, pred_loss, reg_loss).
        If the model has the 'general_regularizer' attribute, the reg_loss will be
        computed with respect to it instead of the weigth decay.

        :param dataloader: The data loader to be evaluated
        :type dataloader: torch.utils.data.DataLoader
        :param sep_losses: Whether to compute regularizer loss and samples loss separately, defaults to False
        :type sep_losses: bool, optional
        :param eval_mode: wheter to compute with the regularizer, defaults to True
        :type eval_mode: bool, optional
        :param node: Node to regularize on if needed, defaults to None
        :type node: Node, optional
        :param with_tqdm: Whether to display process evolution, defaults to True
        :type with_tqdm: bool, optional
        :return: The average loss on the input data
        :rtype: Union[float, Tuple[float, float, float]]
        """    
        self.eval()
        if sep_losses:
            sample_losses = []
            regularizer_losses = []

        total_losses = []
        with torch.no_grad():
            for batch in tqdm(dataloader) if with_tqdm else dataloader:
                hidden = self.init_hidden(dataloader.batch_size)
                outputs, _ = self.forward(batch[:,:-1], hidden)
                outputs = torch.transpose(outputs, 1,2)
                labels = batch[:,1:]
                
                loss = self.criterion(outputs, labels)
                if node is not None:
                    reg_loss = self.regularizer() / len(batch)
                    if hasattr(self, 'general_regularizer'):
                        loss = reg_loss + loss
                        reg_loss = self.general_regularizer(node)
                
                if eval_mode:
                    reg_loss = self.regularizer()
                    total_loss = loss + reg_loss
                else:
                    total_loss = loss
                

                total_losses.append(total_loss.item())
                if sep_losses:
                    sample_losses.append(loss.item())
                    regularizer_losses.append(reg_loss.item())
                
        if sep_losses:
            return total_losses, sample_losses, regularizer_losses
        return np.mean(total_losses)
        
    def epoch_step(
        self,
        data_loader : torch.utils.data.DataLoader,
        node : Node  = None, 
        with_tqdm : bool = True,
        sep_losses : bool = False
    ):
        """Performs a full epoch training step trough the data in the data loader.

        :param data_loader: The data loader to train with
        :type data_loader: torch.utils.data.DataLoader
        :param node: The Node for regularization, defaults to None
        :type node: Node, optional
        :param with_tqdm: Whether to display process evolution, defaults to True
        :type with_tqdm: bool, optional
        :param sep_losses: Whether to return training loss separetly, defaults to False
        :type sep_losses: bool, optional
        :return: The average loss on the input data (see evaluate())
        :rtype: Union[float, Tuple[float, float, float]]
        """
        self.train()

        if sep_losses:
            sample_losses = []
            regularizer_losses = []
        total_losses = []

        if with_tqdm:
            iterator = tqdm(data_loader)
        else:
            iterator = data_loader

        for batch in iterator:
            for param in self.parameters():
                param.grad = None
            hidden = self.init_hidden(data_loader.batch_size)
            outputs, _ = self.forward(batch[:,:-1], hidden)
            outputs = torch.transpose(outputs, 1, 2)
            labels = batch[:,1:]
            
            loss = self.criterion(outputs, labels)
            reg_loss = self.regularizer() / len(batch)
            # If we have a general model reg loss, we say that the reg loss is the latter
            # Otherwise it is the self model regularization loss
            if hasattr(self, 'general_regularizer'):
                loss = reg_loss + loss
                reg_loss = self.general_regularizer(node)

            total_loss = reg_loss + loss
            
            if self.fp16 == 1:
                with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
                
            total_losses.append(total_loss.item())
            if sep_losses:
                sample_losses.append(loss.item())
                regularizer_losses.append(reg_loss.item())
        
        # self.scheduler.step()
        
        if sep_losses:
            return total_losses, sample_losses, regularizer_losses
        else:   
            return total_losses
    
    def update_early_stopping(
        self,
        current_metric : float,
        epoch : int,
        path : str = None
    ) -> int:
        """
        Given a metric and an epoch, determines whether the model should stop early. It
        also uses the self.best_metric, self.best_epoch, self.early_stopping_count and
        self.early_stopping_patience to make the decision: if the metric doesn't improve
        for self.early_stopping_patience epochs, the model saved at the best eposh is loaded
        and the function returns 0.

        :param current_metric: The value of the metric at this epoch
        :type current_metric: float
        :param epoch: The current epoch
        :type epoch: int
        :param path: where to load/save the model, defaults to None
        :type path: str, optional
        :return: 0 if requires stopping 1 otherwise
        :rtype: Union[0, 1]
        """
        if self.early_stopping_metric_best == 'min':
            is_better = self.best_metric > current_metric 
        else:
            is_better = self.best_metric < current_metric
        if is_better:
            print('updating best metric')
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.early_stopping_count = 0
            self.save_model(path = path)
        else:
            self.early_stopping_count+=1
        if self.early_stopping_count == self.early_stopping_patience:
            print('early stopping, patience: {}, loading best epoch: {}.'.format(
                self.early_stopping_patience,
                self.best_epoch
            ))
            if self.load_best:
                self.load_model(path = path)
            return 1
        else:
            return 0
    
    def count_params(
        self,
        only_trainable : bool = True
    ) -> dict:
        """Counts the number of parameters of the model

        :param only_trainable: Whether to count only the trainable paramters or all, defaults to True
        :type only_trainable: bool, optional
        :return: dict containing the layers and their paramter number
        :rtype: dict
        """
        return {
            'total_params' : sum(p.numel() for p in self.parameters() if (p.requires_grad or not only_trainable)),
            'embeddings_params' : sum(p.numel() for p in self.embedding_layer.parameters() if (p.requires_grad or not only_trainable)),
            'rnn_params' : sum(p.numel() for p in self.rnn.parameters() if (p.requires_grad or not only_trainable)),
            'linear' : sum(p.numel() for p in self.linear.parameters() if (p.requires_grad or not only_trainable))
        }
    
    def freeze_embeddings(self):
        for p in self.embedding_layer.parameters():
            p.requires_grad = False

    def unfreeze_embeddings(self):
        for p in self.embedding_layer.parameters():
            p.requires_grad = True

    def regularizer(self) -> torch.Tensor:
        """Computes the regularizer on all the non bias and trainable parameters.
        $$\frac{1}{p} \gamma \sum_w w^p$$

        :return: The tensor with the backward regularization loss
        :rtype: torch.Tensor
        """        
        reg = torch.FloatTensor([0]).to(self.device)
        reg.requires_grad = True
        if self.gamma == 0:
            return reg
        else:
            for name, W in self.named_parameters():
                if W.requires_grad and 'bias' not in name:
                    reg = reg + torch.pow(W, self.q).sum()
            return 1/self.q * self.gamma * reg

    def fit(
        self, 
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        num_epochs : int = 30,
        fp16 : bool = False,
        regularizer : str = 'uniform',
        eval_epoch_0 : bool =  True,
        early_stopping = True,
        early_stopping_patience : int  = 3,
        early_stopping_metric : str = 'val_loss',
        early_stopping_metric_best : str = 'min', # if lower is better (like for loss),
        load_best : bool = True,
        model_path : str = 'models/'
    ) -> dict:
        """Trains the model with the train_dataloader and evaluates with the eval_dataloader.

        :param train_dataloader: The data loader to train with
        :type train_dataloader: torch.utils.data.DataLoader
        :param eval_dataloader: The data to evaluate with
        :type eval_dataloader: torch.utils.data.DataLoader
        :param num_epochs: The maxium number of epochs, defaults to 30
        :type num_epochs: int, optional
        :param fp16: Whether mixed precision is used for training, defaults to False
        :type fp16: bool, optional
        :param regularizer: The type of reguirizer to use, defaults to 'uniform'
        :type regularizer: str, optional
        :param eval_epoch_0: Whether to evaluate the dataloaders at epoch 0, defaults to True
        :type eval_epoch_0: bool, optional
        :param early_stopping: Whether to use early stopping or not, defaults to True
        :type early_stopping: bool, optional
        :param early_stopping_patience: How many epochs to wait before early stopping, defaults to 3
        :type early_stopping_patience: int, optional
        :param early_stopping_metric: Which metrics use as early stopping criterion, defaults to 'val_loss'
        :type early_stopping_metric: str, optional
        :param early_stopping_metric_best: Can be either 'min' or 'max', defaults to 'min'
        :type early_stopping_metric_best: str, optional
        :param load_best: Whether to load the model at the mest metric at the end, defaults to True
        :type load_best: bool, optional
        :param model_path: Where to save/load the model, defaults to 'models/'
        :type model_path: str, optional
        :return: Dictionary containing the train, eval losses and the lr at every epoch.
        :rtype: dict
        """
        make_dir_if_not_exists(model_path)
        self.model_path = os.path.join(model_path, self.model_name)
        self.regularizer_type = regularizer
        self.fp16 = fp16

        if early_stopping:
            self.early_stopping_patience = early_stopping_patience
            self.early_stopping_metric_best = early_stopping_metric_best
            self.early_stopping_count = 0
            self.best_epoch = 0
            self.best_metric = np.inf if early_stopping_metric_best == 'min' else -np.inf
            self.load_best = load_best
            
        metrics = {}

        for epoch in range(0, num_epochs+1):
            if epoch > 0:
                losses = self.epoch_step(train_dataloader)
                train_loss = np.mean(losses)
            elif eval_epoch_0:
                train_loss = self.evaluate(train_dataloader)
            else:
                continue
            metrics[epoch] = {'train_loss' : train_loss}
            eval_loss = self.evaluate(eval_dataloader)
            metrics[epoch]['val_loss'] = eval_loss
            metrics[epoch]['lr'] = self.scheduler.get_last_lr()[0]
            print(f"Train loss at epoch {epoch} : {train_loss}")
            print(f"Eval loss at epoch {epoch} : {eval_loss}")
            if early_stopping:
                current_metric = metrics[epoch][early_stopping_metric]
                if self.update_early_stopping(current_metric, epoch, path = self.model_path):
                    break
                    
        return metrics

    def generate(
        self,
        vocabulary : Vocabulary, 
        start_text : str, 
        num_words : int = 100, 
        random : bool = True
    ) -> str:
        """Given a starting text, generates a string deterministically or according to
        the language model distribution for num_words tokens

        :param vocabulary: The vocabulary of the model
        :type vocabulary: Vocabulary
        :param start_text: The text to start the generation with
        :type start_text: str
        :param num_words: The number of tokens to predict, defaults to 100
        :type num_words: int, optional
        :param random: Whether to use the distribution or the most probable next token, defaults to True
        :type random: bool, optional
        :return: The generated attached token sequence
        :rtype: str
        """    
        start_text = vocabulary.text_cleaner(start_text)
        tokens = [vocabulary.word_to_idx[w] if w in vocabulary.word_to_idx.keys() else 0 for w in start_text.split(' ')]
        self.eval()
        with torch.no_grad():
            s = torch.nn.Softmax(dim = 0)
            for i in range(num_words):
                hidden = self.init_hidden(1)
                x = torch.tensor([tokens]).to(self.device)
                logits, _ = self(x, hidden)
                logits = logits[0][-1]
                p = s(logits).cpu().numpy()
                if random:
                    word_index = np.random.choice(logits.shape[0], p = p)
                else:
                    word_index = np.argmax(p)
                tokens.append(word_index)

        res = ' '.join([vocabulary.idx_to_word[i] for i in tokens])
        return re.sub(r' \.', '.', res)

def init_model(
    vocabulary : Vocabulary,
    lr : float = None,
    **params
) -> NextWordPredictorModel:
    """Initializtes a NextWordPredictorModel with the given parameters. The lr needs
    to be given here since needed for fp16 initialization, as it is required for the
    optimizer.

    :param vocabulary: The used vocabulary
    :type vocabulary: Vocabulary
    :param lr: the model learning rate, defaults to None
    :type lr: float, optional
    :return: The initialized model
    :rtype: NextWordPredictorModel
    """
    def map_weights(weights, m_ = 0.01, M_ = 1):
        weights = 1 / weights
        M, m = max(weights), min(weights)
        return (np.array(weights) - m) * (M_ - m_) / (M - m) + m_

    device = params['device']
    fp16 = params.pop('fp16')
    opt = params.pop('opt')
    
    lr = params.pop('LEARNING_RATE')

    if params['weight'] and vocabulary is not None:
        params['weight'] = map_weights(np.array(list(vocabulary.vocab.values())))
    else:
        params['weight'] = None

    model = NextWordPredictorModel(**params).to(device)
    # need to setup the optimizer there because of the amp initialization
    if opt == 'ADAM':
        model.optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    else:
        model.optimizer = torch.optim.SGD(model.parameters(), lr = lr)


    if fp16:
        model, model.optimizer = amp.initialize(
            model,
            model.optimizer,
            opt_level = 'O1' # https://nvidia.github.io/apex/amp.html
        )
    model.fp16 = fp16

    model.scheduler = torch.optim.lr_scheduler.StepLR(
        model.optimizer,
        1.0, 
        gamma=0.97
    )
    logging.info('model initialized')
    return model


class Pipeline():
    def __init__(
        self,
        config_file : str,
        load_model_data : bool = False
    ):
        """Pipeline class that trains the language model for next word predictions

        :param config_file: Path to the configuration file for the model
        :type config_file: str
        :param load_model_data: Wheter to load the training data, defaults to False
        :type load_model_data: bool, optional
        """
        config_file = os.path.join('config_files', config_file)
        with open(config_file, 'r') as f:
            self.parameters = json.load(f)
        self.load_model_data = load_model_data

        torch.manual_seed(self.parameters['TORCH_SEED'])
        np.random.seed(self.parameters['NUMPY_SEED'])

        data_parameters = self.parameters['DATA_PARAMETERS']
        data_parameters['device'] = self.parameters['DEVICE']
        logging.info('preparing data...')
        self.init_data(**data_parameters)

        model_parameters = self.parameters['MODEL_PARAMETERS']
        model_parameters['device'] = self.parameters['DEVICE']
        model_parameters['vocab_size'] = self.vocabulary.get_vocab_size()
        self.model = init_model(self.vocabulary, **model_parameters)

    def init_data(self, **params):
        """
        Parameters
        ----------
        - **params : dict
            dictionary of arguments. Argument used are:
            - data_name : str
                The type of data used. By default is 'tweets'
            - data_folder : str
                Where the data is contained
            - vocab_from_scratch : bool
                Whether to load the vocabulary or to make it from scratch
            - max_voc_size : int
                The maximum voc size
            - min_word_occ : int
                Least number of times a word must occur to be in voc
            - vocab_file : str
                path to where to load or save the vocabulary (in pickle format)
            - min_seq_length
                minimum sequence length
            - max_seq_length
                maximum sequence length
            - device
                the device where the data is loaded

        :raises AssertionError: if data is not found
        """        
        vocab_path = os.path.join('vocabs', params['vocab_file'])
        make_dir_if_not_exists('./vocabs')

        if params['data_name'] == 'tweets':
            id_ = 2
            data_folder = params['data_folder']

            train_set_file = os.path.join(data_folder, f'train_{id_}.pickle')
            val_set_file = os.path.join(data_folder, f'val_{id_}.pickle')
            test_set_file = os.path.join(data_folder, f'test_{id_}.pickle')

            tokenizer = TweetTokenizer(
                preserve_case = False,
                strip_handles = True, #removes things like: @bob ...,
                reduce_len = True #more than 3 times same characters are limited waaaaayyyy -> waaayyy
            ).tokenize

            with open(train_set_file, 'rb') as f:
                train_set = pickle.load(f)
            train_set = train_set
            if params['vocab_from_scratch']:
                logging.info('generating vocabulary...')
                self.vocabulary = FromTweetsVocabulary(
                    max_voc_size = params['max_voc_size'],
                    min_word_occ = params['min_word_occ'],
                    tweets = train_set,
                    tokenizer = tokenizer,
                    text_cleaner = None
                )
                logging.info('vocabulary generated')
                with open(vocab_path, 'wb') as f:
                    pickle.dump(self.vocabulary, f)
            else:
                try:
                    with open(vocab_path, 'rb') as f:
                        self.vocabulary = pickle.load(f)
                    logging.info('vocabulary loaded')
                except FileNotFoundError:
                    logging.error("vocabulary file not found")
                    sys.exit(1)

        elif 'WikiText' in params['data_name']:
            data_folder = params['data_folder']
            if '103' in params['data_name']:
                id_ = '103'
                path = os.path.join('.', data_folder, 'wikitext-3')
            else:
                id_ = '2'
                path = os.path.join('.', data_folder, 'wikitext-2')

            train_set_file = os.path.join(path, f'train_{id_}.pickle')
            val_set_file = os.path.join(path, f'val_{id_}.pickle')
            test_set_file = os.path.join(path, f'test_{id_}.pickle')

            tokenizer = word_tokenize
            with open(train_set_file, 'rb') as f:
                train_set = pickle.load(f)
            if params['vocab_from_scratch']:
                logging.info('generating vocabulary...')
                self.vocabulary = FromRawTextVocabulary(
                    max_voc_size = params['max_voc_size'],
                    min_word_occ = params['min_word_occ'],
                    text_cleaner = text_cleaner_raw,
                    text = ' '.join(train_set),
                    tokenizer = tokenizer
                )
                logging.info('vocabulary generated')
                with open(vocab_path, 'wb') as f:
                    pickle.dump(self.vocabulary, f)
            else:
                try:
                    with open(vocab_path, 'rb') as f:
                        self.vocabulary = pickle.load(f)
                    logging.info('vocabulary loaded')
                except FileNotFoundError:
                    logging.error("vocabulary file not found")
                    sys.exit(1)
        else:
            raise AssertionError('data argument not allowed')
        
        if self.load_model_data:
            logging.info('creating train dataset...')
            self.train_dataset = SequenceDataset(
                vocabulary = self.vocabulary,
                text = train_set,
                min_seq_length = params['min_seq_length'],
                max_seq_length = params['max_seq_length'],
                device = params['device'],
                with_tqdm = True
            )
            logging.info('train dataset created')
            logging.info('creating validation dataset...')
            with open(val_set_file, 'rb') as f:
                val_set = pickle.load(f)
            self.val_dataset = SequenceDataset(
                vocabulary = self.vocabulary,
                text = val_set,
                min_seq_length = params['min_seq_length'],
                max_seq_length = params['max_seq_length'],
                device = params['device'],
                with_tqdm = True
            )
            logging.info('validation dataset created')
        logging.info('creating test dataset...')
        with open(test_set_file, 'rb') as f:
            test_set = pickle.load(f)
        self.test_dataset = SequenceDataset(
            vocabulary = self.vocabulary,
            text = test_set[:5000],
            min_seq_length = params['min_seq_length'],
            max_seq_length = params['max_seq_length'],
            device = params['device'],
            with_tqdm = True
        )
        logging.info('test dataset created')
        gc.collect()

    def train_model(self, name : str = 'test'):
        """
        Wrapper of the *.fit* method of the NextWordPredictorModel that first instanciate
        the train-val torch.utils.data.DataLoader and then trains the model and uses the
        parameters contained in the self.parameters attribute.

        :param name: name of the generated graphs, defaults to 'test'
        :type name: str, optional
        """        
        logging.info("""*************
        training model
        *************""")
        training_parameters = self.parameters['TRAINING_PARAMETERS']
        batch_size = training_parameters.pop("batch_size")
        model_name = training_parameters.pop("model_name")

        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size = batch_size,
            shuffle = True,
            pin_memory = False,
            drop_last = True
        )
        val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size = batch_size,
            shuffle = False,
            pin_memory = False,
            drop_last = True
        )
        
        self.model.model_name = model_name
        metrics = self.model.fit(
            **training_parameters,
            train_dataloader=train_dataloader,
            eval_dataloader=val_dataloader
        )
        logging.info('terminated training')
        df = pd.DataFrame(metrics).T
        plt.figure()
        plt.plot(df['train_loss'])
        plt.plot(df['val_loss'])
        plt.legend(['train_loss', 'val_loss'])
        plt.savefig(os.path.join('.','results',f'training_plot_{name}.svg'))

    def evaluate(self):
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size = 4,
            shuffle = False,
            pin_memory = False,
            drop_last = True
        )
        return self.model.evaluate(test_dataloader)

    def perplexity(self, dataset = None, **kwargs):
        if dataset is None:
            dataset = self.test_dataset
        dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size = 4,
            shuffle = False,
            pin_memory = False,
            drop_last = True
        )
        return self.model.perplexity(dataloader, **kwargs)

    def generate(self, start_text : str, vocabulary : Vocabulary = None, num_words = 100):
        return self.model.generate(
            start_text = start_text, 
            vocabulary = self.vocabulary if vocabulary is None else vocabulary, 
            num_words = num_words
        )

    def load_model(self, path = None):
        self.model.load_model(path = path)

if __name__ == '__main__':
    sys.path.append('..')
    if sys.argv[1] == 'tweet':
        json_file = os.path.join('.', 'config_files', 'CONFIG_MODEL_TWEETS.json')
        file = 'CONFIG_MODEL_TWEETS.json'
    elif sys.argv[1] == 'wiki':
        json_file = os.path.join('.', 'config_files', 'CONFIG_MODEL_WIKI.json')
        file = 'CONFIG_MODEL_WIKI.json'
    else:
        print('arg but be tweet or wiki')
        sys.exit(0)

    res_file = os.path.join('.', 'results', f'model_results_{sys.argv[1]}.csv')

    if sys.argv[2] == 'batch':
        if not os.path.exists(res_file):    
            df = pd.DataFrame(columns = [
                'type_of_rnn',
                'emb_dim',
                'lr',
                'perplexity',
                'test_loss',
                'exec_time',
                'train_size',
                'val_size',
                'test_size',
                'batch_size'
            ])
            df.to_csv(res_file)
        i = 0
        for bs in [4,8,16]:
            df = pd.read_csv(res_file, index_col=0)
            if i>= len(df):
                update_json(
                    json_file = json_file,
                    TRAINING_PARAMETERS = {
                        'batch_size' : bs
                    }
                )
                pipeline = Pipeline(file, load_model_data = True)
                start_time = time.time()
                pipeline.train_model(name = 'GRU' + '_' + str(256) + '_' + str(1e-3) + '_' + str(bs))
                train_time = int(time.time() - start_time)
                perplexity, test_loss = pipeline.perplexity()
                df.loc[len(df)] = [
                    'GRU',
                    256,
                    1e-3,
                    perplexity,
                    test_loss,
                    train_time,
                    len(pipeline.train_dataset),
                    len(pipeline.val_dataset),
                    len(pipeline.test_dataset),
                    bs
                ]
                df.to_csv(res_file)
            i+=1
    
    elif sys.argv[2] == 'emb_lr':
        if not os.path.exists(res_file):    
            df = pd.DataFrame(columns = [
                'type_of_rnn',
                'emb_dim',
                'lr',
                'perplexity',
                'test_loss',
                'exec_time',
                'train_size',
                'val_size',
                'test_size'
            ])
            df.to_csv(res_file)
        i = 0
        for type_of_rnn in ['GRU', 'LSTM']:
            for emb_dim in [64,128,256]:
                for lr in [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]:
                    df = pd.read_csv(res_file, index_col=0)
                    if i >= len(df):
                        update_json(
                            json_file = json_file,
                            MODEL_PARAMETERS = {
                                'type_of_rnn' : type_of_rnn,
                                'emb_dim' : emb_dim,
                                'LEARNING_RATE' : lr,
                                'fp16' : 1
                            },
                            TRAINING_PARAMETERS = {
                                'fp16' : 1
                            }
                        )
                        
                        pipeline = Pipeline(file, load_model_data = True)
                        start_time = time.time()
                        pipeline.train_model(name = type_of_rnn + '_' + str(emb_dim) + '_' + str(lr))
                        train_time = int(time.time() - start_time)
                        perplexity, test_loss = pipeline.perplexity()
                        df.loc[len(df)] = [
                            type_of_rnn,
                            emb_dim,
                            lr,
                            perplexity,
                            test_loss,
                            train_time,
                            len(pipeline.train_dataset),
                            len(pipeline.val_dataset),
                            len(pipeline.test_dataset)
                        ]
                        df.to_csv(res_file)
                    i+=1

