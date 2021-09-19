import os
import math

from tqdm import tqdm
import numpy as np

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
        """
        Class of a nn.Module that add a positional encoding to the embedding
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
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + torch.transpose(self.pe[:x.size(1)], 0,1)
        return self.dropout(x)

class NextWordPredictorModel(torch.nn.Module):
    def __init__(
        self,
        type_of_rnn : str,
        emb_dim : int,
        vocab_size : int,
        num_lstm_hidden_layers : int,
        hidden_state_size : int,
        dropout : float,
        device : str,
        fp16 : bool = False,
        lr : float = 1e-3,
        weight : list = None,
        positional_encoding : bool = True
    ):
        super().__init__()
        self.lr = lr
        self.emb_dim = emb_dim
        self.num_lstm_hidden_layers = num_lstm_hidden_layers
        self.hidden_state_size = hidden_state_size
        self.device = device
        self.fp16 = fp16
        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.type_of_rnn = type_of_rnn
        
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
            'num_layers' : num_lstm_hidden_layers,
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
            print('Default torch.n.RNN used')
            self.rnn = torch.nn.RNN(**inputs).to(device)

        self.linear = torch.nn.Linear(
            hidden_state_size,
            self.vocab_size
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        self.criterion = torch.nn.CrossEntropyLoss(
            weight = torch.FloatTensor([weight]).to(self.device) if weight is not None else None,
            ignore_index = 0,
            reduction = 'mean'
        ).to(device) # may use the weight as prior n_occ / num_words
        
        
        self.init_weights()

        
    def forward(self, inputs, hidden):
        embeddings = self.embedding_layer(inputs)
        if self.positional_encoding:
            embeddings = self.positional_encoder(embeddings)
        output, hidden = self.rnn(embeddings, hidden)
        output = self.linear(output)
        return output, hidden
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding_layer.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
    
    def save_model(self, path = None):
        """
        Saves the model in the given path. If no path is given, automatically saved
        in the model_path specified at training.
        """
        if path is not None:
            torch.save(self.state_dict(), path)
        else:
            if hasattr(self, 'model_path'):
                torch.save(self.state_dict(), os.path.join(self.model_path, 'model.pth'))
            else:
                print('No path given, please enter a path to save the model.')
    
    def load_model(self, path = None):
        """
        Loads the model from the given path. If no path is given, automatically loaded
        from the model_path specified at training.
        """
        if path is not None:
            self.load_state_dict(torch.load(path), strict = False)
        else:
            if hasattr(self, 'model_path'):
                self.load_state_dict(torch.load(os.path.join(self.model_path, 'model.pth')))
            else:
                print('No path given, please enter a path to load the model.')
    
    def init_hidden(self, batch_size):
        if self.has_cell_state:
            return (
                torch.zeros(
                    self.num_lstm_hidden_layers, batch_size, self.hidden_state_size
                ).to(self.device).detach(),
                torch.zeros(
                    self.num_lstm_hidden_layers, batch_size, self.hidden_state_size
                ).to(self.device).detach()
            )
        else:
            return torch.zeros(
                self.num_lstm_hidden_layers, batch_size, self.hidden_state_size
            ).to(self.device).detach()
    
    def evaluate(self, eval_dataloader):
        self.eval()
        losses = []
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                hidden = self.init_hidden(eval_dataloader.batch_size)
                outputs, _ = self.forward(batch[:,:-1], hidden)
                outputs = torch.transpose(outputs, 1,2)
                labels = batch[:,1:]
                
                loss = self.criterion(outputs, labels) + self.regularizer()
                
                losses.append(loss.item())
                
        return np.mean(losses)
        
    def epoch_step(self, data_loader):
        self.train()
        losses = []
        
        for batch in tqdm(data_loader):
            for param in self.parameters():
                param.grad = None
            hidden = self.init_hidden(data_loader.batch_size)
            outputs, _ = self.forward(batch[:,:-1], hidden)
            outputs = torch.transpose(outputs, 1, 2)
            labels = batch[:,1:]
            
            loss = self.criterion(outputs, labels)
            loss = loss + self.regularizer()
            
            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
                
            losses.append(loss.item())
        
        self.scheduler.step()
        
        return losses
    
    def update_early_stopping(self, current_metric, epoch, path = None):
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
    
    def count_params(self, only_trainable = True):
        return {
            'total_params' : sum(p.numel() for p in self.parameters() if (p.requires_grad or not only_trainable)),
            'embeddings_params' : sum(p.numel() for p in self.embedding_layer.parameters() if (p.requires_grad or not only_trainable)),
            'rnn_params' : sum(p.numel() for p in self.rnn.parameters() if (p.requires_grad or not only_trainable)),
            'linear' : sum(p.numel() for p in self.linear.parameters() if (p.requires_grad or not only_trainable))
        }
    
    def freeze_embeddings(self):
        for p in self.embedding_layer.parameters():
            p.requires_grad = False

    def unfreeze_embeddigns(self):
        for p in self.embedding_layer.parameters():
            p.requires_grad = True

    def regularizer(self):
        reg = torch.FloatTensor([0]).to(self.device)
        reg.requires_grad = True
        if self.regularizer_type == 'uniform':
            for W in self.parameters():
                if W.requires_grad:
                    reg = reg + torch.pow(W, self.p).sum()
        elif self.regularizer_type == 'progressive':
            pass
        return 1/self.p * self.lambda_ * reg


    def fit(
        self, 
        train_dataloader,
        eval_dataloader,
        num_epochs = 30,
        regularizer = 'uniform',
        p = 2,
        eval_epoch_0 : bool =  True,
        early_stopping = True,
        early_stopping_patience = 3,
        early_stopping_metric = 'val_loss',
        early_stopping_metric_best = 'min', # if lower is better (like for loss),
        load_best = True,
        model_path = 'models/'
    ):
        self.model_path = model_path
        self.regularizer_type = regularizer
        self.p = p
        self.lambda_ = 1e-4
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
                if self.update_early_stopping(current_metric, epoch):
                    break
                    
        return metrics