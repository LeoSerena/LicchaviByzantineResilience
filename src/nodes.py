from abc import abstractclassmethod
import os
import re
import sys
import pickle

import torch

sys.path.append('.')
from src.data_processing import  SequenceDataset
from src.utils import split_data
from src.models import init_model

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
        datafolder,
        vocabulary,
        min_seq_length,
        max_seq_length,
        device,
        **kwargs
    ):
        super(UserNode, self).__init__(**kwargs)
        for file in os.listdir(datafolder):
            if re.match(r'node_'+str(kwargs['id_'])+r'_.*\.pickle', file):
                self.file = file

        self.num_bodies = int(re.sub('\.pickle', '', self.file.split('_')[2]))

        with open(os.path.join(datafolder, self.file), 'rb') as f:
            data = pickle.load(f)

        train_set, val_set, test_set = split_data(data[:1000])

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
    def __init__(self, **kwargs):
        if self.__class__ ==  ByzantineNode:
            raise NotImplementedError("""This is an abstract class""")
        super(ByzantineNode, self).__init__(**kwargs)

class DataPoisoningNode(ByzantineNode):
    def __init__(
        self,
        vocabulary,
        min_seq_length,
        max_seq_length,
        device,
        N,
        **kwargs
    ):
        if self.__class__ ==  DataPoisoningNode:
            raise NotImplementedError("""This is an abstract class""")
        super(DataPoisoningNode, self).__init__(**kwargs)
        self.vocabulary = vocabulary
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.device = device
        self.N = N

def init_forged_grad(general_model_state_dict):
    forged_grad = general_model_state_dict.copy()
    for key in forged_grad:
        forged_grad[key] = torch.zeros_like(forged_grad[key])
    return forged_grad
    

def compute_forged_grad(prev_model_state_dict, model_state_dict, prev_lr, lr, prev_forged_grad, target, clamp = 1):
    forged_grad = prev_forged_grad.copy()
    for key in forged_grad:
        if 'embedding' not in key:
            forged_grad[key] = forged_grad[key] + (model_state_dict[key] - target[key]) / lr 
            forged_grad[key] = forged_grad[key] - (prev_model_state_dict[key] - model_state_dict[key]) / prev_lr
            forged_grad[key] = torch.clamp(forged_grad[key], min = -clamp, max = clamp)
    return forged_grad

def forge_model(target, forged_grad):
    forged_model = forged_grad.copy()
    for key in forged_model:
        forged_model[key] = target[key] - forged_grad[key] / 2
    return forged_model
        

class ForgingModelNode(ByzantineNode):
    def __init__(
        self,
        attack_model_path,
        N,
        **kwargs
    ):
        if self.__class__ ==  ForgingModelNode:
            raise NotImplementedError("""This is an abstract class""")
        kwargs.pop('vocabulary')
        kwargs.pop('min_seq_length')
        kwargs.pop('max_seq_length')
        kwargs.pop('device')
        super(ForgingModelNode, self).__init__(**kwargs)
        self.attack_model_path = attack_model_path
        self.data = [0]*N

    def return_model(self):
        raise NotImplementedError

class NormalDataPoisoningNode(DataPoisoningNode):
    def __init__(
        self,
        sentence,
        **kwargs
    ):
        super(NormalDataPoisoningNode, self).__init__(**kwargs)
        self.data = SequenceDataset(
            vocabulary = self.vocabulary,
            text = (sentence * self.N)[1:], #[1:] is to avoid the first space
            min_seq_length = self.min_seq_length,
            max_seq_length = self.max_seq_length,
            device = self.device,
            with_tqdm=False
        )

class NormalModelForgingNode(ForgingModelNode):
    def __init__(
        self,
        **kwargs
    ):
        super(NormalModelForgingNode, self).__init__(**kwargs)

    def return_model(self):
        return torch.load(self.attack_model_path)

class StrategicModelForgingNode(ForgingModelNode):
    def __init__(
        self,
        lr,
        **kwargs
    ):
        super(StrategicModelForgingNode, self).__init__(**kwargs)
        self.lr = lr

    def compute_forged_model(self, general_model_state_dict):
        attack_model_state_dict = torch.load(self.attack_model_path)
        return forge_model(
            attack_model_state_dict,
            general_model_state_dict,
            lambda_ = self.lambda_,
            lr = self.lr
        )

class StrategicDataPoisoningNode():
    def __init__(
        self,
        general_model,
        N,
        lr = 1,
        **kwargs
    ):
        super(NormalModelForgingNode, self).__init__(**kwargs)
        self.lr = lr
        self.general_model = general_model
        self.N = N

    def forge_data(self):
        model_state_dict = forge_model(
            self.attack_model,
            self.general_model,
            lambda_ = self.lambda_,
            lr = self.lr
        )

        model = init_model().load_state_dict(model_state_dict)

        model.generate(self.N)

