from abc import abstractclassmethod
from collections import OrderedDict
import os
import re
import sys
import pickle


import torch

sys.path.append('.')
from src.data_processing import  SequenceDataset, Vocabulary
from src.utils import split_data
from src.models import init_model

class Node():
    def __init__(
        self,
        id_ : int,
        lambda_ : float,
        p : int
    ):
        """Abstract class for nodes

        :param id_: The unique node identifier
        :type id_: int
        :param lambda_: $\lambda_n$ of the node, only for LICCHAVI
        :type lambda_: float
        :param p: $p$ norm for central model distance
        :type p: int
        :raises NotImplementedError: Raises exception if instantiated
        """
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
        datafolder : str,
        vocabulary : Vocabulary,
        min_seq_length : int,
        max_seq_length : int,
        device : str,
        **kwargs
    ):
        """[summary]

        :param datafolder: pickle file path to node data
        :type datafolder: str
        :param vocabulary: vocabulary for tokenization
        :type vocabulary: Vocabulary
        :param min_seq_length: min sequence length for data
        :type min_seq_length: int
        :param max_seq_length: max senquence length for data
        :type max_seq_length: int
        :param device: name of device to put data on
        :type device: str
        """    
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
        vocabulary : Vocabulary,
        min_seq_length : int,
        max_seq_length : int,
        device : str,
        N : int,
        **kwargs
    ):
        """Abstract class of node simulating a data poisoning attack. 
        Given a sentence, will generate a SequenceDataset inly consisting of the sentence.

        :param vocabulary: vocabulary for tokenization
        :type vocabulary: Vocabulary
        :param min_seq_length: minimum sequence length
        :type min_seq_length: int
        :param max_seq_length: maximum sequence length
        :type max_seq_length: int
        :param device: device to put data on
        :type device: str
        :param N: length of the SequenceDataset
        :type N: int
        :raises NotImplementedError: Raises an exception if instantiated
        """    
        if self.__class__ ==  DataPoisoningNode:
            raise NotImplementedError("""This is an abstract class""")
        super(DataPoisoningNode, self).__init__(**kwargs)
        self.vocabulary = vocabulary
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.device = device
        self.N = N

def init_forged_grad(general_model_state_dict : OrderedDict):
    """Initializes a gradient of the shape of the given model state dict
    with only 0's

    :param general_model_state_dict: torch state dict of a model
    :type general_model_state_dict: [type]
    :return: 0 gradient
    :rtype: OrderedDict
    """
    forged_grad = general_model_state_dict.copy()
    for key in forged_grad:
        forged_grad[key] = torch.zeros_like(forged_grad[key])
    return forged_grad
    

def compute_forged_grad(
    prev_model_state_dict : OrderedDict, 
    model_state_dict : OrderedDict, 
    prev_lr : float,
    lr : float, 
    prev_forged_grad : OrderedDict,
    target : OrderedDict,
    clamp : float= 1
) -> OrderedDict:
    """computes the forged gradient according to the paper:
    $$
    g_B^{(t+1)} = g_B^{(t)} + \frac{\theta_{t+1} - \theta_B}{\gamma_{t+1}} - \frac{\theta_{t} - \theta_{t+1}}{\gamma_{t}}
    $$

    :param prev_model_state_dict: $\theta_{t}$
    :type prev_model_state_dict: OrderedDict
    :param model_state_dict: $\theta_{t+1}$:
    :type model_state_dict: OrderedDict
    :param prev_lr: \gamma_t
    :type prev_lr: float
    :param lr: \gamma_{t+1}
    :type lr: float
    :param prev_forged_grad: g_B^{(t)}
    :type prev_forged_grad: OrderedDict
    :param target: \theta_B
    :type target: OrderedDict
    :param clamp: clamping of the gradient, defaults to 1
    :type clamp: float, optional
    :return: $g_B^{(t+1)}, the forged gradient
    :rtype: OrderedDict
    """
    forged_grad = prev_forged_grad.copy()
    for key in forged_grad:
        if 'embedding' not in key:
            forged_grad[key] = forged_grad[key] + (model_state_dict[key] - target[key]) / lr 
            forged_grad[key] = forged_grad[key] - (prev_model_state_dict[key] - model_state_dict[key]) / prev_lr
            forged_grad[key] = torch.clamp(forged_grad[key], min = -clamp, max = clamp)
    return forged_grad

def forge_model(
    target : OrderedDict, 
    forged_grad : OrderedDict
) -> OrderedDict:
    """Given a forged gradient, computes the according forged model:

    $$
    \theta^{t+1} = \theta_B - \frac{1}{2} g_B^{(t+1)}
    $$

    :param target: $\theta_B$, the attack model
    :type target: OrderedDict
    :param forged_grad: $g_B$, the forged gradient
    :type forged_grad: OrderedDict
    :return: The computed attack model
    :rtype: OrderedDict
    """
    forged_model = forged_grad.copy()
    for key in forged_model:
        forged_model[key] = target[key] - forged_grad[key] / 2
    return forged_model
        

class ForgingModelNode(ByzantineNode):
    def __init__(
        self,
        attack_model_path : str,
        N : int,
        **kwargs
    ):
        """Abstract class of forging model node.

        :param attack_model_path: path to attack model
        :type attack_model_path: str
        :param N: data size (for FedAVG)
        :type N: int
        :raises NotImplementedError: Raises an exception when instantiated
        """        
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
        sentence : str,
        **kwargs
    ):
        """Data Poisoning node that generates a dataset with only the given sentence.

        :param sentence: The sentence to build the dataset with.
        :type sentence: str
        """
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
        """Class tat returns the attack model
        """
        super(NormalModelForgingNode, self).__init__(**kwargs)

    def return_model(self):
        """Returns the attack model

        :return: Attack model state_dict()
        :rtype: OrderedDict
        """
        return torch.load(self.attack_model_path)

class StrategicModelForgingNode(ForgingModelNode):
    def __init__(
        self,
        lr : float,
        **kwargs
    ):
        """Computes the forged gradient and the corresponding attack model

        :param lr: The learning rate
        :type lr: float
        """
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

