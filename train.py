import logging
import math
import os
import sys

import torch
from dataclasses import dataclass, field
import wandb
from collections import Counter

from scipy.special import softmax
from sklearn.metrics import roc_auc_score
import numpy as np

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
    EvalPrediction
)
from transformers.trainer_utils import get_last_checkpoint

import datasets
from datasets import load_dataset, load_metric
from typing import Optional, List, Dict, Any, Tuple

from utils.tokenization import MyTokenizer
from utils.configuration_utils import KGTREATConfig
from model.modeling_kgtreat import KGTREATForSequenceClassification, KGTREATForMaskedLMLP

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    vocab_file: Optional[str] = field(
        default=None, metadata={"help": "The vocabulary file (a text file)"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    mask_prediction: bool = field(
        default=False,
        metadata={
            "help": "Whether to do mask prediction"
        },
    )

    # customized parameters
    outcome_prediction: bool = field(
        default=False,
        metadata={
            "help": "Whether to do outcome prediction"
        },
    )

    use_kg: bool = field(
        default=False,
        metadata={
            "help": "Whether to use knowledge graph"
        },
    )

    sep_graph: bool = field(
        default=False,
        metadata={
            "help": "Whether to use two separated graphs for treatment-covariate and outcome-covariate"
        },
    )

    link_prediction: bool = field(
        default=False,
        metadata={
            "help": "Whether to do link prediction"
        },
    )

    time_embedding: bool = field(
        default=True,
        metadata={
            "help": "Whether to use time_embedding"
        },
    )

    residual_ie: int = field(
        default=2,
        metadata={
            "help": "Whether to use residual MInt."
        },
    )

    learning_rate_gnn: float = field(
        default=1e-3, metadata={"help": "learning rate for GNN module"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: List[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."}
    )

    eval_data_file: List[str] = field(
        default=None,
        metadata={"help": "The input eval data file (a text file)."}
    )

    data_path: str = field(
        default=None,
        metadata={"help": "The input training data path (directory)."}
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    # customized parameters
    compared_drug: str = field(
        default=None,
        metadata={
            "help": "compared_drug"
        },
    )

    target_drug: str = field(
        default=None,
        metadata={
            "help": "target_drug"
        },
    )
    baseline_window: int = field(
        default=90,
        metadata={
            "help": "baseline_windowe"
        },
    )
    fix_window_length: int = field(
        default=30,
        metadata={
            "help": "fix_window_length"
        },
    )

    training_set_fraction: float = field(
        default=1,
        metadata={
            "help": "training_set_fraction"
        },
    )

    max_node_num: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input KG node length after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )

    ent_emb_paths: Optional[str] = field(
        default=None,
        metadata={
            "help": "pretrained entity embeddings."
        },
    )

    n_runs: int = field(
        default=0,
        metadata={
            "help": "n_runs"
        },
    )

    wandb_mode: Optional[str] = field(
        default='offline',
        metadata={
            "help": "wandb mode: online, offline."
        },
    )

    clean_out: str = field(
        default=None,
        metadata={
            "help": "clean_out"
        },
    )


@dataclass
class myDataCollator:
    mask_prediction: bool = False
    outcome_prediction: bool = False
    use_kg: bool = False
    link_prediction: bool = False
    mlm_probability: float = 0.15
    tokenizer: MyTokenizer = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = []
        attention_mask = []
        token_type_ids = []
        visit_time_ids = []
        physical_time_ids = []

        for b in batch:
            input_ids.append(b['input_ids'])
            attention_mask.append(b['attention_mask'])
            token_type_ids.append(b['token_type_ids'])
            visit_time_ids.append(b['visit_time_ids'])
            physical_time_ids.append(b['physical_time_ids'])

        input_ids = torch.tensor(input_ids,dtype=torch.long)
        attention_mask = torch.tensor(attention_mask,dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids,dtype=torch.long)
        visit_time_ids = torch.tensor(visit_time_ids, dtype=torch.long)
        physical_time_ids = torch.tensor(physical_time_ids, dtype=torch.long)

        batch_out = {
            "input_ids": input_ids,
             "attention_mask": attention_mask,
             "token_type_ids": token_type_ids,
             "visit_time_ids": visit_time_ids,
             "physical_time_ids": physical_time_ids,
             }

        if self.use_kg:
            concept_ids = []
            node_type_ids = []
            adj_lengths = []
            special_nodes_mask = []
            edge_index = []
            edge_type = []

            for b in batch:
                # construct gnn input
                concept_ids.append(b['concept_ids'])
                node_type_ids.append(b['node_type_ids'])
                adj_lengths.append(b['adj_lengths'])
                special_nodes_mask.append(b['special_nodes_mask'])
                edge_index.append(torch.tensor(b['edge_index']))
                edge_type.append(torch.tensor(b['edge_type']))

            concept_ids = torch.tensor(concept_ids, dtype=torch.long)
            node_type_ids = torch.tensor(node_type_ids, dtype=torch.long)
            adj_lengths = torch.tensor(adj_lengths, dtype=torch.long)
            special_nodes_mask = torch.tensor(special_nodes_mask, dtype=torch.long)

            batch_out['concept_ids'] = concept_ids
            batch_out['node_type_ids'] = node_type_ids
            batch_out['adj_lengths'] = adj_lengths
            batch_out['special_nodes_mask'] = special_nodes_mask
            batch_out['edge_index'] = edge_index
            batch_out['edge_type'] = edge_type

            if self.link_prediction:
                edge_index, edge_type, pos_triples, neg_nodes = self.process_graph_data(edge_index, edge_type,
                                                                                        node_type_ids)
                batch_out['edge_index'] = edge_index
                batch_out['edge_type'] = edge_type
                batch_out['pos_triples'] = pos_triples
                batch_out['neg_nodes'] = neg_nodes

        if self.mask_prediction:
            input_ids, mask_labels, token_type_ids = self.torch_mask_tokens(input_ids,token_type_ids)
            batch_out['input_ids'] =  input_ids
            batch_out['mask_labels'] = mask_labels
            batch_out['token_type_ids'] = token_type_ids

        if self.outcome_prediction:
            outcome_labels = [b['outcome'] for b in batch]
            outcome_labels = torch.tensor(outcome_labels, dtype=torch.long)

            treatment_labels = [b['treatment_label'] for b in batch]
            treatment_labels = torch.tensor(treatment_labels, dtype=torch.long)

            # input_ids_cf = [b['input_ids_cf'] for b in batch]
            # input_ids_cf = torch.tensor(input_ids_cf, dtype=torch.long)

            batch_out['outcome_labels'] = outcome_labels
            batch_out['treatment_labels'] = treatment_labels
            # batch_out['input_ids_cf'] = input_ids_cf

        return batch_out

    def torch_mask_tokens(self, inputs: Any, token_type_ids: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.vocab.get(self.tokenizer.mask_token)
        #
        # # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        #
        # # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        # indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        # mask_token_id = self.tokenizer.vocab.get(self.tokenizer.mask_token)
        # inputs[indices_replaced] = mask_token_id
        # token_type_ids[indices_replaced] = self.tokenizer.convert_token_ids_to_token_type_ids(mask_token_id)

        return inputs, labels, token_type_ids

    def process_graph_data(self, edge_index, edge_type, node_type_ids):
        # edge_index: nested list of shape (n_samples, ), where each entry is tensor[2, E]
        # edge_type:  nested list of shape (n_samples, ), where each entry is tensor[E, ]
        # node_type_ids: tensor[n_samples, num_nodes]
        bs = len(edge_index)
        input_edge_index, input_edge_type, pos_triples, neg_nodes = [], [], [], []
        non_zero_edge = []
        for bid in range(bs):
            _edge_index = edge_index[bid]  # .clone()
            _edge_type = edge_type[bid]  # .clone()
            _node_type_ids = node_type_ids[bid]  # .clone()
            _edge_index, _edge_type, _pos_triples, _neg_nodes = self._process_one_graph(_edge_index, _edge_type,
                                                                                        _node_type_ids)
            non_zero_edge.append(1)
            input_edge_index.append(_edge_index)
            input_edge_type.append(_edge_type)
            pos_triples.append(_pos_triples)
            neg_nodes.append(_neg_nodes)
        input_edge_index = list(
            map(list, zip(*(iter(input_edge_index),))))  # nested list of shape (n_samples, )
        input_edge_type = list(map(list, zip(*(iter(input_edge_type),))))
        pos_triples = list(map(list, zip(*(iter(pos_triples),))))
        neg_nodes = list(map(list, zip(*(iter(neg_nodes),))))

        return input_edge_index, input_edge_type, pos_triples, neg_nodes

    def _process_one_graph(self, _edge_index, _edge_type, _node_type_ids):
        # _edge_index: tensor[2, E]
        # _edge_type:  tensor[E, ]
        # _node_type_ids: tensor[n_nodes, ]
        E = len(_edge_type)
        if E == 0:
            # print ('KG with 0 node', file=sys.stderr)
            effective_num_nodes = 1
        else:
            effective_num_nodes = int(_edge_index.max()) + 1
        device = _edge_type.device

        tmp = _node_type_ids.max().item()
        assert isinstance(tmp, int) and 0 <= tmp <= 5
        try:
            _edge_index_node_type = _node_type_ids[_edge_index]  # [2, E]
        except:
            print(_edge_index)
            print(_edge_index.shape)
            print(_edge_index.dtype)
        _is_special = (_edge_index_node_type == 3)  # [2, E]
        is_special = _is_special[0] | _is_special[1]  # [E,]

        positions = torch.arange(E)
        positions = positions[~is_special]  # [some_E, ]
        drop_count = min(100, int(len(positions) * 0.15))
        if len(positions) > 0 and drop_count > 0:
            drop_idxs = torch.multinomial(torch.full((len(positions),), 1.0), drop_count,
                                          replacement=False)  # [drop_count, ]
        else:
            drop_idxs = torch.tensor([]).long()
        drop_positions = positions[drop_idxs]  # [drop_count, ]

        mask = torch.zeros((E,)).long()  # [E, ]
        mask = mask.index_fill_(dim=0, index=drop_positions, value=1).bool().to(device)  # [E, ]

        real_drop_count = int(drop_count * (1 - 0.1))
        real_drop_positions = positions[drop_idxs[:real_drop_count]]  # [real_drop_count, ]
        real_mask = torch.zeros((E,)).long()  # [E, ]
        real_mask = real_mask.index_fill_(dim=0, index=real_drop_positions, value=1).bool().to(device)  # [E, ]

        assert int(mask.long().sum()) == drop_count
        # print (f'drop_E / total_E = {drop_count} / {E} = {drop_count / E}', ) #E is typically 1000-3000
        input_edge_index = _edge_index[:, ~real_mask]
        input_edge_type = _edge_type[~real_mask]
        assert input_edge_index.size(1) == E - real_drop_count

        pos_edge_index = _edge_index[:, mask]
        pos_edge_type = _edge_type[mask]
        pos_triples = [pos_edge_index[0], pos_edge_type, pos_edge_index[1]]
        # pos_triples: list[h, r, t], where each of h, r, t is [n_triple, ]
        assert pos_edge_index.size(1) == drop_count

        num_edges = len(pos_edge_type)
        num_corruption = 64
        neg_nodes = torch.randint(0, effective_num_nodes, (num_edges, num_corruption),
                                  device=device)  # [n_triple, n_neg]
        return input_edge_index, input_edge_type, pos_triples, neg_nodes

@dataclass
class myDataCollatorForSepGraphs:
    mask_prediction: bool = False
    outcome_prediction: bool = False
    use_kg: bool = False
    link_prediction: bool = False
    mlm_probability: float = 0.15
    tokenizer: MyTokenizer = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = []
        attention_mask = []
        token_type_ids = []
        visit_time_ids = []
        physical_time_ids = []
        sep_graph_labels = []

        num_pair = len(batch[0]['input_ids'])
        for b in batch:
            for i in range(num_pair):
                input_ids.append(b['input_ids'][i])
                attention_mask.append(b['attention_mask'][i])
                token_type_ids.append(b['token_type_ids'][i])
                visit_time_ids.append(b['visit_time_ids'][i])
                physical_time_ids.append(b['physical_time_ids'][i])
                sep_graph_labels.append(b['sep_graph_label'][i])

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        visit_time_ids = torch.tensor(visit_time_ids, dtype=torch.long)
        physical_time_ids = torch.tensor(physical_time_ids, dtype=torch.long)
        sep_graph_labels = torch.tensor(sep_graph_labels, dtype=torch.long)

        batch_out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "visit_time_ids": visit_time_ids,
            "physical_time_ids": physical_time_ids,
            "sep_graph_labels": sep_graph_labels,
        }

        if self.use_kg:
            concept_ids = []
            node_type_ids = []
            adj_lengths = []
            special_nodes_mask = []
            edge_index = []
            edge_type = []

            for b in batch:
                # construct gnn input
                for i in range(num_pair):
                    concept_ids.append(b['concept_ids'][i])
                    node_type_ids.append(b['node_type_ids'][i])
                    adj_lengths.append(b['adj_lengths'][i])
                    special_nodes_mask.append(b['special_nodes_mask'][i])
                    edge_index.append(torch.tensor(b['edge_index'][i]))
                    edge_type.append(torch.tensor(b['edge_type'][i]))

            concept_ids = torch.tensor(concept_ids, dtype=torch.long)
            node_type_ids = torch.tensor(node_type_ids, dtype=torch.long)
            adj_lengths = torch.tensor(adj_lengths, dtype=torch.long)
            special_nodes_mask = torch.tensor(special_nodes_mask, dtype=torch.long)

            batch_out['concept_ids'] = concept_ids
            batch_out['node_type_ids'] = node_type_ids
            batch_out['adj_lengths'] = adj_lengths
            batch_out['special_nodes_mask'] = special_nodes_mask
            batch_out['edge_index'] = edge_index
            batch_out['edge_type'] = edge_type

            if self.link_prediction:
                edge_index, edge_type, pos_triples, neg_nodes = self.process_graph_data(edge_index, edge_type,
                                                                                        node_type_ids)
                batch_out['edge_index'] = edge_index
                batch_out['edge_type'] = edge_type
                batch_out['pos_triples'] = pos_triples
                batch_out['neg_nodes'] = neg_nodes

        if self.mask_prediction:
            input_ids, mask_labels, token_type_ids = self.torch_mask_tokens(input_ids, token_type_ids)
            batch_out['input_ids'] = input_ids
            batch_out['mask_labels'] = mask_labels
            batch_out['token_type_ids'] = token_type_ids

        if self.outcome_prediction:
            outcome_labels = []
            treatment_labels = []
            for b in batch:
                for i in range(num_pair):
                    outcome_labels.append(b['outcome'][i])
                    treatment_labels.append(b['treatment_label'][i])
            outcome_labels = torch.tensor(outcome_labels, dtype=torch.long)
            treatment_labels = torch.tensor(treatment_labels, dtype=torch.long)

            batch_out['outcome_labels'] = outcome_labels
            batch_out['treatment_labels'] = treatment_labels
            # batch_out['input_ids_cf'] = input_ids_cf

        return batch_out

    def torch_mask_tokens(self, inputs: Any, token_type_ids: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[
        Any, Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.vocab.get(self.tokenizer.mask_token)
        #
        # # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        #
        # # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        # indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        # mask_token_id = self.tokenizer.vocab.get(self.tokenizer.mask_token)
        # inputs[indices_replaced] = mask_token_id
        # token_type_ids[indices_replaced] = self.tokenizer.convert_token_ids_to_token_type_ids(mask_token_id)

        return inputs, labels, token_type_ids

    def process_graph_data(self, edge_index, edge_type, node_type_ids):
        # edge_index: nested list of shape (n_samples, ), where each entry is tensor[2, E]
        # edge_type:  nested list of shape (n_samples, ), where each entry is tensor[E, ]
        # node_type_ids: tensor[n_samples, num_nodes]
        bs = len(edge_index)
        input_edge_index, input_edge_type, pos_triples, neg_nodes = [], [], [], []
        non_zero_edge = []
        for bid in range(bs):
            _edge_index = edge_index[bid]  # .clone()
            _edge_type = edge_type[bid]  # .clone()
            _node_type_ids = node_type_ids[bid]  # .clone()
            _edge_index, _edge_type, _pos_triples, _neg_nodes = self._process_one_graph(_edge_index, _edge_type,
                                                                                        _node_type_ids)
            non_zero_edge.append(1)
            input_edge_index.append(_edge_index)
            input_edge_type.append(_edge_type)
            pos_triples.append(_pos_triples)
            neg_nodes.append(_neg_nodes)
        input_edge_index = list(
            map(list, zip(*(iter(input_edge_index),))))  # nested list of shape (n_samples, )
        input_edge_type = list(map(list, zip(*(iter(input_edge_type),))))
        pos_triples = list(map(list, zip(*(iter(pos_triples),))))
        neg_nodes = list(map(list, zip(*(iter(neg_nodes),))))

        return input_edge_index, input_edge_type, pos_triples, neg_nodes

    def _process_one_graph(self, _edge_index, _edge_type, _node_type_ids):
        # _edge_index: tensor[2, E]
        # _edge_type:  tensor[E, ]
        # _node_type_ids: tensor[n_nodes, ]
        E = len(_edge_type)
        if E == 0:
            # print ('KG with 0 node', file=sys.stderr)
            effective_num_nodes = 1
        else:
            effective_num_nodes = int(_edge_index.max()) + 1
        device = _edge_type.device

        tmp = _node_type_ids.max().item()
        assert isinstance(tmp, int) and 0 <= tmp <= 5
        try:
            _edge_index_node_type = _node_type_ids[_edge_index]  # [2, E]
        except:
            print(_edge_index)
            print(_edge_index.shape)
            print(_edge_index.dtype)
        _is_special = (_edge_index_node_type == 3)  # [2, E]
        is_special = _is_special[0] | _is_special[1]  # [E,]

        positions = torch.arange(E)
        positions = positions[~is_special]  # [some_E, ]
        drop_count = min(100, int(len(positions) * 0.15))
        if len(positions) > 0 and drop_count > 0:
            drop_idxs = torch.multinomial(torch.full((len(positions),), 1.0), drop_count,
                                          replacement=False)  # [drop_count, ]
        else:
            drop_idxs = torch.tensor([]).long()
        drop_positions = positions[drop_idxs]  # [drop_count, ]

        mask = torch.zeros((E,)).long()  # [E, ]
        mask = mask.index_fill_(dim=0, index=drop_positions, value=1).bool().to(device)  # [E, ]

        real_drop_count = int(drop_count * (1 - 0.1))
        real_drop_positions = positions[drop_idxs[:real_drop_count]]  # [real_drop_count, ]
        real_mask = torch.zeros((E,)).long()  # [E, ]
        real_mask = real_mask.index_fill_(dim=0, index=real_drop_positions, value=1).bool().to(device)  # [E, ]

        assert int(mask.long().sum()) == drop_count
        # print (f'drop_E / total_E = {drop_count} / {E} = {drop_count / E}', ) #E is typically 1000-3000
        input_edge_index = _edge_index[:, ~real_mask]
        input_edge_type = _edge_type[~real_mask]
        assert input_edge_index.size(1) == E - real_drop_count

        pos_edge_index = _edge_index[:, mask]
        pos_edge_type = _edge_type[mask]
        pos_triples = [pos_edge_index[0], pos_edge_type, pos_edge_index[1]]
        # pos_triples: list[h, r, t], where each of h, r, t is [n_triple, ]
        assert pos_edge_index.size(1) == drop_count

        num_edges = len(pos_edge_type)
        num_corruption = 64
        neg_nodes = torch.randint(0, effective_num_nodes, (num_edges, num_corruption),
                                  device=device)  # [n_triple, n_neg]
        return input_edge_index, input_edge_type, pos_triples, neg_nodes


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.report_to != 'none':
        wandb.init(project="CausalKG", entity="yxqq", mode=data_args.wandb_mode, name=training_args.output_dir)


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed+data_args.n_runs)

    # Load dataset
    if model_args.outcome_prediction:
        compared_drug = data_args.compared_drug
        target_drug = data_args.target_drug
        data_args.train_data_file = [
            os.path.join(data_args.data_path, target_drug)]
        myTokenizer = MyTokenizer(
            vocab_file=model_args.vocab_file,
            baseline_window=data_args.baseline_window,
            fix_window_length=data_args.fix_window_length,
            treatment_list=[compared_drug.split('.')[0],
                            target_drug.split('.')[0]],
            use_kg=model_args.use_kg,
            sep_graph=model_args.sep_graph,
        )

    else:
        data_args.train_data_file = [os.path.join(data_args.data_path, file)
                                     for file in os.listdir(data_args.data_path)]
        myTokenizer = MyTokenizer(
            vocab_file=model_args.vocab_file,
            baseline_window=data_args.baseline_window,
            fix_window_length=data_args.fix_window_length,
            treatment_list=None,
            use_kg=model_args.use_kg,
            sep_graph=model_args.sep_graph,
        )

    if data_args.eval_data_file:
        data_files = {"train": data_args.train_data_file, "validation": data_args.eval_data_file}
        raw_datasets = load_dataset('json', data_files=data_files, field="data")

    else:
        data_files = {"train": data_args.train_data_file}
        raw_datasets = load_dataset('json', data_files=data_files, field="data",cache_dir=model_args.cache_dir)
        if model_args.outcome_prediction:
            overlap_pids = [data['pid'] for data in raw_datasets['train']]
            overlap_pids = Counter(overlap_pids)
            overlap_pids = [pid for pid in overlap_pids if overlap_pids[pid] > 1]
            raw_datasets = raw_datasets.filter(lambda example: example['pid'] not in overlap_pids)
        if data_args.validation_split_percentage > 0:
            raw_datasets = raw_datasets['train'].train_test_split(test_size=data_args.validation_split_percentage / 100,
                                                                  seed=data_args.n_runs+training_args.seed)
            raw_datasets['train'] = raw_datasets['train'].select(
                range(int(data_args.training_set_fraction * len(raw_datasets['train']))))

            test_valid = raw_datasets['test'].train_test_split(test_size=0.5)
            raw_datasets['validation'] = test_valid['train']
            raw_datasets['test'] = test_valid['test']


    max_seq_length = data_args.max_seq_length
    max_node_num = data_args.max_node_num

    def prepare_data_for_sep_graphs(example):
        result = myTokenizer.encode(example, max_length=max_seq_length, max_node_num=max_node_num)
        outcomes = example['outcome']
        result['outcome'] = list(outcomes.values())[0]

        pair_result = {}
        gnn_keys = ['concept_ids', 'node_type_ids', 'adj_lengths', 'special_nodes_mask', 'edge_index', 'edge_type']

        gnn_keys_ = [key+'_t' for key in gnn_keys] + [key+'_o' for key in gnn_keys]
        for key in result.keys() - set(gnn_keys_):
            pair_result[key] = [result[key], result[key]]
        for key in gnn_keys:
            pair_result[key] = [result[f'{key}_t'], result[f'{key}_o']]
        # add additional separate graph label to indicate whether the graph is treatment-covariate or outcome-covariate.
        # 1: treatment graph, 0: outcome graph
        pair_result['sep_graph_label'] = [1, 0]
        return pair_result

    tokenized_datasets = raw_datasets.map(prepare_data_for_sep_graphs, batched=False, num_proc=16,
                                          load_from_cache_file=not data_args.overwrite_cache)
    data_collator = myDataCollatorForSepGraphs(
        tokenizer=myTokenizer,
        mask_prediction=model_args.mask_prediction,
        outcome_prediction=model_args.outcome_prediction,
        link_prediction=model_args.link_prediction,
        use_kg=model_args.use_kg
    )

    if model_args.use_kg:
        tokenized_datasets = tokenized_datasets.filter(lambda example: len(example['edge_index'][0]) != 0, num_proc=1)

    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        train_dataset = tokenized_datasets["train"]

    if training_args.do_eval:
        eval_dataset = tokenized_datasets["validation"]

    def preprocess_logits_for_metrics(logits, labels):
        loss=0
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[1]  # link prediction score
        # logits = logits.softmax(dim=-1)
        return logits

    metric = load_metric("accuracy")

    def compute_metrics(p: EvalPrediction):

        def log_sigmoid(x):
            return np.log(1/(1 + np.exp(-x)))

        # (positive_score, negative_score), labels = eval_preds
        positive_score, negative_score = p.predictions

        y_pred_pos = positive_score.reshape(-1, 1)
        optimistic_rank = (negative_score >= y_pred_pos).sum(axis=1)
        pessimistic_rank = (negative_score > y_pred_pos).sum(axis=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        mrr_list = 1./ranking_list.astype(np.float32)
        mrr = mrr_list.mean()

        negative_score = (softmax(negative_score, axis=1) * log_sigmoid(-negative_score)).sum(axis=1)
        positive_score = log_sigmoid(positive_score).squeeze(1)
        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
        lp_loss = (positive_sample_loss + negative_sample_loss) / 2
        results = {}
        results['lp_loss'] = lp_loss
        results['mrr'] = mrr
        # results['my_loss'] = loss

        return results

    def preprocess_logits_for_metrics_downstream(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[1] # factual outcome logits
        # logits = logits.softmax(dim=-1)
        if isinstance(labels, tuple):
            outcome_labels, sep_graph_labels = labels
            outcome_labels = outcome_labels[sep_graph_labels==0]
        else:
            outcome_labels = labels
        return logits, outcome_labels


    def compute_metrics_downstream(p: EvalPrediction):

        logits, labels = p.predictions
        # logits = logits[1] if isinstance(logits, tuple) else logits.logits_factual
        logits = softmax(logits,axis=-1)
        # preds = logits.argmax(axis=-1)

        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        # preds = preds.reshape(-1)
        # mask = labels != -100
        # labels = labels[mask]
        # preds = preds[mask]

        # results =metric.compute(predictions=preds, references=labels)
        # results['logits'] = logits
        results = {}
        results['auc'] = roc_auc_score(labels,logits[:,1])
        # results['aupr'] = average_precision_score(labels, logits[:, 1])
        # results['f1'] = f1_score(labels,preds)

        return results

    config = KGTREATConfig(
        vocab_size=len(myTokenizer),
        type_vocab_size=len(myTokenizer.type),
        num_node_types = 2,
        num_edge_types=2*(len(myTokenizer.id2relation) + 1),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_visit_time_embeddings=data_args.baseline_window + 1,
        max_physical_time_embeddings=data_args.baseline_window // data_args.fix_window_length + 1,
        time_embedding=model_args.time_embedding,
    )


    if not model_args.model_name_or_path:
        if model_args.mask_prediction or model_args.link_prediction:
            logger.info("Train KGTREATForMaskedLMLP model from scratch...")
            model = KGTREATForMaskedLMLP(config, pretrained_concept_emb_file=data_args.ent_emb_paths)
        else:
            logger.info("Train KGTREATForSequenceClassification model from scratch...")
            model = KGTREATForSequenceClassification(config, pretrained_concept_emb_file=data_args.ent_emb_paths)
    else:
        if model_args.mask_prediction or model_args.link_prediction:
            logger.info("Loading KGTREATForMaskedLMLP from pretrained...")
            model = KGTREATForMaskedLMLP.from_pretrained(model_args.model_name_or_path)
        else:
            logger.info("Loading KGTREATForSequenceClassification from pretrained...")
            model = KGTREATForSequenceClassification.from_pretrained(model_args.model_name_or_path)

    training_args.remove_unused_columns = False
    if model_args.mask_prediction:
        training_args.label_names = ["mask_labels"]

    if model_args.link_prediction:
        training_args.label_names = ["pos_triples", "neg_nodes"]

    if model_args.outcome_prediction:
        training_args.label_names = ["outcome_labels"]
        if model_args.use_kg:
            training_args.label_names.append("sep_graph_labels")


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics_downstream if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics_downstream
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()