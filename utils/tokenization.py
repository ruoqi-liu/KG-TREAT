import collections
import copy

from torch import Tensor
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import os


def load_vocab(vocab_file, use_kg):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    type = set()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
        if ':' not in token:
            type.add('special_token')
        else:
            type.add(token.split(':')[0])
    type = sorted(list(type))
    type = {t:index for index, t in enumerate(list(type))}

    id2relation = None
    if use_kg and os.path.isfile(f"{os.path.dirname(vocab_file)}/relations.txt"):
        id2relation = [r.strip() for r in open(f"{os.path.dirname(vocab_file)}/relations.txt")]
    return vocab, type, id2relation


class MyTokenizer():
    def __init__(
            self,
            vocab_file,
            sep_token="[SEP]",
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            treatment_list = None,
            baseline_window=90,
            fix_window_length=30,
            use_kg = False,
            sep_graph = False,
    ):

        self.sep_token = sep_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.use_kg = use_kg
        self.sep_graph = sep_graph

        self.vocab, self.type, self.id2relation = load_vocab(vocab_file, use_kg)
        self.id2token = {index:token for token, index in self.vocab.items()}
        self.id2type = {index:type for type, index in self.type.items()}

        self.treatment_ids = None
        self.baseline_window = baseline_window
        self.fix_window_length = fix_window_length
        self.treatment_list = treatment_list

    def convert_token_ids_to_token_type_ids(self, token_ids):

        if isinstance(token_ids, int):
            token = self.id2token.get(token_ids)
            token_type_ids = self.type.get(token.split(':')[0]) if ':' in token else self.type.get('special_token')
        else:
            token_type_ids = []
            for token_id in token_ids:
                token = self.id2token.get(token_id)
                type_id = self.type.get(token.split(':')[0]) if ':' in token else self.type.get('special_token')
                token_type_ids.append(type_id)

        return token_type_ids

    def construct_gnn_input(self, concepts,adj_row, adj_col, adj_shape, max_node_num):
        """Construct input for the GNN component of the model."""

        # Set special nodes and links
        context_node = 0
        n_special_nodes = 1
        cxt2qlinked_rel = 0
        half_n_rel = len(self.id2relation) + 1

        # edge_index, edge_type = [], []
        concept_ids = np.full((max_node_num), 1, dtype=np.long)
        node_type_ids = np.full((max_node_num), 0, dtype=np.long)
        special_nodes_mask = np.zeros(max_node_num, dtype=np.long)

        special_nodes_mask[:n_special_nodes] = 1

        num_concept = min(len(concepts) + n_special_nodes, max_node_num)
        adj_lengths_ori = len(concepts)
        adj_lengths = num_concept

        # Prepare nodes
        concepts = np.array(concepts)
        concepts = concepts[:num_concept - n_special_nodes]
        concept_ids[
        n_special_nodes:num_concept] = concepts + 1  # To accomodate contextnode, original concept_ids incremented by 1
        concept_ids[0] = context_node  # this is the "concept_id" for contextnode

        # Prepare node types
        node_type_ids[0] = 1  # context node

        # Load adj
        ij = np.array(adj_row, dtype=np.int64)  # (num_matrix_entries, ), where each entry is coordinate
        k = np.array(adj_col, dtype=np.int64)  # (num_matrix_entries, ), where each entry is coordinate
        n_node = adj_shape[1]
        if n_node > 0:
            # assert len(self.id2relation) == adj.shape[0] // n_node
            i, j = ij // n_node, ij % n_node
        else:
            i, j = ij, ij

        # Prepare edges
        i += 1  # rel_id
        j += 1  # coordinate
        k += 1  # **** increment coordinate by 1, rel_id by 1 ****
        extra_i, extra_j, extra_k = [], [], []
        for _coord in range(len(concepts)):
            _new_coord = _coord + n_special_nodes
            if _new_coord > num_concept:
                break
            extra_i.append(cxt2qlinked_rel)  # rel from contextnode to question concept
            extra_j.append(0)  # contextnode coordinate
            extra_k.append(_new_coord)  # question concept coordinate

        if len(extra_i) > 0:
            i = np.concatenate((i, np.array(extra_i)), axis=0)
            j = np.concatenate((j, np.array(extra_j)), axis=0)
            k = np.concatenate((k, np.array(extra_k)), axis=0)

        mask = (j < max_node_num) & (k < max_node_num)
        i, j, k = i[mask], j[mask], k[mask]
        i, j, k = np.concatenate((i, i + half_n_rel), 0), np.concatenate((j, k), 0), np.concatenate((k, j),
                                                                                                    0)  # add inverse relations
        edge_index = np.stack([j, k], axis=0)  # each entry is [2, E]
        edge_type = i  # each entry is [E, ]

        return concept_ids.tolist(), node_type_ids.tolist(), adj_lengths, special_nodes_mask.tolist(), edge_index, edge_type

    def encode(self, data, max_length=None, max_node_num=None, padding=True, return_tensor=False):

        treatment_group = data['treatment_group']
        covariates = data['covariates']
        covariates_time = data['covariates_time']

        # [cls] demo x1, x2, ...
        input_ids = [self.vocab.get(self.cls_token)]
        token_type_ids = [self.type.get('special_token')]
        treatment_list = self.treatment_list

        visit_time_ids = [0] * len(input_ids)
        physical_time_ids = [0] * len(input_ids)

        padding_idx = self.vocab.get(self.pad_token)

        n_visit = 0
        prev_visit = 0
        for i, covariate in enumerate(covariates):

            visit_time = covariates_time[i]

            if visit_time >= self.baseline_window :
                continue

            if covariate == 'medication:{}'.format(treatment_group):
                continue

            if visit_time != 0 and visit_time != prev_visit:
                n_visit += 1
                prev_visit = visit_time

            visit_time_ids.append(n_visit)
            physical_time_ids.append(visit_time//self.fix_window_length)

            # handle diagnosis code with incorrect dxver flag
            if covariate.startswith('diagnosis') and covariate not in self.vocab:
                covariate_type, covariate_val = covariate.split(':')
                dxver = covariate_val.split('@')[0]
                dxver = '9' if dxver == '0' else '0'
                covariate_val = f'{dxver}@' + covariate_val.split('@')[1]
                covariate = covariate_type + ':' + covariate_val

            input_id = self.vocab.get(covariate, None)
            if not input_id:
                print(f'not found {covariate}')
            assert input_id
            input_ids.append(input_id)
            covariate_type = covariate.split(':')[0]
            token_type_ids.append(self.type.get(covariate_type))

        # truncate
        if max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            token_type_ids = token_type_ids[:max_length]
            visit_time_ids = visit_time_ids[:max_length]
            physical_time_ids = physical_time_ids[:max_length]

        attention_mask = [1] * len(input_ids)
        # padding
        if padding:
            attention_mask += [padding_idx] * (max_length - len(input_ids))
            input_ids += [padding_idx] * (max_length - len(input_ids))
            token_type_ids += [self.type.get('special_token')] * (max_length - len(token_type_ids))
            visit_time_ids += [visit_time_ids[-1]] * (max_length - len(visit_time_ids))
            physical_time_ids += [physical_time_ids[-1]] * (max_length - len(physical_time_ids))

        ids_ = {'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'visit_time_ids': visit_time_ids,
                'physical_time_ids': physical_time_ids,
                'attention_mask': attention_mask,
                }

        if treatment_list:
            treatment_label = treatment_list.index(treatment_group)
            treatment_id = self.vocab.get('medication:{}'.format(treatment_group))
            cf_treatment_id = self.vocab.get('medication:{}'.format(treatment_list[1-treatment_label]))
            input_ids_cf = [id if id != treatment_id else cf_treatment_id for id in input_ids]
            ids_['input_ids_cf'] = input_ids_cf
            ids_['treatment_label'] = treatment_label

        if self.use_kg and not self.sep_graph:
            concepts = data['concepts']
            adj_row = data['adj_row']
            adj_col = data['adj_col']
            adj_shape = data['adj_shape']
            concept_ids, node_type_ids, adj_lengths, special_nodes_mask, edge_index, edge_type = self.construct_gnn_input(concepts,adj_row, adj_col, adj_shape, max_node_num)

            ids_['concept_ids']= concept_ids
            ids_['node_type_ids']= node_type_ids
            ids_['adj_lengths']= adj_lengths
            ids_['special_nodes_mask']= special_nodes_mask
            ids_['edge_index']= edge_index
            ids_['edge_type']= edge_type

        if self.sep_graph:
            concepts = data['concepts_t']
            adj_row = data['adj_row_t']
            adj_col = data['adj_col_t']
            adj_shape = data['adj_shape_t']
            concept_ids, node_type_ids, adj_lengths, special_nodes_mask, edge_index, edge_type = self.construct_gnn_input(
                concepts, adj_row, adj_col, adj_shape, max_node_num)

            ids_['concept_ids_t'] = concept_ids
            ids_['node_type_ids_t'] = node_type_ids
            ids_['adj_lengths_t'] = adj_lengths
            ids_['special_nodes_mask_t'] = special_nodes_mask
            ids_['edge_index_t'] = edge_index
            ids_['edge_type_t'] = edge_type

            concepts = data['concepts_o']
            adj_row = data['adj_row_o']
            adj_col = data['adj_col_o']
            adj_shape = data['adj_shape_o']
            concept_ids, node_type_ids, adj_lengths, special_nodes_mask, edge_index, edge_type = self.construct_gnn_input(
                concepts, adj_row, adj_col, adj_shape, max_node_num)

            ids_['concept_ids_o'] = concept_ids
            ids_['node_type_ids_o'] = node_type_ids
            ids_['adj_lengths_o'] = adj_lengths
            ids_['special_nodes_mask_o'] = special_nodes_mask
            ids_['edge_index_o'] = edge_index
            ids_['edge_type_o'] = edge_type
        return ids_

    def decode(self, token_ids, with_special_tokens=True):
        if isinstance(token_ids,Tensor):
            token_ids = token_ids.numpy()
        result = []

        if with_special_tokens:
            for id in token_ids:
                result.append(self.id2token.get(id))
        else:
            for id in token_ids:
                if self.id2token.get(id) in [self.sep_token, self.pad_token, self.cls_token, self.mask_token]:
                    continue
                result.append(self.id2token.get(id))

        return result

    def get_treatment_ids(self) -> List[int]:
        if not self.treatment_ids:
            self.treatment_ids = [v for k,v in self.vocab.items() if 'treatment' in k]

        return self.treatment_ids

    @property
    def all_special_tokens(self) -> List[str]:
        """
        `List[str]`: All the special tokens (`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        Convert tokens of `tokenizers.AddedToken` type to string.
        """
        all_toks = [self.sep_token,self.unk_token,self.pad_token,self.cls_token,self.mask_token]
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token) for token in tokens]

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.
        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        assert already_has_special_tokens and token_ids_1 is None, (
            "You cannot use ``already_has_special_tokens=False`` with this tokenizer. "
            "Please use a slow (full python) tokenizer to activate this argument. "
            "Or set `return_special_tokens_mask=True` when calling the encoding method "
            "to get the special tokens mask in any tokenizer. "
        )

        all_special_ids = self.all_special_ids  # cache the property
        all_treatment_ids = self.get_treatment_ids()

        combined_ids = all_special_ids + all_treatment_ids

        special_tokens_mask = [1 if token in combined_ids else 0 for token in token_ids_0]

        return special_tokens_mask

    def __len__(self):
        return len(self.vocab)
