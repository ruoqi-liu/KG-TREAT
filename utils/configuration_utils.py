from transformers.modeling_utils import logging, PretrainedConfig


logger = logging.get_logger(__name__)

class KGTREATConfig(PretrainedConfig):
    model_type = "kgtreat"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        # GNN parameters
        num_gnn_layers=5,
        num_node_types=4,
        num_edge_types=38,
        concept_dim=200,
        gnn_hidden_size=200,
        # LP parameters
        link_decoder='DistMult',
        link_gamma=0,
        scaled_distmult=True,
        link_proj_headtail=False,
        link_normalize_headtail=3,
        link_regularizer_weight=0.01,
        link_negative_adversarial_sampling_temperature=1,
        link_negative_adversarial_sampling=True,
        # Task
        num_labels=2,
        **kwargs
    ):
        """Constructs KGTREATConfig."""
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        # LM parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

        # GNN parameters
        self.num_gnn_layers = num_gnn_layers
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.concept_dim = concept_dim
        self.gnn_hidden_size = gnn_hidden_size

        # LP parameters
        self.link_decoder = link_decoder
        self.link_gamma = link_gamma
        self.scaled_distmult = scaled_distmult
        self.link_proj_headtail = link_proj_headtail
        self.link_normalize_headtail = link_normalize_headtail
        self.link_regularizer_weight = link_regularizer_weight
        self.link_negative_adversarial_sampling_temperature = link_negative_adversarial_sampling_temperature
        self.link_negative_adversarial_sampling = link_negative_adversarial_sampling

        # Task
        self.num_labels = num_labels

        # These GNN layer configs rarely change, but they are parameters in
        # the original config file. Keep them here for now.
        default_dropout = 0.2
        self.num_lm_gnn_attention_heads = kwargs.pop("num_lm_gnn_attention_heads", 2)
        self.fc_dim = kwargs.pop("fc_dim", 200)
        self.n_fc_layer = kwargs.pop("n_fc_layer", 0)
        self.p_emb = kwargs.pop("p_emb", default_dropout)
        self.p_gnn = kwargs.pop("p_gnn", default_dropout)
        self.p_fc = kwargs.pop("p_fc", default_dropout)
        self.ie_dim = kwargs.pop("ie_dim", 200)
        self.info_exchange = kwargs.pop("info_exchange", True)
        self.ie_layer_num = kwargs.pop("ie_layer_num", 1)
        self.sep_ie_layers = kwargs.pop("sep_ie_layers", False)
        self.layer_id = kwargs.pop("layer_id", -1)