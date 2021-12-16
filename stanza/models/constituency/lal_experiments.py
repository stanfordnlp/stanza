from stanza.models.constituency.label_attention import LabelAttention, PositionwiseFeedForward, PartitionedPositionwiseFeedForward

# Integration of the Label Attention Layer
lal_params = {
    "lal_d_kv": 64,
    "lal_d_proj": 64,
    "lal_resdrop": True,
    "lal_pwff": True,
    "lal_q_as_matrix": False,
    "lal_partitioned": True,
    "lal_combine_as_self": False,
    "lal_d_positional": 1024 // 2
}

label_attention = LabelAttention(lal_params, 1024, lal_params["lal_d_kv"],
                                      lal_params["lal_d_kv"], 112, lal_params["lal_d_proj"], use_resdrop=lal_params["lal_resdrop"], q_as_matrix=lal_params["lal_q_as_matrix"],
                                      residual_dropout=0.2, attention_dropout=0.2, d_positional=lal_params["lal_d_positional"])
