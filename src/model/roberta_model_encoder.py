# This is the base class for model string encoder. The model is pre-trained using open-source dataset by poolC
# The pretrained-model taken in a code string and its self.net output is used for generate embedding that contain knowledge regarding a model's semnatic behavior.
# The embedding output is then used by the XGBoost regressor to predict a kendall-tau-rank and hence for measuring how similar two model is

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaPreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional

@dataclass
class SiameseOutput(ModelOutput):
    """Custom Hugging Face-compatible output format for similarity models"""
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    similarity_score: torch.FloatTensor = None  # ✅ Explicitly named similarity score
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None

class Roberta_Model_Encoder(RobertaPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.loss_fn_bce = nn.BCEWithLogitsLoss()

    def last_4_layer_avg(self, model_output):
        layer_indices = [-1, -2, -3, -4] 
        stacked_layers = torch.stack([model_output.hidden_states[i] for i in layer_indices])  # (4, batch, seq_len, hidden_dim)
        return stacked_layers.mean(dim=0)[:, 0, :] 

    def forward(
        self,
        input_ids1=None,
        attention_mask1=None,
        input_ids2=None,
        attention_mask2=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs1 = self.roberta(input_ids1, attention_mask=attention_mask1, return_dict=return_dict)
        outputs2 = self.roberta(input_ids2, attention_mask=attention_mask2, return_dict=return_dict)

        emb1 = self.last_4_layer_avg(outputs1)
        emb2 = self.last_4_layer_avg(outputs2)

        emb1 = self.net(emb1)
        emb2 = self.net(emb2)

        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)

        cosine_sim = F.cosine_similarity(emb1, emb2, dim=-1).unsqueeze(1)  # Shape: (batch_size, 1)

        logit_scale = 8 
        margin = 0.2  
        logits = logit_scale * (cosine_sim - margin)
        logits = logits.squeeze(1)

        loss = None
        if labels is not None:
            loss = self.loss_fn_bce(logits, labels.float())  # ✅ Use BCE loss with logits

        return SiameseOutput(
            loss=loss,
            logits=logits,  # ✅ Return logits
            similarity_score=torch.sigmoid(logits),
            hidden_states=outputs1.hidden_states if return_dict else None,
            attentions=outputs1.attentions if return_dict else None,
        )
