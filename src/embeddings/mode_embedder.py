from src.model.roberta_model_encoder import Roberta_Model_Encoder
from transformers import AutoConfig
import torch
import torch.nn.functional as F
import os
from dotenv import load_dotenv
from safetensors.torch import load_file


class Model_Embedder:
    def __init__(self):
        self.model = None
        load_dotenv()
        self._load_pretrained_encoder()

    def _load_pretrained_encoder(self):
        config_path = os.getenv("CONFIG_PATH")
        if not config_path:
            raise ValueError("CONFIG_PATH environment variable is not set.")
        config = AutoConfig.from_pretrained(config_path)

        self.model = Roberta_Model_Encoder(config)

        model_weights = load_file(f"{config_path}/model.safetensors", device="CPU")
        self.model.load_state_dict(model_weights, strict=True)

    def get_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        r""" Interence-only method: Generate normalised embeddings from a single input 
            The siamese-network is trained with two inputs, but we would require only one input
            for inference
        
        Args:
            input_ids (torch.Tensor): Tokenised input IDs,
            attention_mask (torch.Tensor): Attention mask

        Returns:
            torch.Tensor: Normalised 768-dimensional embedding
        """

        with torch.no_grad():
            outputs = self.model.roberta(input_ids = input_ids, attention_mask = attention_mask, return_dict=True)
            emb = self.model.last_4_layer_avg(outputs)
            emb = self.net(emb)
            emb = F.normalize(emb, p=2, dim=-1)
            return emb


