from src.model.roberta_model_encoder import Roberta_Model_Encoder
from transformers import AutoConfig, RobertaTokenizer
import torch
import torch.nn.functional as F
import os
from dotenv import load_dotenv
from safetensors.torch import load_file
import re

class Model_Embedder:
    def __init__(self):
        self.model = None
        self._load_pretrained_encoder()
        self.tokeniser = Tokeniser()

    def _load_pretrained_encoder(self):
        config_path = os.getenv("MODEL_ENCODER_CONFIG_PATH")
        load_dotenv()
        if not config_path:
            raise ValueError("MODEL_ENCODER_CONFIG_PATH environment variable is not set.")
        config = AutoConfig.from_pretrained(config_path)

        self.model = Roberta_Model_Encoder(config)

        model_weights = load_file(f"{config_path}/model.safetensors", device="cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(model_weights, strict=True)

        self.model.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_embedding(self, code_string: str) -> torch.Tensor:
        r""" Interence-only method: Generate normalised embeddings from a single input 
            The siamese-network is trained with two inputs, but we would require only one input
            for inference
        
        Args:
            code_string: string object to get embeded

        Returns:
            torch.Tensor: Normalised 768-dimensional embedding
        """

        tokens = self.tokeniser(code_string)

        input_ids = torch.tensor(tokens["input_ids"]).unsqueeze(0)
        attention_mask = torch.tensor(tokens["attention_mask"]).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model.roberta(input_ids = input_ids, attention_mask = attention_mask, return_dict=True)
            emb = self.model.last_4_layer_avg(outputs)
            emb = self.model.net(emb)
            emb = F.normalize(emb, p=2, dim=-1)
            emb = emb.squeeze(0).tolist()
            return emb

class Tokeniser:
    def __init__(self, max_length=256):
        self.tokeniser = RobertaTokenizer.from_pretrained("roberta-base")
        self.preprocessor = CodePreprocessor()
        self.max_length = max_length

    def __call__(self, code_string: str):
        processed_code = self.preprocessor.preprocess(code_string)
        return self.tokeniser(
            processed_code,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )


class CodePreprocessor:
    """Combines function and annotation preprocessing."""

    def get_function_names(self, code: str):
        """Extracts all function names in the code."""
        return re.findall(r"\ndef ([a-zA-Z0-9_]+)\(", code)

    def should_remove_function(self, code: str, function_name: str):
        """Determines if a function appears only once."""
        return len(re.findall(rf"\b{function_name}\b", code)) <= 1

    def delete_function(self, code: str, function_name: str):
        """Deletes a function definition and its body."""
        pattern = rf"(?s)def {function_name}\(.*?\):.*?(?=\ndef |\Z)"
        return re.sub(pattern, "", code)

    def delete_annotations(self, code: str):
        """Removes comments, import statements, and excessive whitespace."""
        code = re.sub(r"#.*", "", code)  # Remove inline comments
        code = "\n".join(line for line in code.splitlines() if not line.strip().startswith("import")) # Remove import statement
        code = re.sub(r"\s+", " ", code).strip()  # ✅ Remove excessive spaces
        return code

    def preprocess(self, code: str):
        """Full preprocessing pipeline."""
        code = "\n" + code.strip()
        code = self.delete_annotations(code)

        for fn_name in self.get_function_names(code):
            if self.should_remove_function(code, fn_name):
                code = self.delete_function(code, fn_name)

        return re.sub(r"\s+", " ", code).strip()  
