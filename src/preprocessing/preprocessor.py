import re
from transformers import RobertaTokenizer

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

        return re.sub(r"\s+", " ", code).strip()  # Remove excessive whitespace
