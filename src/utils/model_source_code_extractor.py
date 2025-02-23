import ast

def extract_model_source_code(code_string):
    """Extracts the Model class and numpy_dataset definition from code string."""
    try:
        tree = ast.parse(code_string)
        extractor = CodeExtractor()
        extractor.visit(tree)

        model_class_code = extractor.model_class_code

        return model_class_code
    except Exception as e:
        print(f"Failed to parse code: {e}")
        return None, None

class CodeExtractor(ast.NodeVisitor):
    def __init__(self):
        self.model_class_code = None

    def visit_ClassDef(self, node):
        """Extracts the 'Model' class definition."""
        if node.name == "Model":
            self.model_class_code = ast.unparse(node)
        self.generic_visit(node)
