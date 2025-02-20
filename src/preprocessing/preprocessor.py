# Functional and annotational preprocessor that are used to remove comments and extra spaces

import re


class FunctionPreprocessor:
    def get_function(self, code):
        results = []
        fn_list = re.findall(r"\ndef [a-zA-Z0-9_]+\(", code)
        for fn in fn_list:
            results.append(fn[4:-1].strip())
        return results

    def determine_function(self, code, function_name):
        num = len(re.findall(r"[^a-zA-Z]" + function_name + r"[^a-zA-Z]", code))
        return False if num <= 1 else True

    def delete_function(self, code, name):
        start_id, _ = re.search("def " + name, code).span()
        ptr = start_id
        while ptr < len(code) - 1:
            if code[ptr] == "\n" and re.search("[a-zA-Z]", code[ptr + 1]) is not None:
                break
            ptr += 1
        if ptr != len(code) - 1:
            end_id = ptr
            code = code[:start_id] + code[end_id:]
        return code

    def preprocess(self, code):
        code = "\n" + code
        fn_list = self.get_function(code)
        if len(fn_list) == 0:
            return code
        for fn in fn_list:
            flag = self.determine_function(code, fn)
            if not flag:
                code = self.delete_function(code, fn)
        return code

class AnnotationPreprocessor:
    def delete_annotation(self, code):
        sens = code.split("\n")
        sens_processed = [sen.split("#")[0] for sen in sens]  # Remove inline comments
        return "\n".join(sens_processed)

    def delete_import(self, code):
        sens = code.split("\n")
        sens_processed = [sen for sen in sens if "import" not in sen]  # Remove import statements
        return "\n".join(sens_processed)

    def preprocess(self, code):
        code = self.delete_annotation(code)
        code = self.delete_import(code)
        code = re.sub(r"\s+", " ", code).strip()  # Remove excessive whitespace
        return code