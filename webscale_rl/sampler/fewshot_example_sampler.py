import json
import random
from typing import Union, Dict, List


class FewShotExampleSampler:
    """
    A class to sample and process example JSON datasets where each line is a separate JSON entry.
    """

    def __init__(self, few_shot_example_template: str, example_data_path: Dict[str, str]) -> None:
        """
        Initializes the ExampleSampler object by loading data from the specified JSON file path.
        """
        self.few_shot_example_template = few_shot_example_template
        self.example_data = {k: self._read_jsonl_file(v) for k, v in example_data_path.items()}

    @staticmethod
    def _read_jsonl_file(file_path: str):
        """ Utility function to read a JSON file and return the data. """
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def format_examples(self, examples: List[Dict]):
        """
        Formats a list of example dictionaries into a structured string.
        """
        example_str = "\n[BEGIN OF EXAMPLES]\n"
        for i, example in enumerate(examples):
            example_str += f"Example {i}:\n"
            example_str += self.few_shot_example_template.format(
                original_material=example["original_material"],
                persona=example["persona"],
                question=example["question"],
                answer=example["answer"],
            )
            example_str += "\n"
        example_str += "[END OF EXAMPLES]\n\n"
        return example_str

    def sample(self, n, key: str = "all"):
        """
        Samples `n` examples from the loaded example data.
        """
        if key == "all":
            key = list(self.example_data.keys())
        elif key in self.example_data:
            key = [key]
        else:
            raise ValueError(f"Key {key} not found in example data")

        sampled_examples = []
        for k in key:
            examples = random.sample(self.example_data[k], n)
            sampled_examples.extend(examples)

        return sampled_examples
