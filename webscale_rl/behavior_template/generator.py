GENERATOR_FORMAT = """
The output MUST strictly adhere to the following JSON format, and other text MUST NOT be included:
```
{
  "thought": "Describe your reasoning for generating question and answer pair.",
  "question": "The question generated from the material.",
  "answer": "The answer generated from the material.",
}
```
"""

GENERATOR_TEMPLATE = """
You will be given a material from a website which can come from very diverse sources and may not be well-structured. Our final goal is to generate question and answer pair from the material. In this stage, your task is to generate a question and answer pair from the material.

Here are the instructions for the question and answer generation:
- You will act as a given persona. You should generate a question and answer pair from your perspective.
- Both the question and answer should be totally from the material. Do not generate any information that is not in the material.
- You should generate such a question that its corresponding answer is relatively short and can be easily and clearly verified.
- The generated question will be asked without providing the original material. Therefore, you should add a necessary brief introduction of the background before the question. NEVER ask a question with "according to the material".
- When adding introduction to the question, you should NEVER explicitly include the answer in the question, which will be viewed as info leakage and is strictly forbidden.

Here are some examples of QA pairs extracted from the material:
{few_shot_example}

Based on the above instructions and examples, generate the question and answer pair from the material according to your persona.

[Material]
{material}


[Your persona]
{persona}

{format_inst}
"""