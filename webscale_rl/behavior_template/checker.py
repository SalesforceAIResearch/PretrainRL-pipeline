CHECKER_FORMAT = """
The output MUST strictly adhere to the following JSON format, and other text MUST NOT be included:
```
{
  "thought": "Describe your reasoning for checking the question and answer pair.",
  "has_context": "Y for has context, N for no context",
  "answer_correctness": "Y for correct, N for incorrect",
  "info_leakage": "Y for has info leakage, N for no info leakage",
}
```
"""

CHECKER_TEMPLATE = """
You are a data labeler. You will be given a material and a question and answer pair generated from the material. Your task is to check whether the question and answer pair is correct according to the material and whether there is info leakage from question to answer.

Here are the instructions for checking:
- The necesssary context for the question should be added to the question, e.g., a question with "according to the material" should be removed.
- For the answer correctness, you should check whether the answer is correct according to the original material.
- The information leakage indicates that the question explicitly provides information about the answer and then the answer can be directly obtained from the question.

Based on the above instructions, check the QA pair extracted from the original material in terms of the answer correctness and info leakage.

[Original Material]:
{material}

[Extracted Question]:
{question}

[Extracted Answer]:
{answer}

{format_inst}
"""