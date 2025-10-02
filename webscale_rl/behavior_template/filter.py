FILTER_FORMAT = """
The output MUST strictly adhere to the following JSON format, and other text MUST NOT be included:
```
{
  "thought": "Describe your reasoning for identifying whether the data is qualified or not.",
  "qualified": "Y for qualified, N for not qualified.",
}
```
"""

FILTER_TEMPLATE = """
You are a helpful data analyst. You will be given a material which can come from very diverse sources and may not be well-structured. Our final goal is to generate question and answer pair from the material. In this stage, your task is to identify whether the material is qualified for the following criteria:
- The material is informative and self-contained for the user.
- The content has sufficient depth and clarity.
- It's possible to extract question and corresponding answer from the material.
Based on the above instructions, identify whether the material is qualified or not.

Material:
{material}

{format_inst}
"""