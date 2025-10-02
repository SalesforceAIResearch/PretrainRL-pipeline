IDENTIFIER_FORMAT = """
The output MUST strictly adhere to the following JSON format, and other text MUST NOT be included:
```
{
  "thought": "Describe your reasoning for identifying the domain and persona of the material.",
  "domain": "The domain of the material.",
  "persona": "The persona that the material is intended for. Separate with comma if there are multiple personas. Max 3 personas.",
}
```
"""

ALL_DOMAINS = [
    "Math",
    "Technology & Engineering",
    "Coding",
    "Social Science",
    "Natural Science",
    "Travel & Lifestyle",
    "Commerce & Economics",
    "Medicine & Health",
    "Education",
    "Other",
]

IDENTIFIER_TEMPLATE = """
You are a helpful data analyst. You will be given a material from a website which can come from very diverse sources and may not be well-structured. Our final goal is to generate question and answer pair from the material. In this stage, your task is to identify the domain and persona of the material.

Here are the instructions for the domain and persona:
- The domain is the main topic of the material. You should choose from the following domains: {all_domains}. If you find that the material is not related to any of the domains or the domain is not clear, you should choose "Other". If there are multiple domains that the material is related to, you should choose the most relevant domain.
- The persona is the intended audience of the material. If the material is intended for multiple personas, you should list several personas (up to 3) that will be interested in the material.

Based on the above instructions, identify the domain and persona of the material.

Material:
{material}

{format_inst}
"""