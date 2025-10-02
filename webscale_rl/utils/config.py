from dataclasses import asdict, dataclass

@dataclass
class ModelConfig:
    model_name: str = "gpt-4.1"
    temperature: float = 0.7
    max_tokens: int = 4096 # 8196
    port: int = 8001
    num_fewshot: int = 2