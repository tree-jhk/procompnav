import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

class QwenRegistry:
    _tokenizers = {}
    _models = {}

    @classmethod
    def _set_seed(cls, seed=42):
        """Set seeds for reproducibility across runs."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def get_tokenizer(cls, model_name: str):
        """Load and cache the tokenizer for the specified model path."""
        if model_name not in cls._tokenizers:
            cls._set_seed()
            cls._tokenizers[model_name] = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        return cls._tokenizers[model_name]

    @classmethod
    def get_model(cls, model_name: str):
        """Load and cache the model for the specified model path."""
        if model_name not in cls._models:
            cls._set_seed()
            cls._models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cuda",
                dtype=torch.float16,
                trust_remote_code=True,
            )
        return cls._models[model_name]


from types import SimpleNamespace

class QwenCompatibleClient:
    def __init__(self, base_url: str):
        """
        Initializes the Qwen-compatible client that mimics the OpenAI interface.

        Args:
            base_url (str): Local path to the Qwen model directory, e.g., "/workspace/Qwen2.5-72B-Instruct-AWQ"
        """
        self.model_path = base_url
        self.tokenizer = QwenRegistry.get_tokenizer(self.model_path)
        self.model = QwenRegistry.get_model(self.model_path)
        self.model.generation_config.temperature=None
        self.model.generation_config.top_p=None
        self.device = self.model.device
        self.model_name = self.model_path.split("/")[-1]

    @property
    def chat(self):
        """Supports OpenAI-style `chat.completions.create()` structure."""
        return self

    @property
    def completions(self):
        """Supports OpenAI-style `chat.completions.create()` structure."""
        return self

    def create(self, model: str, messages: list, **kwargs):
        """
        Mimics the OpenAI chat completion interface.

        Args:
            model (str): Model name (ignored; kept for API compatibility).
            messages (list): List of messages in OpenAI format:
                             e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            kwargs: Additional keyword arguments such as max_tokens.

        Returns:
            SimpleNamespace: Mimics OpenAI's completion object:
                             .choices[0].message.content contains the generated text.
        """
        prompt = messages[-1]["content"]
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "You are a helpful assistant.")

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Format input with chat template
        prompt_text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        ).to(self.device)
        prompt_len = inputs['input_ids'].shape[1]
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                # max_new_tokens=kwargs.get("max_tokens", 3000),
                max_new_tokens=3000,
                do_sample=False,  # Use deterministic decoding
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output_ids = output[0][prompt_len:]
        decoded = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=decoded))
            ]
        )
