import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
from colorama import Fore
from colorama import init as init_colorama
from openai import OpenAI
import requests

init_colorama(autoreset=True)
from dotenv import load_dotenv
from retrying import retry
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from openai import OpenAI
try:
    from qwen_compatible_client import QwenCompatibleClient
except:
    from .qwen_compatible_client import QwenCompatibleClient


def get_llm_backend(llm_client_params):
    model_name = llm_client_params.get("model", "").lower()
    base_url = llm_client_params.get("base_url", "")
    print(model_name)
    # if "qwen" in model_name.lower() or os.path.exists(llm_client_params.get("base_url", "")):
    #     return QwenCompatibleClient(base_url=llm_client_params["base_url"])
    # else:
    #     return OpenAI(**llm_client_params)
    # vLLM case: Use OpenAI-compatible REST API
    if base_url.startswith("http://") or base_url.startswith("https://"):
        print(Fore.CYAN + "[INFO] Using vLLM backend via OpenAI-compatible API")
        return OpenAI(
            api_key="EMPTY",  # vLLM does not require an API key
            base_url=base_url,
        )

    # Local path case: Use QwenCompatibleClient
    elif os.path.exists(base_url):
        print(Fore.CYAN + "[INFO] Using local Qwen-compatible model")
        return QwenCompatibleClient(base_url=base_url)

    else:
        print(Fore.CYAN + "[INFO] Using default OpenAI backend")
        return OpenAI(**llm_client_params)

class OpenAILLMClient:
    def __init__(self, llm_client_params) -> None:
        print(Fore.YELLOW + f"[INFO] Initializing OpenAI LLM")

        load_dotenv(".env.llm_client_key")
        all_env_vars = os.environ

        self.api_keys = [value for key, value in all_env_vars.items() if key.startswith("LLM_CLIENT_KEY")]
        llm_client_params["api_key"] = self.api_keys[0] if self.api_keys else None
        self.model = llm_client_params.get("model", "gpt-4o")
        del llm_client_params["model"]

        # self.client = OpenAI(**llm_client_params)
        self.client = get_llm_backend(llm_client_params)
        try:
            self.model_name = self.client.models.list().data[0].id
        except:
            try:
                self.model_name = llm_client_params.get("model", "").lower()
            except:
                self.model_name = "gpt-4o"

    @retry(
        retry_on_exception=(
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
            Exception,
        ),
        stop_max_attempt_number=1,
        wait_fixed=6000,
    )
    def ask(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                top_p=1,
                max_tokens=2500,
                # max_tokens=20000,
                seed=42,
            )
            return completion.choices[0].message.content

        except RateLimitError as e:
            print(Fore.RED + "[ERROR] Rate Limit Error")
            print(Fore.RED + f"[ERROR] {e}")
            raise Exception("retry")

    def ask_with_likelihood(
        self,
        prompt: str,
        max_tokens: int = 1,
        top_logprobs: int = 20,
    ):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            top_p=1,
            max_tokens=max_tokens,
            seed=42,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        response_text = completion.choices[0].message.content
        token_logprobs = completion.choices[0].logprobs.content
        first_token_top_logprobs = token_logprobs[0].top_logprobs if token_logprobs else []
        likelihood = [
            (str(item.token).strip(), float(math.exp(float(item.logprob))))
            for item in first_token_top_logprobs
        ]
        return response_text, likelihood


if __name__ == "__main__":
    ## Test with python vlfm/vlm/openai_llm.py

    # you can also use Groq for testing, otherwise it will use OpenAI
    ### make sure to set the environment variable LLM_CLIENT_KEY (inside .env.llm_client_key) to either your OpenAI API key or groq API key
    # If using Groq, register here for a free api (https://groq.com/)
    # test_with_groq = True
    test_with_groq = False

    llm_client_params = {
        # "model": "Meta-Llama-3.1-8B-Instruct",
        # "model": "/workspace/Qwen2.5-72B-Instruct-AWQ",
        "model": "local_model",
        "base_url": "http://localhost:8000/v1",  # vLLM endpoint
    }

    llm_client = OpenAILLMClient(llm_client_params)

    prompt = "What is the capital of France?"
    response = llm_client.ask(prompt)
    print(Fore.GREEN + f"Input: {prompt}")
    print(Fore.GREEN + f"Response: {response}")
