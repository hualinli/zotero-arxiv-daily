from llama_cpp import Llama
from openai import OpenAI
from loguru import logger
from time import sleep
import tiktoken

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None,lang: str = "English"):
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
            self.n_ctx = 128000  # OpenAI models have large context windows
        else:
            self.n_ctx = 5024  # Set context window for local Llama
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=self.n_ctx,
                n_threads=4,
                verbose=False,
            )
        self.model = model
        self.lang = lang

    def generate(self, messages: list[dict]) -> str:
        # Estimate prompt token count and set max_tokens to prevent exceeding context
        prompt_tokens = self._estimate_prompt_tokens(messages)
        # Reserve 512 tokens for generation, with a cap of 512 tokens
        reserved_for_generation = 512
        available_tokens = max(0, self.n_ctx - prompt_tokens - reserved_for_generation)
        max_tokens = min(512, available_tokens)
        
        if isinstance(self.llm, OpenAI):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.chat.completions.create(
                        messages=messages, 
                        temperature=0, 
                        model=self.model,
                        max_tokens=max_tokens
                    )
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    sleep(3)
            return response.choices[0].message.content
        else:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0,
                max_tokens=max_tokens
            )
            return response["choices"][0]["message"]["content"]
    
    def _estimate_prompt_tokens(self, messages: list[dict]) -> int:
        """Estimate the number of tokens in the prompt using tiktoken."""
        try:
            enc = tiktoken.encoding_for_model("gpt-4o")
            # Concatenate all message contents to estimate total tokens
            prompt_text = " ".join([msg.get("content", "") for msg in messages])
            return len(enc.encode(prompt_text))
        except Exception as e:
            logger.warning(f"Failed to estimate tokens using tiktoken: {e}. Using fallback estimation.")
            # Fallback: rough estimate of 4 characters per token
            prompt_text = " ".join([msg.get("content", "") for msg in messages])
            return len(prompt_text) // 4

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM