"""vLLM inference utilities"""

from typing import List, Optional

from vllm import LLM, SamplingParams

from searchlm.config import get_config


class VllmEngine:
    """vLLM engine for inference"""

    def __init__(
        self, model_name: Optional[str] = None, max_model_len: Optional[int] = None
    ):
        """
        Initialize the vLLM engine.

        Args:
            model_name: Model name to load (defaults to config.model.name)
            max_model_len: Maximum model length (defaults to config.model.max_model_len)
        """
        config = get_config()
        self.model_name = model_name or config.model.name
        self.max_model_len = max_model_len or config.model.max_model_len
        self.llm = LLM(model=self.model_name, max_model_len=self.max_model_len)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate completions for prompts.

        Args:
            prompts: List of prompts to generate completions for
            temperature: Sampling temperature (defaults to config.model.temperature)
            max_tokens: Maximum tokens to generate (defaults to config.model.max_tokens)
            **kwargs: Additional arguments to pass to SamplingParams

        Returns:
            List of generated text completions
        """
        sampling_params = SamplingParams(
            temperature=temperature or self.config.model.temperature,
            max_tokens=max_tokens or self.config.model.max_tokens,
            **kwargs,
        )

        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
