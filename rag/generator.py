import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from config import GENERATOR_MODEL, MAX_NEW_TOKENS


class LocalGenerator:
    """
    Generate answers locally using either:

    - Sequence-to-sequence models such as FLAN-T5
    - Causal language models such as Phi-3
    """

    def __init__(
        self,
        model_name: str = GENERATOR_MODEL
    ) -> None:
        self.model_name = model_name

        print(f"Loading generator model: {model_name}")

        # Phi-3 is a decoder-only causal language model.
        if "phi-3" in model_name.lower():
            self.model_type = "causal"

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="eager"
            )

        # FLAN-T5 is an encoder-decoder sequence-to-sequence model.
        else:
            self.model_type = "seq2seq"

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name
            )

        self.model.eval()

        print("Generator model loaded successfully!")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS
    ) -> str:
        """
        Generate an answer from a RAG prompt.
        """
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        if self.model_type == "causal":
            return self._generate_causal(
                prompt=prompt,
                max_new_tokens=max_new_tokens
            )

        return self._generate_seq2seq(
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )

    def _generate_seq2seq(
        self,
        prompt: str,
        max_new_tokens: int
    ) -> str:
        """
        Generate text with FLAN-T5-style sequence-to-sequence models.
        """
        token_count = len(
            self.tokenizer.encode(
                prompt,
                add_special_tokens=True
            )
        )
        print(f"Prompt tokens before truncation: {token_count}")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        decoded_input = self.tokenizer.decode(
            inputs["input_ids"][0],
            skip_special_tokens=True
        )

        print("\n========== ACTUAL MODEL INPUT ==========")
        print(decoded_input)
        print("========================================\n")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        answer = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        return answer.strip()

    def _generate_causal(
        self,
        prompt: str,
        max_new_tokens: int
    ) -> str:
        """
        Generate text with Phi-3-style causal language models.
        """
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        prompt_length = inputs["input_ids"].shape[-1]

        print(f"Prompt tokens before generation: {prompt_length}")

        decoded_input = self.tokenizer.decode(
            inputs["input_ids"][0],
            skip_special_tokens=True
        )

        print("\n========== ACTUAL MODEL INPUT ==========")
        print(decoded_input)
        print("========================================\n")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Causal models return both the input prompt and generated tokens.
        generated_ids = output_ids[0][prompt_length:]

        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        return answer.strip()