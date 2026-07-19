import torch
from config import MAX_NEW_TOKENS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class LocalGenerator:
    """
    Generate answers locally using a pretrained FLAN-T5 model.

    Input:
        model_name: Hugging Face model identifier

    Output:
        A natural-language answer generated from an input prompt
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base"
    ) -> None:
        print("Loading FLAN-T5...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.model.eval()

        print("FLAN-T5 loaded successfully!")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int =  MAX_NEW_TOKENS
    ) -> str:
        """
        Generate an answer from a RAG prompt.

        Input:
            prompt: context and question formatted as text
            max_new_tokens: maximum number of generated answer tokens

        Output:
            Generated answer as a string
        """
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

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