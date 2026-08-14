"""
Drop-in replacement for the RAG project's LocalGenerator.

RAGPipeline only requires a generator that exposes `.tokenizer` and
`.generate(prompt, max_new_tokens)`, so an instance of this class can be handed
straight to the teammate's pipeline without modifying anything under "rag part".

Two deliberate differences from LocalGenerator, both needed to make the
large-scale config grid finish in reasonable time:

1.  The causal model is loaded explicitly onto a single device in fp16 instead
    of with device_map="auto". Once the FAISS index and the 390k chunks are
    resident, accelerate decides the model does not fit and silently offloads
    part of it to disk; every generated token then streams weights off the SSD.
    Measured on an M2 Pro with 16 GB: 80-195 s per question with
    device_map="auto", versus 2.5 s per question when loaded explicitly
    (mean over the 240-question k=3 grid run).

2.  The KV cache is released after every answer. Without that, generation on
    16 GB of unified memory degrades question after question (measured:
    19 s -> 44 s -> 58 s across three consecutive questions).

Neither change affects what the model says: same weights, same prompt, same
greedy decoding (do_sample=False).
"""
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


# Mirrors AVAILABLE_GENERATORS in the RAG project's config.py, so the grid
# runner can refer to a configuration by short name.
MODEL_ALIASES = {
    "flan-base": "google/flan-t5-base",
    "flan-large": "google/flan-t5-large",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
}

MAX_INPUT_TOKENS = 512


def resolve_model_name(name: str) -> str:
    """Accept either a short alias or a full Hugging Face model id."""
    return MODEL_ALIASES.get(name, name)


def pick_device() -> str:
    """Return the fastest device available on this machine."""
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


class FastGenerator:
    """
    Generate answers locally from a RAG prompt.

    Input:
        model_name: short alias ("phi3", "flan-base", ...) or a full model id
        device: override the automatic device choice
        verbose: print the prompt actually fed to the model (debugging only)

    Output:
        .generate(prompt, max_new_tokens) returns the answer as a string
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        verbose: bool = False
    ) -> None:
        self.model_name = resolve_model_name(model_name)
        self.verbose = verbose

        # One tokenizer for both model families. LocalGenerator only builds it
        # inside the causal branch, which makes its FLAN path raise
        # AttributeError; building it here keeps both families working.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if "phi-3" in self.model_name.lower():
            self.model_type = "causal"
            self.device = device or pick_device()

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.float16
                    if self.device != "cpu"
                    else torch.float32
                ),
                attn_implementation="eager"
            ).to(self.device)

        else:
            self.model_type = "seq2seq"

            # FLAN-T5 is numerically unstable in fp16 and small enough to be
            # fast on the CPU (~2.6 s per question for flan-base), so it stays
            # in fp32 on the CPU unless the caller asks otherwise.
            self.device = device or "cpu"

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name
            ).to(self.device)

        self.model.eval()

        dtype = next(self.model.parameters()).dtype
        print(
            f"FastGenerator ready: {self.model_name} "
            f"on {self.device} ({dtype})"
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64
    ) -> str:
        """
        Generate an answer from a RAG prompt.
        """
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        if self.model_type == "causal":
            return self._generate_causal(prompt, max_new_tokens)

        return self._generate_seq2seq(prompt, max_new_tokens)

    def _generate_causal(
        self,
        prompt: str,
        max_new_tokens: int
    ) -> str:
        """
        Generate with a decoder-only model such as Phi-3.
        """
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        prompt_length = inputs["input_ids"].shape[-1]

        if self.verbose:
            print(self.tokenizer.decode(
                inputs["input_ids"][0],
                skip_special_tokens=True
            ))

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Causal models return the prompt followed by the generated tokens.
        generated_ids = output_ids[0][prompt_length:]

        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        del output_ids, generated_ids
        self._release_cache()

        return answer.strip()

    def _generate_seq2seq(
        self,
        prompt: str,
        max_new_tokens: int
    ) -> str:
        """
        Generate with an encoder-decoder model such as FLAN-T5.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        if self.verbose:
            print(self.tokenizer.decode(
                inputs["input_ids"][0],
                skip_special_tokens=True
            ))

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

        del output_ids
        self._release_cache()

        return answer.strip()

    def _release_cache(self) -> None:
        """
        Free the KV cache so the next question starts from a clean allocator.
        """
        if self.device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Standalone smoke test: no retriever, no files written.
    PROMPT = (
        "Use only the context below to answer the question.\n\n"
        "Question:\nWhat is anarchism?\n\n"
        "Context:\n[Document 1]\nTitle: Anarchism\n"
        "Content: Anarchism is a political philosophy and movement that seeks "
        "to abolish all institutions that perpetuate authority, coercion, or "
        "hierarchy, primarily targeting the state and capitalism."
        "\n\nAnswer:"
    )

    generator = FastGenerator("phi3")

    for run in range(3):
        started = time.perf_counter()
        answer = generator.generate(PROMPT, max_new_tokens=64)
        elapsed = time.perf_counter() - started
        print(f"run {run + 1}: {elapsed:6.2f}s -> {answer[:60]}")