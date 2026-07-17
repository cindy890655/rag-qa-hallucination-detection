from rag.generator import LocalGenerator

generator = LocalGenerator()

prompt = """
Context:
Python is a high-level programming language.
It was created by Guido van Rossum.

Question:
Who invented Python?

Answer:
"""

answer = generator.generate(prompt)

print("\nGenerated Answer:\n")
print(answer)