from transformers import pipeline

# load a pretrained NLI model (downloads on first run)
nli = pipeline("text-classification", model="facebook/bart-large-mnli")

# a quick manual test
premise = "Arthur's Magazine was started in 1844."
hypothesis = "Arthur's Magazine was started first."
result = nli(f"{premise} </s></s> {hypothesis}")
print(result)