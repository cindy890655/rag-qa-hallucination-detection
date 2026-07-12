from load_data import load_halueval_qa


def build_samples(data):
    """
    Convert each HaluEval QA record into two labeled samples.
    Input: data - list of records, each with keys
           knowledge / question / right_answer / hallucinated_answer
    Output: samples - list of dicts, each with keys
            question / knowledge / answer / label
            label = 0 means faithful (not hallucination)
            label = 1 means hallucinated
    """
    samples = []
    for record in data:
        # the correct answer -> label 0 (not hallucination)
        samples.append({
            "question": record["question"],
            "knowledge": record["knowledge"],
            "answer": record["right_answer"],
            "label": 0,
        })
        # the hallucinated answer -> label 1 (hallucination)
        samples.append({
            "question": record["question"],
            "knowledge": record["knowledge"],
            "answer": record["hallucinated_answer"],
            "label": 1,
        })
    return samples


# quick check
if __name__ == "__main__":
    data = load_halueval_qa("data/qa_data.json")
    samples = build_samples(data)
    print("Total samples:", len(samples))
    print("A faithful sample (label 0):")
    print(samples[0])
    print("A hallucinated sample (label 1):")
    print(samples[1])