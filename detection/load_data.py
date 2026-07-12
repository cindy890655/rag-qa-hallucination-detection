import json


def load_halueval_qa(filepath):
    """
    Load the HaluEval QA data file.
    Input: filepath - path to the json file (string)
    Output: data - a list where each element is one record (dict) with
            keys: knowledge / question / right_answer / hallucinated_answer
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:                    # each line is one independent json record
            line = line.strip()
            if line:                      # skip empty lines
                data.append(json.loads(line))
    return data


# quick check
if __name__ == "__main__":
    data = load_halueval_qa("data/qa_data.json")
    print("Total records:", len(data))
    print("First record:")
    print(data[0])