import json

if __name__ == "__main__":
    # load bidirectional test results
    with open("data/failed_fwd_rxns.jsonl", "r") as f:
        failed_rxns = [json.loads(line) for line in f]
    success = []
    for rxn in failed_rxns:
        rxn_smiles = rxn["rxn_smiles"]
        template = rxn["template"]
        error = rxn["error"]
        if error.startswith("no error"):
            success.append({"rxn_smiles": rxn_smiles, "template": template})
    with open("data/filtered_fwd_train.jsonl", "w") as f:
        for rxn in success:
            f.write(json.dumps(rxn) + "\n")
