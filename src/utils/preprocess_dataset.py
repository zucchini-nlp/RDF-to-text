import argparse
import json
import re
from typing import List, Dict

from hdt import HDTDocument

#kb = HDTDocument("/data/dbpedia_2015_10.hdt")
dbpedia_prefizes = ["http://dbpedia.org/ontology", "http://dbpedia.org/property", "http://dbpedia.org/resource"]


def execute_sparql_to_triplets(sparql: str, count: bool=True):
    if "COUNT" in query:
        count = True
        ans = 0
    body = re.findall(r'\{(.*)\}', sparql)
    triplets_for_hdt = []
    triplets, unk_triplets = [], []
    for query_triplet in body[0].split(". "):
        if not query_triplet:
            break
        triplets_for_hdt.append(tuple([el.strip("<>") for el in query_triplet.split()]))
        triplet = [el.strip("<>").split("/")[-1] for el in query_triplet.split()]
        sub, rel, obj = triplet
        sub = sub.replace("_", " ")
        obj = obj.replace("_", " ")
        if rel.endswith("#type"):
            rel = "type"
        if not any([(sub==unk or obj==unk) for unk in ["?x", "?uri"]]):
            triplets.append((sub, rel, obj))
        else:
            unk_triplets.append([sub, rel, obj])
    kb_ans = kb.search_join(triplets_for_hdt)
    N = len(list(kb_ans))
    for idx, res in enumerate(kb_ans):
        filled_triplets = unk_triplets.copy()
        for found in res:
            unk_uri, unk_uri_lbl = found[0], found[1]
            unk_uri_lbl = unk_uri_lbl.split("/")[-1].replace("_", " ")
            filled_triplets = [tuple(el.replace(unk_uri, unk_uri_lbl) for el in triplet) for triplet in filled_triplets]
        if count and unk_uri == "?uri":
            ans += 1
            filled_triplets = [
                tuple(
                    el[0].replace(unk_uri, str(ans)),
                    el[1]+"count",
                    el[2].replace(unk_uri, str(ans))) 
                for triplet in filled_triplets if "?uri" in triplet]

            if idx != N-1:
                continue
        triplets += filled_triplets
    return set(triplets)


def linearize_triplets(triplets: List[str]) ->  str:
    linearized = ""
    for t in triplets:
        linearized += f"{str(tuple(t))};"
    return linearized


def process_vquanda(file: str):
    with open(file, "r") as f:
        vquanda = json.load(f)

    dataset = []
    for el in vquanda:
        question = el['question']
        if "List all" in question:
            question = question.replace("List all ", "Which are ")
        gold = el['verbalized_answer'].replace("[", "").replace("]", "")
        triplets = execute_sparql_to_triplets(el['query'])
        dataset.append({
            "question": question,
            "input": linearize_triplets([tuple(t) for t in triplets]),
            "output": [gold]
        })
    return dataset


def process_lcquad(file: str):
    with open(file, "r") as f:
        examples = f.readlines()

    dataset = []
    for i in range(0, len(examples), 5):
        el = examples[i: i+4]
        question, triplets, triplet_lbls, ans = el[0], el[1], el[2], el[3]
        question = question.replace("question:", "").strip()
        if "List all" in question:
            question = question.replace("List all ", "Which are ")
        triplets = eval(triplets.replace("triplets:", "").strip())
        triplets_lbls = eval(triplet_lbls.replace("triplets with labels:", "").strip())
        gold_answers = eval(ans.replace("answer:", "").strip())
        dataset.append({
            "question": question,
            "input": linearize_triplets([tuple(t) for t in triplets_lbls]),
            "output": gold_answers
        })
    return dataset


def process_data(datapath_list: List[str]):
    dataset = []
    for file in datapath_list:
        if "vquanda" in file: 
            dataset.extend(process_vquanda(file))
        elif "lcquad" in file:
            dataset.extend(process_lcquad(file))
    return dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset.")
    parser.add_argument(
        "--raw_data_filepaths",
        metavar="U",
        type=str,
        nargs="+",
        default=["data/lcquad_examples.txt"],
    )
    parser.add_argument("--output_file", type=str, default="data/lcquad_examples_processed.json")
    args = parser.parse_args()

    data_filepaths = args.raw_data_filepaths    
    dataset = process_data(data_filepaths)

    with open(args.output_file, "w") as f:
        json.dump(dataset, f, indent=4)
    print("The datasets has been processed")
