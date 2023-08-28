import argparse
import json
import re
from typing import Dict, List

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize


def get_bleu(ref_sents: List[str], gen_sent: str, ngrams=4) -> float:
    hypothesis = word_tokenize(gen_sent.strip().lower())
    references = [word_tokenize(ref_sent.strip().lower()) for ref_sent in ref_sents]
    weights = [round(1/ngrams, 2) for _ in range(ngrams)] + ([0] * (4-ngrams))
    bleu = sentence_bleu(
        references=references,
        hypothesis=hypothesis,
        weights=tuple(weights)
        )
    return bleu


def get_meteor(ref_sents: List[str], gen_sent: str) -> float:
    hypothesis = word_tokenize(gen_sent.strip().lower())
    references = [word_tokenize(ref_sent.strip().lower()) for ref_sent in ref_sents]
    meteor = nltk.translate.meteor(references, hypothesis)
    return meteor


def calc_scores(data: Dict[List[str]]):
    regex_clean = re.compile(r'The answer is:\n\n?(>.?)\n\n')
    overall_bleu4, overall_bleu2 = 0, 0
    overall_bleu1 = 0
    meteor = 0
    
    golds, preds = data['outputs'], data['preds']
    for gold, hyp in zip(golds, preds):
        hyp = re.sub(r'.*?Answer:', '', hyp).replace("ASSISTANT", "").strip()
        
        if regex_clean.search(hyp):
            hyp = regex_clean.findall(hyp)[0].strip()
        else:
            hyp = hyp[:hyp.find("\n\n")].strip()
        
        overall_bleu1 += get_bleu(gold, hyp.replace("\n", " "), 1)
        overall_bleu2 += get_bleu(gold, hyp.replace("\n", " "), 2)
        overall_bleu4 += get_bleu(gold, hyp.replace("\n", " "), 4)
        meteor += get_meteor(gold, hyp)

    print(f"BLEU1 - {round(overall_bleu1*100/len(data), 2)}")
    print(f"BLEU2: {round(overall_bleu2*100/len(data), 2)}")
    print(f"BLEU4: {round(overall_bleu4*100/len(data), 2)}")
    print(f"METEOR - {round(meteor*100/len(data), 2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", type=str)
    args = parser.parse_args()

    with open(args.predictions_file, "r") as f:
        data = json.load(f)

