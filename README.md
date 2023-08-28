# RDF to text generartion with LLMs

This repository contains code for generating text from RDF triplets. Being able to generate good quality text from RDF data would permit e.g., making this data more accessible to lay users, enriching existing text with information drawn from knowledge bases such as DBpedia or describing, comparing and relating entities present in these knowledge bases. Along with simple RDF to text generation, the repository can be used to generate long form answers to questions based on triplets as context.

The following datasets are used here:

- [Web NLG:](https://synalp.gitlabpages.inria.fr/webnlg-challenge/) The WebNLG corpus comprises of sets of triplets describing facts (entities and relations between them) and the corresponding facts in form of natural language text. The corpus contains sets with up to 7 triplets each along with one or more reference texts for each set.
- [VQUANDA](https://github.com/AskNowQA/VQUANDA): A KBQA dataset containing verbalizations of answers. The dataset is based on LC-QuAD which uses DBpedia v04.16 as the target KB. The dataset contains 5000 examples split into train (80%) and test (20%) sets.
- [LC-QUaD](https://github.com/AskNowQA/LC-QuAD): We release, and maintain a gold standard KBQA (Question Answering over Knowledge Base) dataset containing 5000 Question and SPARQL queries. For the task of RDF to text 100 examples from the dataset were chosen and manually annotated.



## Models

The repository currently supports the following models but can be adapted to other LLMs:

* Vicuna
* Oasst pythia
* Bloomz
* GPT-J


## Usage

First clone the repos and install the requirements.
```
git clone https://github.com/zucchini-nlp/RDF-to-text.git
cd RDF-to-text.git
pip install -r requirements.txt 
```

Then to fine-tune the models and predict, use the following commands

```bash
sh train.sh
sh predict.sh
```

The generated answers will be saved under the data directory by default. To calculate BLEU and METEOR metrics on the generated data, run the following command

```
python eval/calc_metrics.py --predictions_file <predictions_file>
```