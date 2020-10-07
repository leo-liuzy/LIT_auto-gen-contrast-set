import pickle
import json
import os
import argparse
import numpy as np
from collections import defaultdict
from transfer.ergtransfer import get_tense
import torch
import random
from copy import deepcopy
from dl_pipeline.making_sense import uni_predict, bert_predict, ro_predict, xlnet_predict
from transformers import (
    OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    BertForMaskedLM, BertTokenizer,
    RobertaForMaskedLM, RobertaTokenizer,
    XLNetLMHeadModel, XLNetTokenizer,
)
from delphin.codecs import simplemrs, mrsjson

ORI = "original"

# Key: orig_label;1st_trans;2nd_trans, value: index of the rule
# * means the rule could apply to all orig_label, and final_label is the same
snli_rules = {
            "entailment;past simple;future simple": "1a",
            "contradiction;future simple;past simple": "1b",
            f"entailment;may;{ORI}": "2a",
            f"contradiction;may;{ORI}": "2b",
            "entailment;may;may": "3a",
            "contradiction;may;may": "3b",
            "*;passive: ARG2;passive: ARG2": "4",
            "*;it cleft: ARG1;it cleft: ARG1": "5",
            # compositional rules (so far we only consider combination of two phenomena)
            "entailment;past simple+it cleft: ARG1;future simple+it cleft: ARG1": "7a",
            "contradiction;future simple+it cleft: ARG1;past simple+it cleft: ARG1": "7b",
            "entailment;past simple+passive: ARG2;future simple+passive: ARG2": "8a",
            "contradiction;future simple+passive: ARG2;past simple+passive: ARG2": "8b",
            "entailment;modality: may+passive: ARG2;modality: may+passive: ARG2": "9a",
            "contradiction;modality: may+passive: ARG2;modality: may+passive: ARG2": "9b"
            }

mnli_rules = {
    # basic rules
    "entailment;past simple;future simple": "1a",
    "contradiction;future simple;past simple": "1b",
    f"entailment;may;{ORI}": "2a",
    f"contradiction;may;{ORI}": "2b",
    "entailment;may;may": "3a",
    "contradiction;may;may": "3b",
    "*;passive: ARG2;passive: ARG2": "4",
    "*;it cleft: ARG1;it cleft: ARG1": "5",
    # compositional rules (so far we only consider combination of two phenomena)
    "entailment;past simple+it cleft: ARG1;future simple+it cleft: ARG1": "7a",
    "contradiction;future simple+it cleft: ARG1;past simple+it cleft: ARG1": "7b",
    "entailment;past simple+passive: ARG2;future simple+passive: ARG2": "8a",
    "contradiction;future simple+passive: ARG2;past simple+passive: ARG2": "8b",
    "entailment;modality: may+passive: ARG2;modality: may+passive: ARG2": "9a",
    "contradiction;modality: may+passive: ARG2;modality: may+passive: ARG2": "9b"
}

preserving_rule_types = ["cleft", "passive"]
changing_rule_type = ["past", "future", "may"]

sts_rules = {}

tc_rules = {}

ALL_RULES = {
            "snli": snli_rules,
            "mnli": mnli_rules,
            "sts": sts_rules,
            "tc": tc_rules,
            }

MODEL_CLASSES = {
            "gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, uni_predict),
            "gpt2": (GPT2LMHeadModel, GPT2Tokenizer, uni_predict),
            "bert": (BertForMaskedLM, BertTokenizer, bert_predict),
            "roberta": (RobertaForMaskedLM, RobertaTokenizer, ro_predict),
            "xlnet": (XLNetLMHeadModel, XLNetTokenizer, xlnet_predict),
            }


def sample(data, sample):
    if sample <= 0 or sample > len(data):
        return data
    random.shuffle(data)
    return data[:sample]


def select_best(data, model, tokenizer, predictor):
    cleaned_data = []
    for idx, entry in enumerate(data):
        cleaned_entry = defaultdict(lambda: defaultdict(str))
        cleaned_entry["gold_label"] = entry["gold_label"]
        for sent in ["sentence1", "sentence2"]:
            if isinstance(entry[sent], str):
                cleaned_entry[sent][ORI] = entry[sent]
                continue
            for form, results in entry[sent].items():
                if any([(tense in form) for tense in ["future", "present", "past"]]) and not entry[sent][ORI][0]["aspect"] in form:
                    continue
                if not isinstance(results, str):
                    if len(results) > 1:
                        results = [r["surface"] for r in results]
                        scores = [predictor(r, model, tokenizer) for r in results]
                        results = results[np.argmax(scores)]
                    else:
                        results = results[0]["surface"]
                if not results:
                    continue
                cleaned_entry[sent][form] = results
        cleaned_data.append(cleaned_entry)
        print("processed " + str(idx) + "/" + str(len(data)), end="\r")
    return cleaned_data


def wirte_readable_text(cleaned_data, output_dir):
    with open(os.path.join(output_dir, "samples.txt"), "w") as file:
        for i, d in enumerate(cleaned_data):
            file.write("data: " + str(i) + "\n")
            for sent in ["sentence1", "sentence2"]:
                file.write(sent + "\n" + d[sent][ORI] + "\n\n")
                for k, v in d[sent].items():
                    if k != ORI:
                        file.write(k + ": \n" + v + "\n\n")


def apply_rules(cleaned_data, rules):
    final_data = []
    for idx, entry in enumerate(cleaned_data):
        final_entry = {}
        label = entry["gold_label"]
        sent1 = entry["sentence1"]
        sent2 = entry["sentence2"]
        assert type(sent1) is dict and type(sent2) is dict
        assert ORI in sent1 and ORI in sent2
        assert type(sent1[ORI]) is dict and type(sent2[ORI]) is dict

        final_entry["0"] = "\t".join([label, sent1[ORI]["surface"], sent2[ORI]["surface"]])
        if "genre" in entry:
            final_entry["genre"] = entry["genre"]

        # if one of the two sentence can not be parsed, then continue
        if len(sent1) == 1 or len(sent2) == 1:
            final_data.append(final_entry)
            continue
        assert "mrs" in sent1[ORI] and "mrs" in sent2[ORI]
        assert type(sent1[ORI]['mrs']) is dict and type(sent2[ORI]['mrs']) is dict
        # print(sent1.keys())
        # print(sent2.keys())

        mrs1 = mrsjson.from_dict(sent1[ORI]['mrs'])
        mrs2 = mrsjson.from_dict(sent2[ORI]['mrs'])

        for rule, ridx in rules.items():
            if ridx in ["1a", "1b"]:
                # if the current rule is tense, and the sentence pair are not both in present tense, move on
                if get_tense(mrs1) != "present" or get_tense(mrs2) != "present":
                    continue

            match1, match2 = None, None
            orig_label, trans1, trans2 = rule.split(";")

            for k1 in sent1.keys():
                phenomena = trans1.split("+")
                # if "+" in trans1:
                #     print("trans1")
                assert len(phenomena) <= 2
                if len(phenomena) == 2:
                    # so far, we consider one label-preserving phenomenon + one label-changing phenomenon
                    assert any(any(t in phenomenon for t in preserving_rule_types) for phenomenon in phenomena)

                if all(p in k1 for p in phenomena):
                    # make sure all the key words in this rule are in the sentence transformation name
                    match1 = k1
                    break

            for k2 in sent2.keys():
                phenomena = trans2.split("+")
                # if "+" in trans2:
                #     print("trans2")
                assert len(phenomena) <= 2
                if len(phenomena) == 2:
                    # so far, we consider one label-preserving phenomenon + one label-changing phenomenon
                    assert any(any(t in phenomenon for t in preserving_rule_types) for phenomenon in phenomena)

                if all(b in k2 for b in phenomena):
                    # make sure all the key words in this rule are in the sentence transformation name
                    match2 = k2
                    break
            if not (match1 and match2):
                # if we didn't find a matched transformation pair
                continue

            if orig_label == "*":
                new_label = label
            elif orig_label == label:
                new_label = "neutral"
            else:
                continue
            if match1 == ORI:
                s1 = sent1[ORI]["surface"]
            else:
                s1 = sent1[match1]

            if match2 == ORI:
                s2 = sent2[ORI]["surface"]
            else:
                s2 = sent2[match2]

            final_entry[ridx] = "\t".join([new_label, s1, s2])

        final_data.append(final_entry)
    return final_data


def read_jsonl(filepath: str):
    ret = []
    import json
    assert ".jsonl" in filepath
    print(f"Reading from: {filepath}")
    with open(filepath, "r") as f:
        for line in f:
            if line == "":
                continue
            ret.append(json.loads(line))
    return ret


def write_to_file(final_data, output_dir, divide_data):
    random.shuffle(final_data)
    if divide_data == "train":
        pickle.dump(final_data, open(os.path.join(output_dir, "train"), "wb"))
    elif divide_data == "dev":
        pickle.dump(final_data[:int(0.8 * len(final_data))], open(os.path.join(output_dir, "train"), "wb"))
        pickle.dump(final_data[int(0.8 * len(final_data)):], open(os.path.join(output_dir, "dev"), "wb"))
    elif divide_data == "test":
        pickle.dump(final_data[:int(0.8 * len(final_data))], open(os.path.join(output_dir, "train"), "wb"))
        pickle.dump(final_data[int(0.8 * len(final_data)):int(0.9 * len(final_data))], open(os.path.join(output_dir, "dev"), "wb"))
        pickle.dump(final_data[int(0.9 * len(final_data)):], open(os.path.join(output_dir, "test"), "wb"))


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        required=True,
        help="The path to the .pkl file containing original data and possible transformations.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory to put processed data files.",
    )
    
    # Other parameters
    parser.add_argument(
        "--model_type",
        default="gpt2",
        type=str,
        help="Type of the model used to select best sentence in a category, choose from gpt, gpt2, bert, roberta, xlnet.",
    )
    parser.add_argument(
        "--model_name",
        default="gpt2",
        type=str,
        help="Name of the model used to select best sentence in a category.",
    )
    parser.add_argument(
        "--divide_data",
        default="train",
        type=str,
        help="Select from [train, dev, test]. If train, generate train only. If dev, train+dev. If test, all three.",
    )
    parser.add_argument(
        "--sample",
        default=-1,
        type=int,
        help="If true, generate train, dev and test. Else, generate only the first two.",
    )
    parser.add_argument(
        "--write_text",
        default=False,
        type=bool,
        help="Whether to write a readable text file of selected sentences.",
    )
    parser.add_argument(
        "--task",
        default="snli",
        type=str,
        help="One of snli, sts, tc, mnli",
    )
    SPLITS = ["test", "dev"]
    
    args = parser.parse_args()
    
    if os.path.isdir(args.data_path):
        paths = [os.path.join(args.data_path, filename) for filename in os.listdir(args.data_path)][::-1]
    else:
        paths = [args.data_path]
    
    # data = []
    # for path in paths:
    #     with open(path) as file:
    #         for line in file:
    #             data.append(json.loads(line.strip()))
    # print(len(data))

    # model_class, tokenizer_class, predictor = MODEL_CLASSES[args.model_type]
    # model = model_class.from_pretrained(args.model_name)
    # model.to("cuda")
    # model.eval()
    # tokenizer = tokenizer_class.from_pretrained(args.model_name)
    
    # data = sample(data, args.sample)
    # cleaned_data = select_best(data, model, tokenizer, predictor)
    # if args.write_text:
    #     wirte_readable_text(cleaned_data, args.output_dir)
    basic_rules = ["1a", "1b", "2a", "2b", "3a", "3b", "4", "5"]
    comp_rules = ["7a", "7b", "8a", "8b", "9a", "9b"]
    import json
    rules = {v: k for k, v in json.load(open("/home/lzy/proj/AnalyzingNeuralLMs/datasets/final_snli/rules.json")).items()}
    sent1_trans = [r.split(";")[1] for r in rules.values()]
    sent2_trans = [r.split(";")[2] for r in rules.values()]
    for path in paths:
        if not any(s in path for s in SPLITS):
            continue
        index = [split in path for split in SPLITS].index(True)
        result_filename = SPLITS[index]

        cleaned_data = read_jsonl(path)
        sent1_stats = {}
        sent2_stats = {}
        for d in cleaned_data:
            sent1 = d["sentence1"]
            sent2 = d["sentence2"]
            for k in sent1.keys():
                if k not in sent1_stats:
                    sent1_stats[k] = 0
                sent1_stats[k] += 1

            for k in sent2.keys():
                if k not in sent2_stats:
                    sent2_stats[k] = 0
                sent2_stats[k] += 1

        print(f"sentence1: {sent1_stats}")
        print(f"sentence2: {sent2_stats}")
        print()
        final_data = apply_rules(cleaned_data, ALL_RULES[args.task])
        pickle.dump(final_data, open(os.path.join(args.output_dir, result_filename), "wb"))

        # now we created dataset, we then split the dataset into basic and compositional
        basic_data = []
        comp_data = []
        for d in final_data:
            assert "0" in d
            basic_d = {"0": d["0"]}
            comp_d = {"0": d["0"]}
            if "genre" in d:
                basic_d["genre"] = d["genre"]
                comp_d["genre"] = d["genre"]
            if "genre" not in d and len(d) == 1 or "genre" in d and len(d) == 2:
                basic_data.append(basic_d)
                comp_data.append(comp_d)
                continue

            for r in basic_rules:
                if r in d:
                    basic_d[r] = d[r]
            basic_data.append(basic_d)

            for r in comp_rules:
                if r in d:
                    comp_d[r] = d[r]
            comp_data.append(comp_d)
        assert len(basic_data) == len(comp_data) == len(final_data)
        pickle.dump(basic_data, open(os.path.join(f"{args.output_dir}/basic/", result_filename), "wb"))
        pickle.dump(comp_data, open(os.path.join(f"{args.output_dir}/compositional/", result_filename), "wb"))

    # write_to_file(final_data, args.output_dir, args.divide_data)
    import json
    # json.dump(ALL_RULES[args.task], open(f"{args.output_dir}/rules.json", "w"))


if __name__ == "__main__":
    main()

