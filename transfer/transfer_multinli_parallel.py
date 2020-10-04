import json
import os
import copy
import argparse
from tqdm import tqdm
from ergtransfer import transfer, get_best_parse
from delphin.codecs import mrsjson

from ergtransfer import transfer

GENRES = ["fiction", "government", "slate", "telephone", "travel"]

parser = argparse.ArgumentParser("Generating sentences using ERG")
parser.add_argument("--data_dir", type=str,
                    default=f"{os.getcwd()}/../datasets/multinli_1.0_split_continued")
parser.add_argument("--target_data_dir", type=str,
                    default=f"{os.getcwd()}/../datasets/aug_multinli_1.0_split_continued")

parser.add_argument("--tenses", nargs='+', default=[])
parser.add_argument("--modalities", nargs='+', default=[])
parser.add_argument("--progs", nargs='+', default=[])
parser.add_argument("--perfs", nargs='+', default=[])
parser.add_argument("--grm_path", type=str,
                    default=f"{os.getcwd()}/erg-1214-linux-64-0.9.30.dat")
parser.add_argument("--ace_path", type=str,
                    default=f"{os.getcwd()}/ace-0.9.30/ace")
parser.add_argument("--checkpoint_period", type=int, default=500)
parser.add_argument("--timeout", type=int, default=5)


def read_multinli(file_path):
    """

    :param data_dir: path to snli directory
    :return: return a dictionary like <"train": [train examples]>
    """
    extracted_fields = ['gold_label',
                        'sentence1', 'sentence2', 'pairID']
    data = []
    with open(file_path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            added_obj = {}
            for field in extracted_fields:
                added_obj[field] = json_obj[field]
            data.append(added_obj)
    return data


def parse_file(args, filename, worker_idx):
    x = args, filename, worker_idx
    data = read_multinli(f"{args.data_dir}/{filename}")
    statistics = {'unparsed_sentence': 0, "timeout": 0}
    aug_data = []
    target_file_name = f"{args.target_data_dir}/aug_{filename}"

    for i, datum in enumerate(data):
        print(f"Worker {worker_idx}, i {i}")
        aug_datum = copy.deepcopy(datum)
        for idx in ['1', '2']:
            sentence = datum[f'sentence{idx}']
            # transfer now returns a dict incl. 'original': best_parse (Result)
            try:
                transforms = transfer(sentence, args.grm_path, args.ace_path,
                                      timeout=args.timeout,
                                      tenses=args.tenses,
                                      progs=args.progs,
                                      perfs=args.perfs)
            except:
                statistics["unparsed_sentence"] += 1
                continue

            if len(transforms) <= 1:
                statistics["unparsed_sentence"] += 1
                continue
            elif "timeout" in transforms:
                statistics["timeout"] += 1
                continue

            aug_datum[f'sentence{idx}'] = transforms
            parse = get_best_parse(sentence, args.grm_path, args.ace_path, args.timeout)
            if parse is None:
                continue
            aug_datum[f'sentence{idx}']["original"] = {"surface": sentence, "mrs": mrsjson.to_dict(parse.mrs())}
        aug_data.append(aug_datum)

        if (i + 1) % args.checkpoint_period == 0:
            print(f"Worker {idx} saving ....")
            with open(target_file_name, 'w') as outfile:
                for aug_datum in aug_data:
                    json.dump(aug_datum, outfile)
                    outfile.write('\n')

    with open(target_file_name, 'w') as outfile:
        for aug_datum in aug_data:
            json.dump(aug_datum, outfile)
            outfile.write('\n')
    with open(f"{args.target_data_dir}/"
              f"parse_stats_{'.'.join(filename.split('.')[:-1])}.json", 'w') as outfile:
        json.dump(statistics, outfile)

    return True


if __name__ == "__main__":
    import multiprocessing
    import numpy as np

    args = parser.parse_args()

    print("Reading all MultiNLI 1.0 data splits")
    if not os.path.exists(args.target_data_dir):
        os.mkdir(args.target_data_dir)
    data_file_prefix = "multinli_1.0"
    splits = ["train"]
    file_idxs = [0, 1, 2, 3, 4, 5, 6, 7]
    num_division = len(file_idxs)
    pool = multiprocessing.Pool(processes=num_division)
    import sys
    # sys.stderr = open('err.txt', 'w')
    for split in tqdm(splits[:1]):
        parallel_inputs = [
            (args, f"{data_file_prefix}_{split}_{genre}.jsonl", i) for i, genre in enumerate(GENRES)]
        outputs = pool.starmap(parse_file, parallel_inputs)
        assert all(outputs)

    print("Finish")
