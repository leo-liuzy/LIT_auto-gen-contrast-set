import json
import os
import copy
import argparse
from tqdm import tqdm
from transfer import transfer

parser = argparse.ArgumentParser("Generating sentences using ERG")
parser.add_argument("--data_dir", type=str, default=f"{os.getcwd()}/../datasets/snli_1.0")
parser.add_argument("--target_data_dir", type=str, default=f"{os.getcwd()}/../datasets/aug_snli_1.0")
parser.add_argument("--tenses", nargs='+', default=['past', 'pres', 'fut'])
parser.add_argument("--modalities", nargs='+', default=['_may_v_modal'])
parser.add_argument("--progs", nargs='+', default=['+', '-'])
parser.add_argument("--grm_path", type=str, default=f"{os.getcwd()}/erg-1214-linux-64-0.9.30.dat")
parser.add_argument("--ace_path", type=str, default=f"{os.getcwd()}/ace-0.9.30/ace")
parser.add_argument("--checkpoint_period", type=int, default=200)
parser.add_argument("--timeout", type=int, default=3)


def read_snli(file_path):
    """

    :param data_dir: path to snli directory
    :return: return a dictionary like <"train": [train examples]>
    """
    extracted_fields = ['gold_label',
                        'sentence1', 'sentence2', 'pairID', 'captionID']
    data = []
    with open(file_path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            added_obj = {}
            for field in extracted_fields:
                added_obj[field] = json_obj[field]
            data.append(added_obj)
    return data[:3]


def parse_file(x):
    args, filename, workerid = x
    data = read_snli(f"{args.data_dir}/{filename}")
    statistics = {'unparsed_sentence': 0, "timeout": 0}
    aug_data = []
    target_file_name = f"{args.target_data_dir}/aug_{filename}"

    for i, datum in enumerate(data):
        aug_datum = copy.deepcopy(datum)
        # print(f"Worker: {workerid}, {i}")
        for idx in ['1', '2']:
            sentence = datum[f'sentence{idx}']
            aug_sentence = {'original': sentence}
            transforms = transfer(sentence, args.grm_path, args.ace_path,
                                  timeout=args.timeout,
                                  tenses=args.tenses,
                                  progs=args.progs, perfs=[])
            if len(transforms) == 0:
                statistics["unparsed_sentence"] += 1
                continue
            elif "timeout" in transforms:
                statistics["timeout"] += 1
                continue

            for trans_type in transforms:
                aug_sentence[trans_type] = [t['surface'] for t in transforms[trans_type]]

            aug_datum[f'sentence{idx}'] = aug_sentence
        aug_data.append(aug_datum)

        if (i + 1) % args.checkpoint_period == 0:
            print(f"{i + 1}th datum: Worker {workerid} saving.....")
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

    args = parser.parse_args()
    print("Reading all SNLI 1.0 data splits")
    if not os.path.exists(args.target_data_dir):
        os.mkdir(args.target_data_dir)
    data_file_prefix = "snli_1.0"
    splits = ["train", "dev", "test"]
    num_division = 20
    pool = multiprocessing.Pool(processes=num_division)
    import sys
    sys.stderr = open('err.txt', 'w')
    for split in tqdm(splits[:1]):
        parallel_inputs = [(args, f"{data_file_prefix}_{split}_0{i}.jsonl", i) for i in range(num_division)]
        outputs = pool.map(parse_file, parallel_inputs)
        assert all(outputs)

    print("Finish")
