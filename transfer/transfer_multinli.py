import json
import os
import copy
import argparse
import pickle
from tqdm import tqdm
from ergtransfer import transfer

parser = argparse.ArgumentParser("Generating sentences using ERG")
parser.add_argument("--data_dir", type=str, default=f"{os.getcwd()}/../datasets/snli_1.0")
parser.add_argument("--target_data_dir", type=str, default=f"{os.getcwd()}/../datasets/aug_snli_1.0")
parser.add_argument("--tenses", nargs='+', default=['past', 'pres', 'fut'])
parser.add_argument("--modalities", nargs='+', default=['_may_v_modal'])
parser.add_argument("--progs", nargs='+', default=['+', '-'])
parser.add_argument("--grm_path", type=str, default=f"{os.getcwd()}/erg-1214-linux-64-0.9.30.dat")
parser.add_argument("--ace_path", type=str, default=f"{os.getcwd()}/ace-0.9.30/ace")
parser.add_argument("--checkpoint_period", type=int, default=500)
parser.add_argument("--timeout", type=int, default=5)


def read_snli(data_dir):
    """

    :param data_dir: path to snli directory
    :return: return a dictionary like <"train": [train examples]>
    """
    ret = {}
    prefix = data_dir.split('/')[-1] + '_'
    suffix = '.jsonl'
    extracted_fields = ['gold_label',
                        'sentence1', 'sentence2', 'pairID']
    for split in tqdm(['train', 'test', 'dev']):
        file_path = f"{data_dir}/{prefix + split + suffix}"
        data = []
        with open(file_path, "r") as f:
            for line in f:
                json_obj = json.loads(line)
                added_ohj = {}
                for field in extracted_fields:
                    added_ohj[field] = json_obj[field]
                data.append(added_ohj)
        ret[split] = data
    return ret


if __name__ == "__main__":
    args = parser.parse_args()
    print("Reading all SNLI 1.0 data splits")
    datasets = read_snli(args.data_dir)
    target_file_name_prefix = args.target_data_dir.split('/')[-1]
    target_file_name_suffix = '.jsonl'
    statistics = {}
    if not os.path.exists(args.target_data_dir):
        os.mkdir(args.target_data_dir)

    for split in tqdm(['train', 'test', 'dev']):
        data = datasets[split]
        statistics[split] = {'unparsed_sentence': 0, "timeout": 0}
        aug_data = []
        for i, datum in enumerate(data):
            aug_datum = copy.deepcopy(datum)
            for idx in ['1', '2']:
                print(f"{split}   i: {i}  sent_idx: {idx}")
                sentence = datum[f'sentence{idx}']
                aug_sentence = {'original': sentence}
                transforms = transfer(sentence, args.grm_path, args.ace_path,
                                      timeout=args.timeout,
                                      tenses=args.tenses,
                                      progs=args.progs, perfs=[])
                if len(transforms) == 0:
                    statistics[split]["unparsed_sentence"] += 1
                    continue
                elif "timeout" in transforms:
                    statistics[split]["timeout"] += 1
                    continue

                for trans_type in transforms:
                    aug_sentence[trans_type] = []
                    for t in transforms[trans_type]:
                        aug_sentence[trans_type].append(dict(t))
                aug_datum[f'sentence{idx}'] = aug_sentence
            aug_data.append(aug_datum)

            if (i+1) % args.checkpoint_period == 0:
                with open(f"{os.getcwd()}/aug_data.pkl", 'wb') as outfile:
                    pickle.dump(statistics, outfile)

                with open(f"{args.target_data_dir}/parse_stats.pkl", 'w') as outfile:
                    json.dump(statistics, outfile)

        target_file_name = f"{args.target_data_dir}/" \
                           f"{target_file_name_prefix + split + target_file_name_suffix}"
        # write data
        with open(target_file_name, 'w') as outfile:
            for aug_datum in aug_data:
                json.dump(aug_datum, outfile)
                outfile.write('\n')

    with open(f"{args.target_data_dir}/parse_stats.pkl", 'w') as outfile:
        json.dump(statistics, outfile)

    print("Finish")
