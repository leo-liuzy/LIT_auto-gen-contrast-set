import csv
import json
import os
import copy
import argparse
from tqdm import tqdm
from ergtransfer import transfer

parser = argparse.ArgumentParser("Generating sentences using ERG")
parser.add_argument("--data_dir", type=str,
                    default=f"{os.getcwd()}/../datasets/stsbenchmark")
parser.add_argument("--target_data_dir", type=str,
                    default=f"{os.getcwd()}/../datasets/aug_stsbenchmark")
parser.add_argument("--tenses", nargs='+', default=[])
parser.add_argument("--modalities", nargs='+', default=[])
parser.add_argument("--progs", nargs='+', default=[])
parser.add_argument("--perfs", nargs='+', default=[])
parser.add_argument("--grm_path", type=str, default='erg-1214.dat')
parser.add_argument("--ace_path", type=str, default=f'{os.getcwd()}/ace')
parser.add_argument("--checkpoint_period", type=int, default=500)
parser.add_argument("--timeout", type=int, default=5)


def read_sts(file_path):
    """

    :param data_dir: path to snli directory
    :return: return a dictionary like <"train": [train examples]>
    """
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, line in enumerate(reader):
            if len(line) != 7:
                continue
            added_ohj = {}
            added_ohj['genre'] = line[0]
            added_ohj['file'] = line[1]
            added_ohj['year-split'] = line[2]
            added_ohj['infileID'] = line[3]
            added_ohj['gold_label'] = line[4]
            added_ohj['sentence1'] = line[5]
            added_ohj['sentence2'] = line[6]
            added_ohj['pairID'] = i
            data.append(added_ohj)

    return data


def parse_file(x):
    args, filename, _ = x
    data = read_sts(f"{args.data_dir}/{filename}")
    statistics = {'unparsed_sentence': 0, "timeout": 0}
    aug_data = []
    target_file_name = f"{args.target_data_dir}/aug_{filename}"

    for i, datum in enumerate(data):
        print(f"{filename}, i {i}")
        aug_datum = copy.deepcopy(datum)
        for idx in ['1', '2']:
            sentence = datum[f'sentence{idx}']
            transforms = transfer(sentence, args.grm_path, args.ace_path,
                                  timeout=args.timeout,
                                  tenses=args.tenses,
                                  progs=args.progs,
                                  perfs=args.perfs)
            if len(transforms) <= 1:
                statistics["unparsed_sentence"] += 1
                continue
            elif "timeout" in transforms:
                statistics["timeout"] += 1
                continue

            aug_datum[f'sentence{idx}'] = transforms
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

    args = parser.parse_args()
    print("Reading all STSBenchmark data splits")
    if not os.path.exists(args.target_data_dir):
        os.mkdir(args.target_data_dir)
    data_file_prefix = "sts"
    splits = ["train", "dev", "test"]
    num_division = 5
    pool = multiprocessing.Pool(processes=num_division)
    import sys
    sys.stderr = open('err.txt', 'w')
    for split in tqdm(splits):
        parallel_inputs = [
            (args, f"{data_file_prefix}-{split}_0{i}.csv", i) for i in range(num_division)]
        outputs = pool.map(parse_file, parallel_inputs)
        assert all(outputs)

    print("Finish")
