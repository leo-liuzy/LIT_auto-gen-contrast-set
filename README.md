# LIT: auto-generator for contrast set

This is the code for out EMNLP 2020 paper: Linguistically-Informed Transformations (LIT): A Method forAutomatically Generating Contrast Sets



## Citation

Coming soon

## Processed Dataset

* We released the [datasets](https://drive.google.com/drive/folders/1qkqOg1Lk9VcaI5l-l1d6RHE_CBCTXl-A?usp=sharing) used in our paper. 
* We will also release the MRS parses with which people can transform sentences by their defined perturbation.

**Note:** This is **not** complete parallel datasets of the original SNLI and MNLI. There are some sentences missing because the parser sometimes can’t parse the representation. You might need to run your transformation on some missed out data.



## Environment setup

```
conda env create -f environment.yml
conda activate lit
```



## Structure of Repo

`transfer`: module that contain all functions we mentioned in our paper. Within it, :

* `README.md` gives a detailed documentation of current config of our parser.

* `transfer_example.py` is an illustrative example of how to use our parser.
* `transfer_snli_parallel.py` is the script we used (some local modification needs to be made) to parse SNLI in parallel. Parallel processing is **strongly encouraged**

`post-process`: after processing the dataset, you need some cleaning of the parsed dataset to put in the right form.

* `making_sense.py` contains choices of sentence selectors in scoring different generated surface sentences
*  `process.py` contains functions that:
  * select generated sentences
  * apply defined rules to generate contrast set

