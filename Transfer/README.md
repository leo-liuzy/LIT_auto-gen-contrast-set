# Transfer module

includes

- pre-compiled ERG 1214 grammar files for linux and macos
- pre-compiled ACE parser for linux and macos
    - needs to be added to PATH
- transfer.py

## usage

    import transfer

or

    from transfer import transfer

then 

    transfer(orig_sent, grm, tenses=None, progs=None, perfs=None, modalities=None, print_orig_mrs=False, print_orig_tree=False)

- orig_sent: sentence to be tranformed (string)
- grm: path to ERG 1214 grammar file (string)
- tenses: list of tenses to be generated (list)
    - possible tenses: 'past' (past), 'pres' (present), 'fut' (future)
    - if not specified: transform for all possible tenses
- progs: list of values of progressive aspect
    - possible values: '+' (progressive), '-' (non-progressive)
        - '+': Alice is smoking.
        - '-': Alice smokes.
    - if not specified: keep unchanged
- perfs: list of values of perfect aspect
    - possible values: '+' (perfect), '-' (non-perfect)
        - '+': Alice has graduated.
        - '-': Alice graduated.
    - if not specified: keep unchanged
- modalities: list of values of modality
    - ! The only currently possible modality now is '_may_v_modal' (may)
    - if not specified: generate all possible modality transforms
- print_orig_mrs: boolean
    - if True: print out MRS semantics of original sentence
- print_orig_tree: boolean
    - if True: print out HPSG derivation tree of original sentence in UDF format

The function returns a dictionary

    {transform_type: [transform_result]}

- transform_type: string indicating type of transformation
- transform_result: a pydelphin class
    - use transform_result['surface'] to access the transformed sentence as a string

Note:

- if none of tenses, progs and perfs is specified: transform for all possible combinations of tenses and aspects (12 in total)
