'''
Utility for parse, generate and transfer sentences with English Resource Grammar (ERG) and Answer Constraint Engine (ACE) parser. This module assumes 1214 version of ERG, with little to no compatibility with ERG2018. ACE parser provides parsing and generation functionality. This module provides a collection of transfer rules, by operating and inspecting the Minimal Recursion Semantics (MRS) and Head-Driven Phrase Structure Grammar (HPSG) output from ACE parsers.
'''

from functools import wraps
import errno
import os
import signal
from delphin import mrs, variable
from delphin.ace import ACEParser, ACEGenerator
from delphin.codecs import simplemrs
from re import findall, sub, search
TIMEOUT = 5


def swap_subj(orig_sem, index_ep):
    if index_ep.args.get('ARG2'):
        new_args = index_ep.args.copy()
        new_vars = copy_vars(orig_sem)
        new_rels = orig_sem.rels[:]
        # swap ARG1 and ARG2
        new_args['ARG1'] = index_ep.args['ARG2']
        new_args['ARG2'] = index_ep.args['ARG1']
        new_ep = mrs.EP(predicate=index_ep.predicate,
                        label=index_ep.label,
                        args=new_args)
        # update number for agreement
        # new_pers = new_vars[index_ep.args['ARG2']]['PERS']
        # new_num = new_vars[index_ep.args['ARG2']]['NUM']
        # new_vars[index_ep.args['ARG0']]['PERS'] = new_pers
        # new_vars[index_ep.args['ARG0']]['NUM'] = new_num
        # update index_ep
        new_rels.remove(index_ep)
        new_rels.append(new_ep)

    new_sem = copy_sem(orig_sem,
                       rels=new_rels,
                       variables=new_vars)
    return {'swap subj/obj': new_sem}


def negation(orig_sem, index_ep):
    '''
    return a negated MRS, assuming the original MRS is not negated

    :param  orig_sem: original MRS
    :type   orig_sem: MRS
    :param  index_ep: EP whose intrinsic variable matches sentential index
    :type   index_ep: EP
    :rtype  list(MRS)
    '''
    # TODO
    # Do we need this at all?
    new_sems = {}

    parg_ep = is_passive(orig_sem, index_ep)
    if parg_ep:
        subj = parg_ep.args['ARG2']
    else:
        subj = index_ep.args.get('ARG1')
    if subj:
        indef_qs = ['_a_q', '_some_q']
        for i, ep in enumerate(orig_sem.rels[:]):
            if ep.args.get('ARG0') == subj:
                if ep.predicate in indef_qs:
                    new_sems = negate_subj_Q(orig_sem, i, subj)

    if not new_sems:
        new_sems = negate_ep(orig_sem, index_ep)

    return new_sems


def negate_ep(orig_sem, ep):
    '''
    return an MRS whose index EP is negated by EP:neg

    :param  orig_sem: original MRS
    :type   orig_sem: MRS
    :param  ep:       EP whose intrinsic variable matches sentential index
    :type   ep:       EP
    :rtype  MRS
    '''
    new_rels = orig_sem.rels[:]
    new_hcons = orig_sem.hcons[:]
    new_vars = copy_vars(orig_sem)
    last_vid = get_max_vid(orig_sem)
    neg_label = 'h'+str(last_vid+1)
    neg_index = 'e'+str(last_vid+2)
    neg_arg = 'h'+str(last_vid+3)

    neg_ep = mrs.EP(
        predicate='neg',
        label=neg_label,
        args={'ARG0': neg_index, 'ARG1': neg_arg})
    new_rels.append(neg_ep)

    for hc in new_hcons:
        if hc.lo == ep.label:
            hi_mid = mrs.HCons(hc.hi, 'qeq', neg_label)
            new_hcons.append(hi_mid)
            new_hcons.remove(hc)
    mid_lo = mrs.HCons(neg_arg, 'qeq', ep.label)
    new_hcons.append(mid_lo)

    new_vars[neg_index] = {'SF': 'prop',
                           'TENSE': 'untensed',
                           'MOOD': 'indicative',
                           'PROG': '-',
                           'PERF': '-'}

    new_sem = copy_sem(orig_sem,
                       rels=new_rels, hcons=new_hcons, variables=new_vars)

    return {'negation': new_sem}


def negate_subj_Q(orig_sem, i, subj):
    '''
    return an MRS whose index EP's ARG1's quantifier is changed to EP:no_q

    :param  orig_sem: original MRS
    :type   orig_sem: MRS
    :param  index_ep: EP whose intrinsic variable matches sentential index
    :type   index_ep: EP
    :rtype  MRS
    '''
    new_rels = orig_sem.rels[:]
    new_vars = copy_vars(orig_sem)

    new_rels[i] = copy_ep(new_rels[i], predicate='_no_q')

    new_vars[subj]['NUM'] = 'sg'

    new_sem = copy_sem(orig_sem, rels=new_rels, variables=new_vars)

    return {'negation': new_sem}


def tense_aspect(orig_sem, var, tenses=[], progs=[], perfs=[]):
    '''
    return a list of MRS with changed tense and aspects

    :param  orig_sem: original MRS
    :type   orig_sem: MRS
    TODO change ep to var
    :param  ep:       EP whose intrinsic variable matches sentential index
    :type   ep:       EP
    :param  tenses:   tenses to be used for transformation
                      tense in {past, pres, fut}
    :type   tenses:   list(str)
    :param  progs:    progressive aspect values to be used for transformation
                      prog in {+, -}
    :type   progs:    list(str)
    :param  perfs:    perfect aspect values to be used for transformation
                      prog in {+, -}
    :type   perfs:    list(str)
    :rtype  list(MRS)
    '''
    def one_tense_aspect(orig_sem, var, tense, prog, perf):
        ''' change orig_sem to one combination of tense, prog, perf '''
        new_vars = copy_vars(orig_sem)

        new_vars[var]['TENSE'] = tense
        new_vars[var]['PROG'] = prog
        new_vars[var]['PERF'] = perf

        new_sem = copy_sem(orig_sem, variables=new_vars)
        return new_sem

    def tense_aspect_to_str(x):
        ''' return readable string for given tense and aspect '''
        tense, prog, perf = x
        tense_dict = {'past': 'past', 'pres': 'present', 'fut': 'future'}
        aspect_dict = {('-', '-'): 'simple',
                       ('-', '+'): 'perfect',
                       ('+', '-'): 'progressive',
                       ('+', '+'): 'perfect progressive'}

        return tense_dict[tense] + ' ' + aspect_dict[(prog, perf)]

    if not tenses and not progs and not perfs:
        return []

    if not tenses:
        tenses = [orig_sem.variables[var]['TENSE']]
    if not progs:
        progs = [orig_sem.variables[var]['PROG']]
    if not perfs:
        perfs = [orig_sem.variables[var]['PERF']]

    tenses_aspects = [(tense, prog, perf)
                      for tense in tenses for prog in progs for perf in perfs]
    new_sems = {tense_aspect_to_str(x): one_tense_aspect(orig_sem, var, x[0], x[1], x[2])
                for x in tenses_aspects}

    return new_sems


def itcleft(orig_sem, ep):
    '''
    return a list of MRS whose index EP's arguments are extracted by it clefting

    :param  orig_sem: original MRS
    :type   orig_sem: MRS
    :param  index_ep: EP whose intrinsic variable matches sentential index
    :type   index_ep: EP
    :rtype  list(MRS)
    '''
    def itcleft_arg(orig_sem, ep, arg_id, passive_ep=False):
        '''
        return an MRS with arg extracted by it-cleft
        handle is the handle of original index EP
        '''
        new_rels = orig_sem.rels[:]
        new_hcons = orig_sem.hcons[:]
        new_vars = copy_vars(orig_sem)

        last_vid = get_max_vid(orig_sem)
        cleft_label = 'h'+str(last_vid+1)
        cleft_index = 'e'+str(last_vid+2)

        cleft_ep = mrs.EP(
            predicate='_be_v_itcleft',
            label=cleft_label,
            args={'ARG0': cleft_index,
                  'ARG1': ep.args[arg_id],
                  'ARG2': ep.label}
        )
        new_rels.append(cleft_ep)

        new_vars[cleft_label] = {}
        new_vars[cleft_index] = {'SF': 'prop',
                                 'TENSE': 'pres',
                                 'MOOD': 'indicative',
                                 'PROG': '-',
                                 'PERF': '-'}

        for i, hc in enumerate(new_hcons):
            if hc.lo == ep.label:
                new_hcons[i] = mrs.HCons(hc.hi, 'qeq', cleft_label)

        new_sem = copy_sem(orig_sem,
                           index=cleft_index, rels=new_rels, hcons=new_hcons, variables=new_vars)
        if passive_ep:
            new_sem = passive(new_sem, ep, arg_id=arg_id)['passive: '+arg_id]
        return new_sem

    return {'it cleft: ' + i: itcleft_arg(orig_sem, ep, i, passive_ep=i != 'ARG1')
            for i in get_extractable(ep)}


def passive(orig_sem, ep, arg_id=None):
    '''
    return an MRS whose index EP's ARG2 is extracted by passivisation

    :param  orig_sem: original MRS
    :type   orig_sem: MRS
    :param  index_ep: EP whose intrinsic variable matches sentential index
    :type   index_ep: EP
    :rtype  list(MRS)
    '''
    # TODO
    def passive_arg(orig_sem, ep, arg_id):
        ''' return a MRS with ep passivised with arg_id as syntactic specifier '''
        new_rels = orig_sem.rels[:]
        new_vars = copy_vars(orig_sem)
        parg = ep.args[arg_id]

        last_vid = get_max_vid(orig_sem)
        parg_label = ep.label
        parg_index = 'e'+str(last_vid+1)
        parg_ep = mrs.EP(
            predicate='parg_d',
            label=parg_label,
            args={'ARG0': parg_index,
                  'ARG1': ep.args['ARG0'],
                  'ARG2': parg})
        new_rels.append(parg_ep)

        new_vars[parg_label] = {}
        new_vars[parg_index] = {'SF': 'prop',
                                'TENSE': 'untensed',
                                'MOOD': 'indicative',
                                'PROG': '-',
                                'PERF': '-'}

        new_sem = copy_sem(orig_sem,
                           rels=new_rels,
                           variables=new_vars)
        return new_sem

    if arg_id:
        arg_ids = [arg_id]
    else:
        arg_ids = get_extractable(ep)

    return {'passive: ' + str(i): passive_arg(orig_sem, ep, i)
            for i in arg_ids if i != 'ARG1'}


def inv_polar_question(orig_sem, index):
    '''
    return an MRS whose index EP's intrinsic variable's mode is 'ques'

    :param  orig_sem: original MRS
    :type   orig_sem: MRS
    :param  index_ep: EP whose intrinsic variable matches sentential index
    :type   index_ep: EP
    :rtype  list(MRS)
    '''
    # TODO
    new_vars = copy_vars(orig_sem)

    new_vars[index]['SF'] = 'ques'

    new_sem = copy_sem(orig_sem, variables=new_vars)
    return {'inverted polar question': new_sem}


def modality(orig_sem, ep, modalities):
    '''
    return a dict {type:[MRS]} with changed tense and aspects

    :param orig_sem:   original MRS
    :type  orig_sem:   MRS
    # TODO change index_ep to h
    :param index_ep: EP whose intrinsic variable matches sentential index
    :type  index_ep: EP
    :param modalities: modalities to be used for transformation
                       TODO determine the set of possible modalities
    :type  modalities: list(str)
    :rtype list(MRS)
    '''
    def one_modality(orig_sem, ep, modality):
        '''
        return one new MRS with added modality
        '''
        last_vid = get_max_vid(orig_sem)
        modal_label = 'h'+str(last_vid+1)
        modal_index = 'e'+str(last_vid+2)
        modal_arg = 'h'+str(last_vid+3)
        new_sem = replace_handle(orig_sem, ep.label, modal_label)

        new_index = new_sem.index
        new_rels = new_sem.rels[:]
        new_hcons = new_sem.hcons[:]
        new_vars = copy_vars(new_sem)

        if new_sem.index == ep.args['ARG0']:
            new_index = modal_index

        modal_ep = mrs.EP(predicate=modality, label=modal_label,
                          args={'ARG0': modal_index, 'ARG1': modal_arg})
        new_rels.append(modal_ep)

        modal_arg_hc = mrs.HCons(modal_arg, 'qeq', ep.label)
        new_hcons.append(modal_arg_hc)

        new_vars[modal_label] = {}
        new_vars[modal_arg] = {}
        new_vars[modal_index] = {'SF': 'prop',
                                 'TENSE': 'pres',
                                 'MOOD': 'indicative',
                                 'PROG': '-',
                                 'PERF': '-'}
        new_vars[ep.args['ARG0']]['TENSE'] = 'untensed'

        new_sem = copy_sem(new_sem,
                           index=new_index, rels=new_rels, hcons=new_hcons, variables=new_vars)
        return new_sem

    def modal_to_str(predicate):
        ''' convert modality predicate to more readable strings '''
        predicate = predicate[1:]
        idx = predicate.index('_')
        if idx:
            predicate = predicate[:idx]
            return f'modality: {predicate}'
        else:
            return 'invalid'

    new_sems = {modal_to_str(m): one_modality(orig_sem, ep, m)
                for m in modalities}
    return new_sems


def find_eps(sem, handle=None, index=None, exclude=None):
    '''
    return a list of EPs that match the handle/index criteria

    :param sem:    semantics
    :type  sem:    MRS
    :param handle: handle for EPs' labels to match
    :type  handle: str
    :param index:  semantic index for EP's intrinsic variable to match
    :type  index:  str
    :rtype list(EP)
    '''
    eps = sem.rels
    if handle:
        qeqs = get_qeqs(sem)
        handles = [handle] + qeqs.get(handle, [])
        eps = [ep for ep in eps if ep.label in handles]

    if index:
        eps = [ep for ep in eps if ep.args.get('ARG0') == index]

    if exclude:
        eps = [ep for ep in eps if ep not in exclude]

    return eps


def get_index_ep(sem):
    ''' return an EP whose intrisic variables matches the sentential index '''
    for ep in sem.rels:
        if ep.args.get('ARG0') == sem.index:
            return ep


def skip_itcleft(sem, itcleft):
    ''' return the ARG2 EP of itcleft '''
    # TODO is it safe to return None?
    for ep in find_eps(sem, handle=itcleft.args['ARG2']):
        if ep.predicate != 'parg':
            return ep

    return


def get_qeqs(sem):
    ''' returns a dictionary of qeq relations '''
    def find_qeqs_recur(qeqs, hi):
        los = qeqs.get(hi, [])
        if not los:
            return los

        lowers = []
        for lo in los:
            new_lowers = find_qeqs_recur(qeqs, lo)
            lowers.extend(new_lowers)
        return los + lowers

    qeqs = {hc.hi: [] for hc in sem.hcons}
    for hc in sem.hcons:
        qeqs[hc.hi].append(hc.lo)
    for hi in qeqs:
        for lo in qeqs[hi]:
            lowers = find_qeqs_recur(qeqs, lo)
            qeqs[hi].extend(lowers)

    return qeqs


def get_max_vid(sem):
    ''' return the maximum vid in sem.variables '''
    return max([variable.id(v) for v in sem.variables], default=0)


def copy_sem(orig_sem, top=None, index=None, rels=None, hcons=None, icons=None, variables=None):
    ''' return an MRS with specified changes to original MRS '''
    if not top:
        top = orig_sem.top
    if not index:
        index = orig_sem.index
    if not rels:
        rels = orig_sem.rels[:]
    if not hcons:
        hcons = orig_sem.hcons[:]
    if not icons:
        icons = orig_sem.icons[:]
    if not variables:
        variables = copy_vars(orig_sem)

    return mrs.MRS(top=top, index=index, rels=rels, hcons=hcons, icons=icons, variables=variables)


def copy_vars(sem):
    ''' return a copy of orig_var: {var: {property: value}}'''
    return {v: sem.variables[v].copy() for v in sem.variables}


def copy_ep(orig_ep, predicate=None, label=None, args=None):
    ''' return a copy of orig_ep with specified changes '''
    if not predicate:
        predicate = orig_ep.predicate
    if not label:
        label = orig_ep.label
    if not args:
        args = orig_ep.args.copy()

    return mrs.EP(predicate=predicate, label=label, args=args)


def replace_handle(orig_sem, orig_h, new_h):
    ''' replace all occurences of orig_h with new_h in orig_sem '''
    new_rels = orig_sem.rels[:]
    new_hcons = orig_sem.hcons.copy()

    for i, ep in enumerate(new_rels):
        for arg_id in ep.args:
            if ep.args[arg_id] == orig_h:
                new_args = ep.args.copy()
                new_args[arg_id] = new_h
                new_rels[i] = copy_ep(ep, args=new_args)
                break

    for i, hc in enumerate(new_hcons):
        if hc.hi == orig_h:
            new_hcons[i] = mrs.HCons(new_h, 'qeq', hc.lo)
            continue
        if hc.lo == orig_h:
            new_hcons[i] = mrs.HCons(hc.hi, 'qeq', new_h)

    return copy_sem(orig_sem, rels=new_rels, hcons=new_hcons)


def get_extractable(ep):
    ''' return a list of arg_id for entity arguments of ep '''
    # ignore CARG (constant arg) when extracting variables
    arg_ids = [x for x in ep.args if x.startswith('ARG')]
    return [arg_id for arg_id in arg_ids
            if variable.type(ep.args[arg_id]) == 'x']


def transfer(orig_sent, grm, ace, timeout=TIMEOUT, rules=None, tenses=None, progs=None, perfs=None, modalities=None,
             print_orig_mrs=False, print_orig_tree=False, parsed_sem=None):
    '''
    :param print_orig_tree:
                 if True, print derivation tree of best parse in udf format
    :type  print_orig_tree:
                 boolean
    :param print_orig_mrs:
                 if True, print mrs of best parse with indentation
    :type  print_orig_mrs:
                 boolean
    '''
    def get_trans_list(orig_sem, qeqs, ep, rules=None, tenses=None, progs=None, perfs=None, modalities=None):
        '''
        return list of possible transformations
        negation, tense, question, modal, it cleft, passive
        if is_conjunction(): skip
        if is_unknown(): skip
        if is_question(): skip
        if is_negated(): no negation, it cleft
        if is_modal(): no modal, tense
        if is_it_cleft(): no negation, it cleft, passive
        if is_passive(): no it cleft, passive
        '''
        if is_conjunction(ep):
            return {}
        if is_unknown(ep):
            return {}
        if is_question(orig_sem):
            return {}

        if not tenses:
            tenses = ['past', 'pres', 'fut']
        if not modalities:
            modalities = ['may', ]  # 'might', 'must',
            # 'should', 'would', 'can', 'could', 'gotta']
        modalities = [f'_{x}_v_modal' for x in modalities]
        rules_pool = {0: lambda x: negation(x, ep),
                      1: lambda x: tense_aspect(x, ep.args['ARG0'], tenses=tenses, progs=progs, perfs=perfs),
                      2: lambda x: modality(x, ep, modalities),
                      #   3: lambda x: inv_polar_question(x, x.index),
                      4: lambda x: itcleft(x, ep),
                      5: lambda x: passive(x, ep),
                      6: lambda x: {'native': x},
                      7: lambda x: swap_subj(x, ep)}
        if not rules:
            rules = rules_pool
        else:
            rules = {i: rules_pool[i] for i in rules}

        if is_itcleft(ep):
            rules = pop_all(rules, [0, 4])
            ep = find_eps(orig_sem, handle=ep.args['ARG2'])[0]
            if rules.get(1):
                rules[1] = lambda x: tense_aspect(
                    x, ep.args['ARG0'], tenses=tenses, progs=progs, perfs=perfs)
            return get_trans_list(orig_sem, qeqs, ep, rules=rules)
        if is_modal(ep):
            rules = pop_all(rules, [1, 2, 5])
            ep = find_eps(orig_sem, handle=ep.args['ARG1'])[0]
            return get_trans_list(orig_sem, qeqs, ep, rules=rules)
        if is_because_x(ep):
            ep = find_eps(orig_sem, handle=ep.args['ARG1'])[0]
            return get_trans_list(orig_sem, qeqs, ep, rules=rules)
        if is_passive(orig_sem, ep):
            rules = pop_all(rules, [4, 5])
        if is_negated(orig_sem, qeqs, ep):
            rules = pop_all(rules, [0])

        return rules

    simp_trans_sems = {}
    comp_trans_sems = {}
    # trans_sems = {}
    transforms = {}
    # set timeout here to avoid waiting too long to parse
    if not parsed_sem:
        try:
            orig = get_best_parse(orig_sent, grm, ace, timeout)
            if not orig:
                return
            orig_sem = orig.mrs()
        except:
            return {'timeout': 1}
    else:
        orig_sem = parsed_sem

    # mrs_enc = simplemrs.encode(orig.mrs(), indent=True)
    # if print_orig_mrs:
    #     print(mrs_enc)
    # if print_orig_tree:
    #     deriv_enc = orig.derivation().to_udf()
    #     print(deriv_enc)
    qeqs = get_qeqs(orig_sem)
    index_ep = get_index_ep(orig_sem)
    if index_ep is None:
        return {}

    rules = get_trans_list(orig_sem, qeqs, index_ep, rules=rules,
                           tenses=tenses, progs=progs, perfs=perfs, modalities=modalities)
    try:
        for r in rules.values():
            simp_trans_sems.update(r(orig_sem))
    except:
        return {}
    for trans_type, s in simp_trans_sems.items():
        if trans_type == 'native':
            continue
        new_qeqs = get_qeqs(s)
        new_index_ep = get_index_ep(s)
        if index_ep is None:
            continue
        comp_rules = get_trans_list(s, new_qeqs, new_index_ep,
                                    rules=[4, 5], tenses=tenses, progs=progs, perfs=perfs, modalities=modalities)
        for r in comp_rules.values():
            comp_sems = r(s)
            for comp_type, comp_sem in comp_sems.items():
                comp_trans_sems[f'{trans_type}+{comp_type}'] = comp_sem

    simp_trans_sems.update(comp_trans_sems)

    for trans_type in simp_trans_sems:
        if trans_type == 'inverted polar question':
            type_transforms = generate(
                simp_trans_sems[trans_type], grm, ace, timeout, filter_func=is_inverted)
        else:
            type_transforms = generate(
                simp_trans_sems[trans_type], grm, ace, timeout)
        if type_transforms:
            type_transforms = [{'surface': x['surface']}
                               for x in type_transforms]
            transforms.update({trans_type: type_transforms})
    transforms['original'] = [{
        'surface': orig_sent,
        'mrs': simplemrs.encode(orig_sem)
    }]

    return transforms


def get_best_parse(sent, grm, ace, timeout):
    '''
    return highest ranked parse result of sent by grm

    :param sent: sentenec to be parsed
    :type  sent: str
    :param grm:  path to grammar file
    :type  grm:  str
    :rtype Result
    '''
    import os
    with ACEParser(grm, executable=ace, cmdargs=['--timeout', str(timeout)]) as parser:
        results = parser.interact(sent).results()
        if not results:
            return
        best = results[0]
        return best


def generate(sem, grm, ace, timeout, filter_func=lambda x: True):
    '''
    return filtered generation results from sem by grm

    :param sem: semantics
    :type  sem: MRS
    :param grm: path to grammar file
    :type  grm: str
    :param filter_func:
                boolean function that returns True for surface strings to be kept
    :type  filter_func:
                function
    :rtype list(str)
    '''
    import os
    sem_enc = simplemrs.encode(sem, lnk=False, indent=True)
    with ACEGenerator(grm, executable=ace, cmdargs=['--timeout', str(timeout)]) as generator:
        try:
            results = generator.interact(sem_enc).results()
        except:
            return
        if not results:
            # print(sem_enc)
            return
        results = [r for r in results if filter_func(r)]
        results = remove_dupl(results)

    return results


def get_tense(sem, index_ep=None):
    if not index_ep:
        index_ep = get_index_ep(sem)

    if is_itcleft(index_ep):
        index_ep = find_eps(sem, handle=index_ep.args.get('ARG2'))[0]
        return get_aspect(sem, index_ep=index_ep)
    if is_modal(index_ep):
        return
    if is_because_x(index_ep):
        index_ep = find_eps(sem, handle=index_ep.args['ARG1'])[0]
        return get_aspect(sem, index_ep=index_ep)

    index = index_ep.args.get('ARG0')
    if not index:
        return
    tense = sem.variables[index].get('TENSE')
    if not tense:
        return
    tense_dict = {'past': 'past',
                  'fut': 'future',
                  'pres': 'present',
                  'untensed': 'untensed',
                  'tensed': 'tensed'}

    return tense_dict[tense]


def get_aspect(sem, index_ep=None):
    if not index_ep:
        index_ep = get_index_ep(sem)

    if is_itcleft(index_ep):
        index_ep = find_eps(sem, handle=index_ep.args.get('ARG2'))[0]
        return get_aspect(sem, index_ep=index_ep)
    if is_modal(index_ep):
        return
    if is_because_x(index_ep):
        index_ep = find_eps(sem, handle=index_ep.args['ARG1'])[0]
        return get_aspect(sem, index_ep=index_ep)

    index = index_ep.args.get('ARG0')
    if not index:
        return
    prog = sem.variables[index].get('PROG')
    perf = sem.variables[index].get('PERF')
    if not (prog and perf):
        return
    aspect_dict = {('-', '-'): 'simple',
                   ('-', '+'): 'perfect',
                   ('+', '-'): 'progressive',
                   ('+', '+'): 'perfect progressive'}

    return aspect_dict[(prog, perf)]


def is_because_x(ep):
    ''' return True if ep is _because_x '''
    return ep.predicate == '_because_x'


def is_conjunction(ep):
    ''' return True if ep is a conjunctive '''
    return ep.predicate.endswith('_c')


def is_itcleft(ep):
    ''' return True if ep is _be_v_itcleft() '''
    return ep.predicate == '_be_v_itcleft'


def is_modal(ep):
    ''' return True if ep is .*_modal '''
    return ep.predicate.endswith('_modal')


def is_negated(sem, qeqs, index_ep):
    ''' return True if either is_negated or has_neg_q '''
    if is_negated_ep(sem, qeqs, index_ep):
        return True
    if has_neg_q(sem, index_ep)[0]:
        return True

    return False


def is_negated_ep(sem, qeqs, index_ep):
    ''' return True if ep is negated in sem '''
    for ep in sem.rels:
        if ep.predicate == 'neg':
            try:
                if index_ep.label in qeqs.get(ep.args.get('ARG1')):
                    return True
            except:
                return False

    return False


def has_neg_q(sem, index_ep):
    '''
    return
        True, list index of _no_q if ARG1 of index_ep is bound by _no_q
        False, 0 if ARG1 of index_ep is an x variable and is not bound by _no_q
        False, 1 if index_ep does not have an ARG of type x
    '''
    subj = index_ep.args.get('ARG1')
    if not subj:
        return False, 1
    if variable.type(subj) != 'x':
        return False, 1
    for i, ep in enumerate(sem.rels):
        if ep.args.get('ARG0') == subj:
            if ep.predicate == '_no_q':
                return True, i

    return False, 0


def is_passive(sem, ep):
    '''return True if there is an EP:parg sharing ep's label '''
    for other_ep in find_eps(sem, handle=ep.label, exclude=[ep]):
        if other_ep.predicate == 'parg_d':
            return other_ep

    return False


def is_question(sem):
    ''' return True if sem is a question '''
    return sem.variables[sem.index].get('SF') == 'ques'


def is_unknown(ep):
    ''' return True if ep is 'unknown' '''
    return ep.predicate == 'unknown'


def remove_dupl(results):
    '''
    return a list of results without duplicate surfaces.
    sentences different only in number of commas treated as duplicates.
    '''
    def clean(result):
        sent = sub(r',', r'', result['surface'])
        cnt = len(findall(r',', result['surface']))
        return sent, cnt

    if len(results) <= 1:
        return results

    cleaned = [clean(r) for r in results]
    dupl_results = [(i, c[1]) for i, c in enumerate(cleaned)
                    if c[0] == cleaned[0][0]]
    left_results = [results[i] for i, _ in enumerate(results)
                    if i not in [d[0] for d in dupl_results]]
    kept_i = min(dupl_results, key=lambda x: x[1])[0]

    return [results[kept_i]] + remove_dupl(left_results)


def is_inverted(result):
    ''' return True if root node of result's HPSG derivation is liscenced by hd_yesno_c '''
    return result.derivation().entity == 'hd_yesno_c'


def pop_all(d, xs):
    ''' d.pop all keys in xs '''
    for x in xs:
        d.pop(x, None)
    return d


def negate_from_mrs(orig_sem, grm, ace, timeout):
    index_ep = get_index_ep(orig_sem)
    negated_sem = list(negation(orig_sem, index_ep).values())[0]
    transforms = generate(negated_sem, grm, ace, timeout)
    for result in transforms:
        del result['derivation']
        del result['mrs']
    transforms = {'negation': transforms}
    return transforms
