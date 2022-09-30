import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import numpy as np
import pandas as pd
import os
import json
import math
import argparse
from tqdm import trange
from copy import deepcopy
from generate_retro_templates import process_an_example
from rdkit import DataStructs
from preprocess import get_train_dataset
from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(4)


def do_one(prod_smiles, beam_size):
    global jx_cache

    
    ex = Chem.MolFromSmiles(prod_smiles)
    rct = rdchiralReactants(prod_smiles)
    fp = getfp(prod_smiles)
    
    sims = similarity_metric(fp, [fp_ for fp_ in train_product_fps])
    js = np.argsort(sims)[::-1]

    # Get probability of precursors
    probs = {}
    
    for j in js[:100]:
        jx = j
        
        if jx in jx_cache:
            (template, rcts_ref_fp) = jx_cache[jx]
        else:
            try:
                template = '(' + process_an_example(train_rxn_smiles[jx], super_general=True).replace('>>', ')>>')
            except:
                return []
            rcts_ref_fp = getfp(train_rxn_smiles[jx].split('>')[0])
            jx_cache[jx] = (template, rcts_ref_fp)

        try:    
            rxn = rdchiralReaction(template)
        except:
            return []

        try:
            outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
        except Exception as e:
            print(e)
            outcomes = []
            
        for precursors in outcomes:
            precursors_fp = getfp(precursors)
            precursors_sim = similarity_metric(precursors_fp, [rcts_ref_fp])[0]
            if precursors in probs:
                probs[precursors] = max(probs[precursors], precursors_sim * sims[j])
            else:
                probs[precursors] = precursors_sim * sims[j]
        
    testlimit = 50
    mols = []
    legends = []
    score = []
    found_rank = 9999
    for r, (prec, prob) in enumerate(sorted(probs.items(), key=lambda x:x[1], reverse=True)[:testlimit]):
        mols.append(Chem.MolFromSmiles(prec))
        legends.append('overall score: {:.3f}'.format(prob))
        score.append(-math.log10(prob+1e-10))
    
    answer = []
    aim_size = beam_size
    for i in range(len(mols)):
        if aim_size == 0:
            break
        answer.append([sorted(Chem.MolToSmiles(mols[i], True).split(".")), score[i]])
        aim_size -= 1
    return answer


def load_dataset(split):
    file_name = "%s_dataset.json" % split
    file_name = os.path.expanduser(file_name)
    dataset = [] # (product_smiles, materials_smiles, depth)
    with open(file_name, 'r') as f:
        _dataset = json.load(f)
        for product, reaction_trees in _dataset.items():
            materials_list = []
            for i in range(1, int(reaction_trees['num_reaction_trees'])+1):
                materials_list.append(reaction_trees[str(i)]['materials'])
            dataset.append({
                "product": product, 
                "targets": materials_list, 
                "depth": reaction_trees['depth']
            })

    return dataset


def check_reactant_is_material(reactant):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] in stock_inchikeys


def check_reactants_are_material(reactants):
    for reactant in reactants:
        if Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] not in stock_inchikeys:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--beam_size", help="beam size", type=int, default=5)
    parser.add_argument('--inference_dataset', type=str, default='test', help='valid or test')

    args = parser.parse_args()
    beam_size = args.beam_size
    
    np.random.seed(42)

    similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity
    similarity_label = 'Tanimoto'
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=True)
    getfp_label = 'Morgan2Feat'

    train_products_list, train_reactants_list = get_train_dataset(True)
    train_product_fps = [getfp(smi) for smi in train_products_list]
    train_rxn_smiles = [train_reactants_list[i]+'>>'+train_products_list[i] for i in range(len(train_reactants_list))]
    jx_cache = {}

    stock = pd.read_hdf('zinc_stock_17_04_20.hdf5', key="table")  
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])

    test_set = load_dataset(args.inference_dataset)
    overall_result = np.zeros((beam_size, 2))
    depth_hit = np.zeros((2, 15, beam_size))
    for epoch in trange(0, len(test_set)):
        # Initialization
        answer_set = []
        queue = []
        queue.append({
            "score": 0.0,
            "routes_info": [{"route": [test_set[epoch]["product"]], "depth": 0}],  # List of routes information
            "starting_materials": [],
        })
        max_depth = test_set[epoch]["depth"]
        while True:
            if len(queue) == 0:
                break
            nxt_queue = []
            for item in queue:
                score = item["score"]
                routes_info = item["routes_info"]
                starting_materials = item["starting_materials"]
                first_route_info = routes_info[0]
                first_route, depth = first_route_info["route"], first_route_info["depth"]
                if depth > max_depth:
                    continue
                for expansion_solution in do_one(first_route[-1], beam_size):
                    iter_routes = deepcopy(routes_info)
                    iter_routes.pop(0)
                    iter_starting_materials = deepcopy(starting_materials)
                    expansion_reactants, expansion_score = expansion_solution[0], expansion_solution[1]
                    expansion_reactants = sorted(expansion_reactants)
                    if check_reactants_are_material(expansion_reactants) and len(iter_routes) == 0:
                        answer_set.append({
                            "score": score+expansion_score,
                            "starting_materials": iter_starting_materials+expansion_reactants,
                            })
                    else:
                        for reactant in expansion_reactants:
                            if check_reactant_is_material(reactant):
                                iter_starting_materials.append(reactant)
                            else:
                                iter_routes = [{"route": first_route+[reactant], "depth": depth+1}] + iter_routes
                        nxt_queue.append({
                            "score": score+expansion_score,
                            "routes_info": iter_routes,
                            "starting_materials": iter_starting_materials
                        })
            queue = sorted(nxt_queue, key=lambda x: x["score"])[:beam_size]
                
        answer_set = sorted(answer_set, key=lambda x: x["score"])
        record_answers = set()
        final_answer_set = []
        for item in answer_set:
            score = item["score"]
            starting_materials = item["starting_materials"]
            answer_keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(m))[:14] for m in starting_materials]
            if '.'.join(sorted(answer_keys)) not in record_answers:
                record_answers.add('.'.join(sorted(answer_keys)))
                final_answer_set.append({
                    "score": score,
                    "answer_keys": answer_keys
                })
        final_answer_set = sorted(final_answer_set, key=lambda x: x["score"])[:beam_size]

        # Calculate answers
        ground_truth_keys_list = [
            set([
                Chem.MolToInchiKey(Chem.MolFromSmiles(target))[:14] for target in targets
            ]) for targets in test_set[epoch]["targets"]
        ]
        overall_result[:, 1] += 1
        depth_hit[1, test_set[epoch]["depth"], :] += 1
        for rank, answer in enumerate(final_answer_set):
            answer_keys = set(answer["answer_keys"])
            flag = False
            for ground_truth_keys in ground_truth_keys_list:
                if ground_truth_keys == answer_keys:
                    overall_result[rank:, 0] += 1
                    depth_hit[0, test_set[epoch]["depth"], rank:] += 1
                    flag = True
                    break
            if flag: break
        if (epoch + 1) % 100 == 0: 
            print("overall_result: ", overall_result)
            print("depth_hit: ", depth_hit)
    print("overall_result: ", overall_result, overall_result[:, 0] / overall_result[:, 1])
    print("depth_hit: ", depth_hit, depth_hit[0, :, :] / depth_hit[1, :, :])
