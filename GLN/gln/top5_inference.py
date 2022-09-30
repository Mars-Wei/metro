import os
import numpy as np
import torch
import pandas as pd
import math
import json
from copy import deepcopy
from rdkit import Chem
from tqdm import trange
from gln.common.cmd_args import cmd_args
from gln.test.model_inference import RetroGLN


def get_inference_answer(smiles, beam_size):
    pred_struct = model.run(smiles, 5*beam_size, 5*beam_size, rxn_type='UNK')
    if pred_struct is None:
        return []
    reactants_list = pred_struct['reactants']
    scores_list = pred_struct['scores']
    answer = []
    aim_size = beam_size
    for i in range(len(reactants_list)):
        if aim_size == 0:
            break
        reactants = reactants_list[i].split('.')
        score = scores_list[i]
        num_valid_reactant = 0
        sms = set()
        for r in reactants:
            m = Chem.MolFromSmiles(r)
            if m is not None:
                num_valid_reactant += 1
                sms.add(Chem.MolToSmiles(m))
        if num_valid_reactant != len(reactants):
            continue
        if len(sms):
            answer.append([sorted(list(sms)), -math.log10(score+1e-10)]) 
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
    beam_size = 5
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    predict_epoch = 86
    model_dump = os.path.join(cmd_args.save_dir, 'model-%d.dump' % predict_epoch)
    model = RetroGLN(model_dump)

    stock = pd.read_hdf('zinc_stock_17_04_20.hdf5', key="table")  
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])

    test_set = load_dataset("test")
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
                for expansion_solution in get_inference_answer(first_route[-1], beam_size):
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
