import os 
import sys
import numpy as np
import pandas as pd
import time
from copy import deepcopy
import json
import argparse
import heapq
import logging

from collections import defaultdict, deque
from tqdm import tqdm
from rdkit import Chem

import torch
from torchdrug import core, tasks, models, utils, data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from g2g import dataset

logger = logging.getLogger(__file__)

def check_reactants_is_material(reactant, stock_inchikeys):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] in stock_inchikeys


def load_dataset(split):
    file_name = "~/scratch/molecule-datasets/%s_dataset.json" % split
    file_name = os.path.expanduser(file_name)
    dataset = [] # (product_smiles, materials_smiles, depth)
    with open(file_name, 'r') as f:
        _dataset = json.load(f)
        for product, reaction_trees in _dataset.items():
            _product = reaction_trees["1"]["retro_routes"][0][0].split(">")[0]
            materials_list = []
            for i in range(1, int(reaction_trees['num_reaction_trees'])+1):
                materials_list.append(reaction_trees[str(i)]['materials'])
            dataset.append({
                "product": _product, 
                "targets": materials_list, 
                "depth": reaction_trees['depth']
            })

    return dataset


def load_model(beam_size):
    # Center Identification
    reaction_dataset = dataset.USPTOFull("~/scratch/molecule-datasets", 
                atom_feature="center_identification", kekulize=True)
    reaction_train, reaction_valid, reaction_test = reaction_dataset.split()

    reaction_model = models.RGCN(input_dim=reaction_dataset.node_feature_dim,
                    hidden_dims=[512, 512, 512, 512],
                    # hidden_dims=[10, 10],
                    num_relation=reaction_dataset.num_bond_type,
                    concat_hidden=True)
    reaction_task = tasks.CenterIdentification(reaction_model,
                                            feature=("graph", "atom", "bond"))

    # Synthon Completion
    synthon_dataset = dataset.USPTOFull("~/scratch/molecule-datasets/", as_synthon=True,
                atom_feature="synthon_completion", kekulize=True)
    synthon_train, synthon_valid, synthon_test = synthon_dataset.split()

    synthon_model = models.RGCN(input_dim=synthon_dataset.node_feature_dim,
                            hidden_dims=[512, 512, 512, 512],
                            # hidden_dims=[10, 10],
                            num_relation=synthon_dataset.num_bond_type,
                            concat_hidden=True)
    synthon_task = tasks.SynthonCompletion(synthon_model, feature=("graph",))

    # Retrosynthesis
    reaction_task.preprocess(reaction_train, None, None)
    synthon_task.preprocess(synthon_train, None, None)
    task = tasks.Retrosynthesis(reaction_task, synthon_task, center_topk=beam_size,
                                num_synthon_beam=beam_size, max_prediction=beam_size)
    optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
    solver = core.Engine(task, reaction_train, reaction_valid, reaction_test,
                        optimizer, gpus=args.gpus, batch_size=32)
    solver.load("g2gs_reaction_model.pth", load_optimizer=False)
    solver.load("g2gs_synthon_model.pth", load_optimizer=False)

    return task, reaction_dataset


def get_batch(product_smiles, kwargs):
    batch = []
    for i, smiles in enumerate(product_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue
        product = data.Molecule.from_molecule(mol, **kwargs)
        with product.node():
            product.node_label = torch.zeros(product.num_node)
            product.atom_map = torch.arange(product.num_node) + 1
        with product.edge():
            product.edge_label = torch.zeros(product.num_edge)
            product.bond_stereo[:] = 0
        batch.append({
            "graph": (product, product),    # Fake reactant
            "reaction": 0,
            "sample id": i,
        })

    batch = data.graph_collate(batch)
    if args.gpus:
        batch = utils.cuda(batch)
    return batch


def get_prediction(model, batch):
    reactants, num_prediction = model.predict(batch)
    num_prediction = num_prediction.cumsum(0)
    answer = [[]]
    for i, graph in enumerate(reactants):
        if i == num_prediction[len(answer)-1]: 
            answer.append([])
        _reactants = graph.connected_components()[0]
        answer[-1].append([
            [reactant.to_smiles(isomeric=False, atom_map=False, canonical=True) for reactant in _reactants],
            graph.logps.item()
        ])
    assert len(answer) == num_prediction.shape[0]
    return answer


def select_topk(answers, k):
    answer_set = []

    def dfs(i, t, cur_answer):
        if i == len(answers):
            if len(answer_set) >= k:
                heapq.heappushpop(answer_set, (t, deepcopy(cur_answer)))
            else:
                heapq.heappush(answer_set, (t, deepcopy(cur_answer)))
        else:
            for answer in answers[i]:
                cur_answer += answer[0]
                dfs(i+1, t+answer[1], cur_answer)
                for j in range(len(answer[0])):
                    cur_answer.pop()

    dfs(0, 0, [])
    return answer_set


parser = argparse.ArgumentParser()
parser.add_argument("-k", "--beam_size", help="beam size", type=int, default=5)
parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=16)
parser.add_argument("-g", "--gpus", help="device", default=None)

args, unparsed = parser.parse_known_args()
args.gpus = utils.literal_eval(args.gpus)
batch_size = args.batch_size
beam_size = args.beam_size


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    stock = pd.read_hdf('~/scratch/molecule-datasets/zinc_stock_17_04_20.hdf5', key="table")  
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])

    test_set = load_dataset("test")
    model, reaction_dataset = load_model(beam_size)

    overall_result = np.zeros((beam_size, 2))
    depth_hit = np.zeros((20, beam_size, 2))

    logger.warning("Size of test set: %d" % len(test_set))
    for i in range(0, len(test_set), batch_size):
        # Initialization
        answer_set = []
        queue = []
        max_depth = 0
        for j in range(i, min(i+batch_size, len(test_set))):
            item = test_set[j] # (product_smiles, materials_smiles, depth)
            answer_set.append([])
            queue.append([{
                "id": j-i,
                "depth": 0,
                "score": 0.0,
                "products": [item["product"]],  # List of SMILES
                "starting_materials": [],
            }])
            max_depth = max(max_depth, item["depth"])

        # BFS
        for depth in range(max_depth):
            logger.warning("Index %d Depth %d" % (i, depth))
            nxt_queue = [[] for _ in range(len(queue))]
            items = []
            products = []
            for j, q in enumerate(queue):
                if depth >= test_set[i+j]["depth"]: continue
                if len(q) == 0: break
                items += q
                for item in q:
                    products += item["products"]
            reactants = []
            for j in range(0, len(products), batch_size):
                batch = get_batch(products[j:j+batch_size], reaction_dataset.kwargs)
                reactants += get_prediction(model, batch)
            k = 0
            for item in items:
                # Select the best k from all combinations of reactions
                new_products = select_topk(reactants[k:k+len(item["products"])], k=beam_size)                        
                k += len(item["products"])
                for (new_score, new_product) in new_products:
                    new_item = deepcopy(item)
                    new_item["depth"] += 1
                    new_item["score"] += new_score
                    new_item["products"] = []
                    for _product in new_product:
                        # Seperate starting and non-starting materials
                        if check_reactants_is_material(_product, stock_inchikeys):
                            new_item["starting_materials"].append(_product)
                        else:
                            new_item["products"].append(_product)
                    # Check whether we need to continue searching
                    if len(new_item["products"]) > 0:
                        nxt_queue[new_item["id"]].append(new_item)
                    else:
                        answer_set[new_item["id"]].append(new_item)
            queue = [sorted(q, key=lambda x: -x["score"])[:beam_size] for q in nxt_queue]
        
        # Calculate answers
        answer_set = [sorted(answers, key=lambda x: -x["score"])[:beam_size] for answers in answer_set]
        for answers, ground_truth in zip(answer_set, test_set[i:i+batch_size]):
            ground_truth_keys_list = [
                set([
                    Chem.MolToInchiKey(Chem.MolFromSmiles(target))[:14] for target in targets
                ]) for targets in ground_truth["targets"]
            ]
            overall_result[0, 1] += 1
            depth_hit[ground_truth["depth"], 0, 1] += 1
            for rank, answer in enumerate(answers):
                answer_keys = set([
                    Chem.MolToInchiKey(Chem.MolFromSmiles(material))[:14] 
                        for material in answer["starting_materials"]
                ])
                flag = False
                for ground_truth_keys in ground_truth_keys_list:
                    if ground_truth_keys == answer_keys:
                        overall_result[rank, 0] += 1
                        depth_hit[ground_truth["depth"], rank, 0] += 1
                        flag = True
                        break
                if flag: break
    overall_result = overall_result.cumsum(axis=-2)
    depth_hit = depth_hit.cumsum(axis=-2)
    print("overall_result: ", overall_result, overall_result[:, 0] / overall_result[:, 1])
    print("depth_hit: ", depth_hit, depth_hit[:, :, 0] / depth_hit[:, :, 1])
