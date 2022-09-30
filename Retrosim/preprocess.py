import json

chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"
vocab_size = len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }


def get_chars():
    return chars

def get_vocab_size():
    return vocab_size

def get_char_to_ix():
    return char_to_ix

def get_ix_to_char():
    return ix_to_char

def get_train_dataset(single_input):
    file_name = "train_dataset.json"
    products_list = []
    reactants_list = []
    retro_reaction_set = set()
    with open(file_name, 'r') as f:
        dataset = json.load(f)
        for _, reaction_trees in dataset.items():
            max_num_materials = 0
            final_retro_routes_list = None
            for i in range(1, int(reaction_trees['num_reaction_trees'])+1):
                if single_input:
                    if len(reaction_trees[str(i)]['materials']) > max_num_materials:
                        max_num_materials = len(reaction_trees[str(i)]['materials'])
                        final_retro_routes_list = reaction_trees[str(i)]['retro_routes']
                else:
                    retro_routes_list = reaction_trees[str(i)]['retro_routes']
                    for retro_route in retro_routes_list:
                        for retro_reaction in retro_route:
                            if retro_reaction not in retro_reaction_set:
                                retro_reaction_set.add(retro_reaction)
                                products_list.append(retro_reaction.split('>>')[0])
                                reactants_list.append(retro_reaction.split('>>')[1])

            if single_input:
                for retro_route in final_retro_routes_list:
                    for retro_reaction in retro_route:
                        if retro_reaction not in retro_reaction_set:
                            retro_reaction_set.add(retro_reaction)
                            products_list.append(retro_reaction.split('>>')[0])
                            reactants_list.append(retro_reaction.split('>>')[1])

    return products_list, reactants_list

