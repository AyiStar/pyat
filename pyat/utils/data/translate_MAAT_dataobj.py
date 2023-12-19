# This script is totally specific for equivalently translating the dataset released in previous project
# https://github.com/AyiStar/MAAT/tree/master/datasets/assistment
import json
import csv
import logging
import itertools
import functools
import operator
from typing import Tuple, Dict, List

import pandas as pd
import pickle
from prettytable import PrettyTable


def triplets_to_user_data_obj(triplets: [Tuple[int, int, int]]) -> List[Dict]:
    data_obj = []
    for user_no, item_no, correct in sorted(triplets, key=lambda t: t[0]):
        if user_no >= len(data_obj):
            assert(user_no == len(data_obj))
            data_obj.append({})
        data_obj[user_no][item_no] = correct
    return data_obj


def concept_map_to_item_data_obj(concept_map: Dict[str, List[int]]) -> List[Dict]:
    data_obj = []
    for item_no in sorted(concept_map.keys()):
        if item_no >= len(data_obj):
            assert(item_no == len(data_obj))
            data_obj.append({'knowledge': concept_map[item_no]})
        data_obj[item_no]
    return data_obj


def show_dataset_info(train_dataobj: Dict,
                      test_dataobj: Dict,
                      logger: logging.Logger):
    info_table = PrettyTable()
    info_table.add_column("assistment-MAAT",
                          ["#Users",
                           "#Items",
                           "#Knolwedge",
                           "#Logs",
                           "#Avg.Logs",
                           "#Min.Logs",
                           "#Max.Logs"])
    info_table.add_column("Train",
                          [train_dataobj['meta_data']['num_users'],
                           train_dataobj['meta_data']['num_items'],
                           train_dataobj['meta_data']['num_knowledge'],
                           train_dataobj['meta_data']['num_logs'],
                           sum(list(map(len, train_dataobj['user_data']))) / len(train_dataobj['user_data']),
                           min(list(map(len, train_dataobj['user_data']))),
                           max(list(map(len, train_dataobj['user_data'])))])
    info_table.add_column('Test',
                          [test_dataobj['meta_data']['num_users'],
                           test_dataobj['meta_data']['num_items'],
                           test_dataobj['meta_data']['num_knowledge'],
                           test_dataobj['meta_data']['num_logs'],
                           sum(list(map(len, test_dataobj['user_data']))) / len(test_dataobj['user_data']),
                           min(list(map(len, test_dataobj['user_data']))),
                           max(list(map(len, test_dataobj['user_data'])))])
    info_table.float_format = '.2'
    logger.info('\n' + info_table.get_string())



train_triplets = pd.read_csv(f'./train_triplets.csv', encoding='utf-8').to_records(index=False)
test_triplets = pd.read_csv(f'./test_triplets.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'./concept_map.json', 'r'))
concept_map = {int(k):v for k,v in concept_map.items()}
metadata = json.load(open(f'./metadata.json', 'r'))

train_user_data = triplets_to_user_data_obj(train_triplets)
test_user_data = triplets_to_user_data_obj(test_triplets)
item_data = concept_map_to_item_data_obj(concept_map)

all_knowledge = functools.reduce(set.union, map(set, concept_map.values()))
train_dataobj = {
    'user_data': train_user_data,
    'item_data': item_data,
    'meta_data': {
        'num_users': len(train_user_data),
        'num_items': len(item_data),
        'num_knowledge': len(all_knowledge),
        'num_logs': sum(map(len, train_user_data)),
        'user_no2id': {x: str(x) for x in range(len(train_user_data))},
        'item_no2id': {x: str(x) for x in range(len(item_data))},
        'knowledge_no2id': {x: str(x) for x in all_knowledge}
    },
    'process_args': {
        
    }
}

test_dataobj = {
    'user_data': test_user_data,
    'item_data': item_data,
    'meta_data': {
        'num_users': len(test_user_data),
        'num_items': len(item_data),
        'num_knowledge': len(all_knowledge),
        'num_logs': sum(map(len, test_user_data)),
        'user_no2id': {x: str(x + train_dataobj['meta_data']['num_users']) for x in range(len(test_user_data))},
        'item_no2id': {x: str(x) for x in range(len(item_data))},
        'knowledge_no2id': {x: str(x) for x in all_knowledge}
    },
    'process_args': {
        'min_logs_per_student': 150
    }
}

print(metadata)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
show_dataset_info(train_dataobj, test_dataobj, logging)

with open('assistment-MAAT_train_seed_0.pkl', 'wb') as f:
    pickle.dump(train_dataobj, f)

with open('assistment-MAAT_test_seed_0.pkl', 'wb') as f:
    pickle.dump(test_dataobj, f)