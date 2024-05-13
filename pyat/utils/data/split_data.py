import os.path
import random

""" data object format
    {
        'user_data': [{item_id: correct(int)}]
        'item_data': [{attr_name: attr_value}]
        'process_args': args
        'meta_data': {
            'num_users': int
            'num_items': int
            'num_knowledge': int
            'num_logs': int
            'user_no2id': {int: str}
            'item_no2id': {int: str}
            'knowledge_no2id': {int: str}
        }
    }
"""


def split_data_by_user(data_obj, split_ratio, random_seed=0, by_length=False):
    assert 0 < split_ratio < 1
    user_data = data_obj["user_data"]
    user_no2id = data_obj["meta_data"]["user_no2id"]
    user_idx = list(range(len(user_data)))
    if by_length:
        user_idx.sort(key=lambda k: len(user_data[k]))
    else:
        random.Random(random_seed).shuffle(user_idx)
    split_num = int(split_ratio * len(user_data))
    # get split data
    split_user_data = [user_data[i] for i in user_idx[:split_num]]
    split_num_users = len(split_user_data)
    split_user_no2id = {i: user_no2id[user_idx[i]] for i in range(split_num_users)}
    split_num_logs = sum(len(x) for x in split_user_data)
    # get rest data
    rest_user_data = [user_data[i] for i in user_idx[split_num:]]
    rest_num_users = len(rest_user_data)
    rest_user_no2id = {
        i: user_no2id[user_idx[i + split_num]] for i in range(rest_num_users)
    }
    rest_num_logs = sum(len(x) for x in rest_user_data)
    # check
    assert len(set(split_user_no2id.values()) & set(rest_user_no2id.values())) == 0
    assert set(split_user_no2id.values()) | set(rest_user_no2id.values()) == set(
        user_no2id.values()
    )
    assert split_num_logs + rest_num_logs == data_obj["meta_data"]["num_logs"]
    # return
    split_data = {
        "user_data": split_user_data,
        "item_data": data_obj["item_data"],
        "process_args": data_obj["process_args"],
        "meta_data": {
            **data_obj["meta_data"],
            "num_users": split_num_users,
            "num_logs": split_num_logs,
            "user_no2id": split_user_no2id,
        },
    }
    rest_data = {
        "user_data": rest_user_data,
        "item_data": data_obj["item_data"],
        "process_args": data_obj["process_args"],
        "meta_data": {
            **data_obj["meta_data"],
            "num_users": rest_num_users,
            "num_logs": rest_num_logs,
            "user_no2id": rest_user_no2id,
        },
    }
    return split_data, rest_data


def split_data_by_log(data_obj, split_ratio, random_seed=0):
    assert 0 < split_ratio < 1
    user_data = data_obj["user_data"]
    # get split data
    split_user_data, rest_user_data = [], []
    rand = random.Random(random_seed)
    for user_logs in user_data:
        all_items = list(user_logs.keys())
        rand.shuffle(all_items)
        split_user_logs = {
            i: user_logs[i] for i in all_items[: int(split_ratio * len(all_items))]
        }
        rest_user_logs = {
            i: user_logs[i] for i in all_items[int(split_ratio * len(all_items)) :]
        }
        split_user_data.append(split_user_logs)
        rest_user_data.append(rest_user_logs)
    split_num_logs = sum(len(x) for x in split_user_data)
    rest_num_logs = sum(len(x) for x in rest_user_data)
    # check
    assert split_num_logs + rest_num_logs == data_obj["meta_data"]["num_logs"]
    # return
    split_data = {
        "user_data": split_user_data,
        "item_data": data_obj["item_data"],
        "process_args": data_obj["process_args"],
        "meta_data": {**data_obj["meta_data"], "num_logs": split_num_logs},
    }
    rest_data = {
        "user_data": rest_user_data,
        "item_data": data_obj["item_data"],
        "process_args": data_obj["process_args"],
        "meta_data": {**data_obj["meta_data"], "num_logs": rest_num_logs},
    }
    return split_data, rest_data


if __name__ == "__main__":
    import sys
    import pickle

    sys.path.append("../..")
    import pyat.utils.global_logger as global_logger

    global_logger.init_logger()
    all_seeds = list(range(5))
    # all_datasets = ['ednet', 'junyi', 'assistment-2010', 'assistment-2017',
    #                 'nips-2020-1', 'nips-2020-2', 'math', 'ecpe', 'timss']
    all_datasets = ["ecpe", "timss"]
    global_logger.logger.info(f"All seeds: {all_seeds}")
    global_logger.logger.info(f"All datasets: {all_datasets}")
    for dataset in all_datasets:
        if not os.path.exists(f"../datasets/{dataset}/"):
            os.mkdir(f"../datasets/{dataset}/")
        for seed in all_seeds:
            global_logger.logger.info(f"Split dataset {dataset} with seed {seed}")
            with open(f"../datasets/{dataset}.pkl", "rb") as f:
                data = pickle.load(f)
            test_data, train_data = split_data_by_user(data, 0.2, random_seed=seed)
            global_logger.logger.info("Write data")
            with open(
                f"../datasets/{dataset}/{dataset}_train_seed_{seed}.pkl", "wb"
            ) as f:
                pickle.dump(train_data, f)
            with open(
                f"../datasets/{dataset}/{dataset}_test_seed_{seed}.pkl", "wb"
            ) as f:
                pickle.dump(test_data, f)
