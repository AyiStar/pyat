import torch.utils.data as data


""" Processed data format
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


class LogDataset(data.Dataset):

    def __init__(self, data_obj):
        super().__init__()
        self.user_data = data_obj['user_data']
        self.item_data = data_obj['item_data']

        self.num_users = data_obj['meta_data']['num_users']
        self.num_items = data_obj['meta_data']['num_items']
        self.num_knowledge = data_obj['meta_data']['num_knowledge']
        self.num_logs = data_obj['meta_data']['num_logs']
        self.user_no2id = data_obj['meta_data']['user_no2id']
        self.item_no2id = data_obj['meta_data']['item_no2id']
        self.knowledge_no2id = data_obj['meta_data']['knowledge_no2id']

        self.log_data = []
        for user_no, user_logs in enumerate(self.user_data):
            for item_no, correct in user_logs.items():
                self.log_data.append((user_no, item_no, correct))

    def __len__(self):
        return len(self.log_data)

    def __getitem__(self, item):
        return self.log_data[item]
