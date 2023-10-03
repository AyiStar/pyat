import torch.utils.data as data


class SessionDataset(data.Dataset):

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

        self.session_data = []
        for user_no, user_logs in enumerate(self.user_data):
            # item_ids = list(user_logs.keys())
            # labels = [user_logs[item_id] for item_id in item_ids]
            # self.session_data.append((user_no, item_ids, labels))
            session = {
                'user_no': user_no,
                'user_id': self.user_no2id[user_no],
                'all_logs': user_logs
            }
            self.session_data.append(session)

    def __len__(self):
        return len(self.session_data)

    def __getitem__(self, item):
        return self.session_data[item]


def session_dataset_collate_fn(batch):
    return batch
