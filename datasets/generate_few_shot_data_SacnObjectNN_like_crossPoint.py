import pickle
import numpy as np
import random
import os
import h5py

# gdown https://drive.google.com/uc?id=1r3nJ6gEu6cL59h7cIfMrBeY5p1P_JiIV
# tar -xzvf scanobjectnn.tar.gz

root = '../data/ScanObjectNN_cross_point/main_split'
target = '../data/ScanObjectNN_Fewshot_like_crossPoint_all'

train_data_path = os.path.join(root, 'train.h5')
test_data_path = os.path.join(root, 'test.h5')

# train
train_data_scan = h5py.File(train_data_path)
train_list_of_points = train_data_scan['data'][:].astype('float32')
train_list_of_labels = train_data_scan['label'][:].astype('int64')


def generate_fewshot_data(way, shot, prefix_ind, eval_sample=20):
    train_cls_dataset = {}
    test_cls_dataset = {}
    train_dataset = []
    test_dataset = []

    # build a dict containing different class
    train_num = len(train_list_of_points)
    for i in range(train_num):
        point = train_list_of_points[i]
        label = train_list_of_labels[i]
        if train_cls_dataset.get(label) is None:
            train_cls_dataset[label] = []
        train_cls_dataset[label].append(point)
    # build a dict containing different class

    print(sum([train_cls_dataset[i].__len__() for i in range(15)]))
    # import pdb; pdb.set_trace()
    keys = list(train_cls_dataset.keys())
    random.shuffle(keys)

    for i, key in enumerate(keys[:way]):
        train_data_list = train_cls_dataset[key]
        random.shuffle(train_data_list)
        assert len(train_data_list) >= (shot + eval_sample)
        for data in train_data_list[:shot]:
            train_dataset.append((data, i, key))

        # make fs_test from train_set, but the data is different
        for data in train_data_list[shot:(shot + eval_sample)]:
            test_dataset.append((data, i, key))

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)
    dataset = {
        'train': train_dataset,
        'test': test_dataset
    }
    save_path = os.path.join(target, f'{way}way_{shot}shot')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, f'{prefix_ind}.pkl'), 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    # ways = [5, 10]
    # shots = [10, 20]
    # for way in ways:
    #     for shot in shots:
    #         for i in range(10):
    #             generate_fewshot_data(way=way, shot=shot, prefix_ind=i)

    ways = [5, 10, 15]
    shots = [1, 3, 5, 10, 20, 40]
    for way in ways:
        for shot in shots:
            for i in range(10):
                generate_fewshot_data(way=way, shot=shot, prefix_ind=i)