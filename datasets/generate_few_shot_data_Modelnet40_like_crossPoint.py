import pickle
import numpy as np
import random
import os
import glob
import h5py

target = './data/ModelNetFewshot_like_crossPoint_all'
DATA_DIR = './data/ModelNet_crossPoint'


def load_modelnet_data(partition):


    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

train_list_of_points, train_list_of_labels = load_modelnet_data('train')



def generate_fewshot_data(way, shot, prefix_ind, eval_sample=20):
    train_cls_dataset = {}
    test_cls_dataset = {}
    train_dataset = []
    test_dataset = []


    # build a dict containing different class
    train_num = len(train_list_of_points)
    for i in range(train_num):
        point = train_list_of_points[i]
        label = train_list_of_labels[i][0]
        if train_cls_dataset.get(label) is None:
            train_cls_dataset[label] = []
        train_cls_dataset[label].append(point)

    print(sum([train_cls_dataset[i].__len__() for i in range(40)]))
    keys = list(train_cls_dataset.keys())
    random.shuffle(keys)

    for i, key in enumerate(keys[:way]):
        train_data_list = train_cls_dataset[key]
        random.shuffle(train_data_list)
        assert len(train_data_list) >= (shot+eval_sample)
        for data in train_data_list[:shot]:
            train_dataset.append((data, i, key))

        # make fs_test from train_set, but the data is different
        for data in train_data_list[shot:(shot+eval_sample)]:
            test_dataset.append((data, i, key))

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)
    dataset = {
        'train': train_dataset,
        'test' : test_dataset
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
    #             generate_fewshot_data(way = way, shot = shot, prefix_ind = i)

    ways = [5, 10, 20 ,40]
    shots = [1, 3, 5, 10, 20, 40]
    for way in ways:
        for shot in shots:
            for i in range(10):
                generate_fewshot_data(way = way, shot = shot, prefix_ind = i)
