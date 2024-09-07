import numpy as np




def partition_data(dataset, partition= "hetero-dir", alpha=0.5, args=None, test_dataset=None):
    np.random.seed(0)
    X_train = dataset.data
    y_train = np.array(dataset.targets)
    n_train = y_train.shape[0]
    num_clients = args.world_size
    if test_dataset is not None:
        X_test = test_dataset.data
        y_test = np.array(test_dataset.targets)
        n_test = y_test.shape[0]
        net_dataidx_map_test = {}
    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, num_clients)
        net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}

    elif partition == "hetero-dir":
        min_size = 0
        K = 10 #number_class
        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(num_clients)]
            # for each class in the dataset
            if test_dataset is not None:
                idx_batch_test = [[] for _ in range(num_clients)]
                N_test = y_test.shape[0]

            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                if test_dataset is not None:
                    proportions_test = np.copy(proportions)
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

                if test_dataset is not None:
                    idx_k_test = np.where(y_test == k)[0]
                    np.random.shuffle(idx_k_test)
                    # proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

                    proportions_test = np.array(
                        [p * (len(idx_j) < N_test / num_clients) for p, idx_j in zip(proportions_test, idx_batch_test)])
                    proportions_test = proportions_test / proportions_test.sum()
                    proportions_test = (np.cumsum(proportions_test) * len(idx_k_test)).astype(int)[:-1]
                    idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(idx_k_test, proportions_test))]
                    # min_size = min([len(idx_j) for idx_j in idx_batch_test])

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

            if test_dataset is not None:
                np.random.shuffle(idx_batch_test[j])
                net_dataidx_map_test[j] = idx_batch_test[j]
    if test_dataset is not None:
        return net_dataidx_map, net_dataidx_map_test
    else:
        return net_dataidx_map