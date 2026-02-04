import torch
import numpy as np
import copy
from sklearn.cluster import DBSCAN

def ensemble_cluster(neups, ddifs, biases):
    biases = np.array([bias.cpu().numpy() for bias in biases])
    N = len(neups)
    # 使用DBSCAN对偏置进行聚类
    cosine_labels = DBSCAN(min_samples=3, metric='cosine').fit(biases).labels_

    neup_labels = DBSCAN(min_samples=3).fit(neups).labels_

    ddif_labels = DBSCAN(min_samples=3).fit(ddifs).labels_

    # 计算各聚类的距离
    dists_from_cluster = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            dists_from_cluster[i, j] = (int(cosine_labels[i] == cosine_labels[j]) + int(
                neup_labels[i] == neup_labels[j]) + int(ddif_labels[i] == ddif_labels[j])) / 3.0
            dists_from_cluster[j, i] = dists_from_cluster[i, j]

    ensembled_labels = DBSCAN(min_samples=3, metric='precomputed').fit(dists_from_cluster).labels_

    return ensembled_labels

def deepsight_average(global_model, client_models, chosen_id, helper):
    """
    Perform FedAvg algorithm and perform some clustering on top of it.
    """
    weight_accumulator = create_weight_accumulator(helper)
    for i in range(len(chosen_id)):
        for name, data in client_models[chosen_id[i]].state_dict().items():
            if name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - global_model.state_dict()[name])
    lr = 1

    for name, data in global_model.state_dict().items():
        if name == 'decoder.weight':
            continue
        update_per_layer = weight_accumulator[name] * (1/len(chosen_id)) * lr
        update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
        data.add_(update_per_layer.cuda())

    return True

def create_weight_accumulator(helper):
    weight_accumulator = dict()
    for name, data in helper.global_model.state_dict().items():
        ### don't scale tied weights:
        if name == 'decoder.weight' or '__'in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)
    return weight_accumulator

def deepsight_aggregate_global_model_v2(clients, chosen_ids, global_model, weight_accumulator_by_client, helper):
    '''
    使用DeepSight算法聚合全局模型
    '''
    global_weight = list(global_model.state_dict().values())[-2]
    global_bias = list(global_model.state_dict().values())[-1]

    biases = [(list(clients[i].state_dict().values())[-1] - global_bias) for i in chosen_ids]
    weights = [list(clients[i].state_dict().values())[-2] for i in chosen_ids]

    n_client = len(chosen_ids)
    cosine_similarity_dists = np.array((n_client, n_client))
    neups = list()
    n_exceeds = list()

    # calculate neups
    sC_nn2 = 0
    for i in range(len(chosen_ids)):
        C_nn = torch.sum(weights[i]-global_weight, dim=[1]) + biases[i]-global_bias
        C_nn2 = C_nn * C_nn
        neups.append(C_nn2)
        sC_nn2 += C_nn2
        
        C_max = torch.max(C_nn2).item()
        threshold = 0.01 * C_max if 0.01 > (1 / len(biases)) else 1 / len(biases) * C_max
        n_exceed = torch.sum(C_nn2 > threshold).item()
        n_exceeds.append(n_exceed)
    # normalize
    neups = np.array([(neup/sC_nn2).cpu().numpy() for neup in neups])
    
    rand_input = None
    if helper.config.dataset == 'cifar10' or helper.config.dataset == 'GTSRB':
        rand_input = torch.randn((256, 3, 32, 32)).cuda()
    elif helper.config.dataset == 'tiny-imagenet':
        rand_input = torch.randn((128, 3, 64, 64)).cuda()

    global_ddif = torch.mean(torch.softmax(global_model(rand_input), dim=1), dim=0)
    client_ddifs = [torch.mean(torch.softmax(clients[i](rand_input), dim=1), dim=0)/ global_ddif
                    for i in chosen_ids]
    client_ddifs = np.array([client_ddif.cpu().detach().numpy() for client_ddif in client_ddifs])

    # use n_exceed to label
    classification_boundary = np.median(np.array(n_exceeds)) / 2
    
    identified_mals = [int(n_exceed <= classification_boundary) for n_exceed in n_exceeds]
    
    clusters = ensemble_cluster(neups, client_ddifs, biases)
    cluster_ids = np.unique(clusters)

    deleted_cluster_ids = list()
    for cluster_id in cluster_ids:
        n_mal = 0
        cluster_size = np.sum(cluster_id == clusters)
        for identified_mal, cluster in zip(identified_mals, clusters):
            if cluster == cluster_id and identified_mal:
                n_mal += 1
        if (n_mal / cluster_size) >= (1 / 3):
            deleted_cluster_ids.append(cluster_id)
    
    temp_chosen_ids = copy.deepcopy(chosen_ids)
    for i in range(len(chosen_ids)-1, -1, -1):
        if clusters[i] in deleted_cluster_ids:
            del chosen_ids[i]

    print("final clients length:{}".format(len(chosen_ids)))
    if len(chosen_ids)==0:
        chosen_ids = temp_chosen_ids
    
    for i in range(len(temp_chosen_ids)):
        if temp_chosen_ids[i] in chosen_ids:
            for name, data in global_model.state_dict().items():
                if name == 'decoder.weight':
                    continue
                update_per_layer = weight_accumulator_by_client[i][name] * \
                                (1/len(chosen_ids)) 
                update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
                data.add_(update_per_layer.cuda())

    return True 