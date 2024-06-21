# coding=utf-8
import faiss

from sklearn.mixture import GaussianMixture
import numpy as np
import random


def select_clean_samples(x, y, nb_prototypes, nb_classes=2):
    """
    Clean the dataset
    :param x: features of all examples in the dataset
    :param y: original labels
    :param args: hyper-parameter
    :return:
    """
    prototypes = get_prototypes(x, y, nb_prototypes)
    prototypes = np.vstack(prototypes)  # combine prototypes of all classes
    # prototypes = preprocessing.normalize(prototypes)  # normalize the class prototypes

    # each item x_{ij} represents the similarity between the sample i and the prototype j
    similarities_proto = x.dot(prototypes.T)
    similarities_class = np.zeros((x.shape[0], nb_classes), dtype=np.float64)
    for i in range(nb_classes):
        similarities_class[:, i] = np.mean(
            similarities_proto[:, i * nb_prototypes:(i + 1) * nb_prototypes], axis=1)
    # select the samples by GMM
    clean_set = []
    for i in range(nb_classes):
        class_idx = np.where(y == i)[0]
        class_sim = similarities_proto[class_idx, i]
        # split the dataset using GMM
        class_sim = class_sim.reshape((-1, 1))
        # gm = GaussianMixture(n_components=2, random_state=args.seed).fit(class_sim)
        gm = GaussianMixture(n_components=2, random_state=13).fit(class_sim)
        class_clean_idx = np.where(gm.predict(class_sim) == gm.means_.argmax())[0]
        clean_set.extend(class_idx[class_clean_idx])

    return x[clean_set], y[clean_set], clean_set


def get_prototypes(x, y, nb_prototypes, nb_classes=2):
    """
    Obtain class prototypes for each category via KMeans
    :param x: features for all examples
    :param y: targets for all examples (We use pseudo-label here)
    :param args: hyper-parameters
    :return:
    """
    cluster = []  # the set of class prototypes of the categories
    for i in range(nb_classes):
        y_x = x[y == i]  # the features of class i
        if len(y_x) < nb_prototypes:
            cluster.append(y_x)
        else:
            cluster.append(kmeans(y_x, nb_prototypes, nb_classes=2))
    return cluster


def kmeans(x, nb_prototypes, nb_classes=2):
    # obtain class prototypes by K-Means
    d = x.shape[1]
    k = int(nb_prototypes)
    cluster = faiss.Clustering(d, k)
    cluster.verbose = True
    cluster.niter = 20
    cluster.nredo = 5

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    index = faiss.GpuIndexFlatIP(res, d, cfg)

    cluster.train(x.astype(np.float32), index)
    centroids = faiss.vector_to_array(cluster.centroids).reshape(k, d)

    return centroids