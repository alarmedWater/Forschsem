from codes import mvtecad
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import PatchDataset_NCHW, NHWC2NCHW, distribute_scores


__all__ = ['eval_encoder_NN_multiK', 'eval_embeddings_NN_multiK']


def infer(x, enc, K, S):
    print(f"Input shape: {x.shape}")
    x = NHWC2NCHW(x)
    print(f"Converted shape: {x.shape}")
    dataset = PatchDataset_NCHW(x, K=K, S=S)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)
    embs = np.empty((dataset.N, dataset.row_num, dataset.col_num, enc.D), dtype=np.float32)
    enc = enc.eval()
    with torch.no_grad():
        for xs, ns, iis, js in loader:
            xs = xs.cuda()
            embedding = enc(xs)
            embedding = embedding.detach().cpu().numpy()

            for embed, n, i, j in zip(embedding, ns, iis, js):
                embs[n, i, j] = np.squeeze(embed)
    print(f"Embeddings shape: {embs.shape}")
    return embs


def assess_anomaly_maps(obj, anomaly_maps):
    auroc_seg = mvtecad.segmentation_auroc(obj, anomaly_maps)

    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    auroc_det = mvtecad.detection_auroc(obj, anomaly_scores)
    return auroc_det, auroc_seg


#########################

def eval_encoder_NN_multiK(enc, obj):
    x_tr = mvtecad.get_x_standardized(obj, mode='train')
    x_te = mvtecad.get_x_standardized(obj, mode='test')

    embs64_tr = infer(x_tr, enc, K=64, S=16)
    embs64_te = infer(x_te, enc, K=64, S=16)

    embs32_tr = infer(x_tr, enc.enc, K=32, S=4)
    embs32_te = infer(x_te, enc.enc, K=32, S=4)

    print(f"embs64_tr shape: {embs64_tr.shape}, embs64_te shape: {embs64_te.shape}")
    print(f"embs32_tr shape: {embs32_tr.shape}, embs32_te shape: {embs32_te.shape}")

    embs64 = embs64_tr, embs64_te
    embs32 = embs32_tr, embs32_te

    return eval_embeddings_NN_multiK(obj, embs64, embs32)


def eval_embeddings_NN_multiK(obj, embs64, embs32, NN=1):
    emb_tr, emb_te = embs64
    print(f"embs64 train shape: {emb_tr.shape}, test shape: {emb_te.shape}")
    maps_64 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN)
    print(f"maps_64 shape: {maps_64.shape}")

    emb_tr, emb_te = embs32
    print(f"embs32 train shape: {emb_tr.shape}, test shape: {emb_te.shape}")
    maps_32 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN)
    print(f"maps_32 shape: {maps_32.shape}")

    return {
        'maps_64': maps_64,
        'maps_32': maps_32,
    }


########################

def measure_emb_NN(emb_te, emb_tr, method='kdt', NN=1):
    from .nearest_neighbor import search_NN
    D = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, D)
    print(f"train_emb_all shape: {train_emb_all.shape}")
    print(f"emb_te shape: {emb_te.shape}")

    l2_maps, _ = search_NN(emb_te, train_emb_all, NN=NN)
    print(f"l2_maps shape: {l2_maps.shape}")
    anomaly_maps = np.mean(l2_maps, axis=-1)

    return anomaly_maps
from sklearn.neighbors import KDTree
import numpy as np

__all__ = ['search_NN']

def search_NN(test_emb, train_emb_flat, NN=1):
    kdt = KDTree(train_emb_flat)

    Ntest, I, J, D = test_emb.shape
    closest_inds = np.empty((Ntest, I, J, NN), dtype=np.int32)
    l2_maps = np.empty((Ntest, I, J, NN), dtype=np.float32)

    for n in range(Ntest):
        for i in range(I):
            dists, inds = kdt.query(test_emb[n, i, :, :], return_distance=True, k=NN)
            closest_inds[n, i, :, :] = inds[:, :]
            l2_maps[n, i, :, :] = dists[:, :]

    return l2_maps, closest_inds
