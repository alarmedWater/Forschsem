import argparse
import torch
from codes import mvtecad
from functools import reduce
from torch.utils.data import DataLoader
from codes.datasets import *
from codes.networks import *
from codes.inspection import eval_encoder_NN_multiK, infer  # Importiere infer
from codes.utils import *
from sklearn.neighbors import NearestNeighbors
import numpy as np
from codes.nearest_neighbor import search_NN

parser = argparse.ArgumentParser()

parser.add_argument('--obj', default='bottle', type=str)  # Standardwert auf 'bottle' geändert
parser.add_argument('--lambda_value', default=1, type=float)
parser.add_argument('--D', default=64, type=int)

parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr', default=1e-4, type=float)

args = parser.parse_args()

DATASET_PATH = r"C:\Users\Maxi\Documents\Forschsem\Erdbeeren\Patch-level SVDD for Anomaly Detection and Segmentation\Anomaly-Detection-PatchSVDD-PyTorch\ckpts"

def train():
    obj = args.obj
    D = args.D
    lr = args.lr
        
    with task('Networks'):
        enc = EncoderHier(64, D).cuda()
        cls_64 = PositionClassifier(64, D).cuda()
        cls_32 = PositionClassifier(32, D).cuda()

        modules = [enc, cls_64, cls_32]
        params = [list(module.parameters()) for module in modules]
        params = reduce(lambda x, y: x + y, params)

        opt = torch.optim.Adam(params=params, lr=lr)

    with task('Datasets'):
        train_x = mvtecad.get_x_standardized(obj, mode='train')
        train_x = NHWC2NCHW(train_x)

        rep = 100
        datasets = dict()
        datasets[f'pos_64'] = PositionDataset(train_x, K=64, repeat=rep)
        datasets[f'pos_32'] = PositionDataset(train_x, K=32, repeat=rep)
        
        datasets[f'svdd_64'] = SVDD_Dataset(train_x, K=64, repeat=rep)
        datasets[f'svdd_32'] = SVDD_Dataset(train_x, K=32, repeat=rep)

        dataset = DictionaryConcatDataset(datasets)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    print('Start training')
    for i_epoch in range(args.epochs):
        if i_epoch != 0:
            for module in modules:
                module.train()

            for d in loader:
                d = to_device(d, 'cuda', non_blocking=True)
                opt.zero_grad()

                loss_pos_64 = PositionClassifier.infer(cls_64, enc, d['pos_64'])
                loss_pos_32 = PositionClassifier.infer(cls_32, enc.enc, d['pos_32'])
                loss_svdd_64 = SVDD_Dataset.infer(enc, d['svdd_64'])
                loss_svdd_32 = SVDD_Dataset.infer(enc.enc, d['svdd_32'])

                loss = loss_pos_64 + loss_pos_32 + args.lambda_value * (loss_svdd_64 + loss_svdd_32)

                loss.backward()
                opt.step()

        aurocs = eval_encoder_NN_multiK(enc, obj)
        log_result(obj, aurocs)
        enc.save(obj)

    return enc


def log_result(obj, aurocs):
    det_64 = aurocs['det_64'] * 100
    seg_64 = aurocs['seg_64'] * 100

    det_32 = aurocs['det_32'] * 100
    seg_32 = aurocs['seg_32'] * 100

    det_sum = aurocs['det_sum'] * 100
    seg_sum = aurocs['seg_sum'] * 100

    det_mult = aurocs['det_mult'] * 100
    seg_mult = aurocs['seg_mult'] * 100

    print(f'|K64| Det: {det_64:4.1f} Seg: {seg_64:4.1f} |K32| Det: {det_32:4.1f} Seg: {seg_32:4.1f} |mult| Det: {det_sum:4.1f} Seg: {seg_sum:4.1f} |mult| Det: {det_mult:4.1f} Seg: {seg_mult:4.1f} ({obj})')


def search_NN_sklearn(test_emb, train_emb, NN=5):
    nbrs = NearestNeighbors(n_neighbors=NN, algorithm='auto').fit(train_emb)
    distances, indices = nbrs.kneighbors(test_emb)
    return distances, indices

def search_NN(test_emb, train_emb_flat, method='sklearn', NN=5):
    if method == 'sklearn':
        return search_NN_sklearn(test_emb, train_emb_flat, NN=NN)
    else:
        raise ValueError(f"Unsupported method: {method}")


def measure_emb_NN(emb_te, emb_tr, NN=1):
    from codes.nearest_neighbor import search_NN
    D = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, D)
    print(f"train_emb_all shape: {train_emb_all.shape}")
    print(f"emb_te shape: {emb_te.shape}")

    l2_maps, _ = search_NN(emb_te, train_emb_all, NN=NN)
    print(f"l2_maps shape: {l2_maps.shape}")
    anomaly_maps = np.mean(l2_maps, axis=-1)

    return anomaly_maps


if __name__ == '__main__':
    print(f"DATASET_PATH: {mvtecad.DATASET_PATH}")
    print(f"Objekt: {args.obj}")  # Verwende args.obj statt obj

    obj = args.obj
    x_tr = mvtecad.get_x_standardized(obj, mode='train')
    x_tr = NHWC2NCHW(x_tr)

    x_te = mvtecad.get_x_standardized(obj, mode='test')
    x_te = NHWC2NCHW(x_te)

    enc = train()  # Hole enc aus der Funktion train()

    embs32_tr = infer(x_tr, enc.enc, K=32, S=8)  # Erhöhe den Schritt S
    embs32_te = infer(x_te, enc.enc, K=32, S=8)
