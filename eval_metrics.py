# -*- coding: utf-8 -*
from __future__ import print_function, absolute_import
from joblib import PrintTime
import numpy as np
import copy
import faiss
from tqdm import tqdm
import random
import torch
from sklearn.metrics import det_curve,roc_curve, DetCurveDisplay, RocCurveDisplay,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
    

def evaluate_rank(gf,qf,g_pids,q_pids,max_rank=50):
    num_q, num_g = len(q_pids),len(g_pids)
    gf=gf.numpy()
    qf=qf.numpy()
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    index = faiss.IndexFlatL2(128)
    index.add(gf)
    print('Index build finished,Num of index:',index.ntotal)
    D, I = index.search(qf, max_rank)

    all_cmc = []
    print(I[0],g_pids[I[0][0]],q_pids[0])
    for i in range(len(I)):
        for j in range(len(I[i])):
            I[i][j]=g_pids[I[i][j]]
            I[i][j]=(I[i][j]==q_pids[i])
        I[i]=np.asarray(I[i]).astype(np.int32)
        I[i]=I[i].cumsum()
        I[i][I[i]>1]=1
        all_cmc.append(I[i])
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_q
    return all_cmc

def evaluate_rank_2048(gf,qf,g_pids,q_pids,max_rank=50):
    num_q, num_g = len(q_pids),len(g_pids)
    gf=gf.numpy()
    qf=qf.numpy()
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    index = faiss.IndexFlatL2(2048)
    index.add(gf)
    print('Index build finished,Num of index:',index.ntotal)
    D, I = index.search(qf, max_rank)
    all_cmc = []
    print(I[0],g_pids[I[0][0]],q_pids[0])
    for i in range(len(I)):
        for j in range(len(I[i])):
            I[i][j]=g_pids[I[i][j]]
            I[i][j]=(I[i][j]==q_pids[i])
        I[i]=np.asarray(I[i]).astype(np.int32)
        I[i]=I[i].cumsum()
        I[i][I[i]>1]=1
        all_cmc.append(I[i])
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_q
    return all_cmc

def evaluate(distmat, q_pids, g_pids, max_rank=3):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    print("computing")
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]

        # compute cmc curve
        orig_cmc = matches[q_idx][:] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc[0:max_rank].cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        # num_rel = orig_cmc.sum()
        # tmp_cmc = orig_cmc.cumsum()
        # tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        # tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        # AP = tmp_cmc.sum() / num_rel
        # AP = 1.0 / (np.where(orig_cmc > 0)[0][0] + 1)
        # all_AP.append(AP)
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def _get_anchor_positive_triplet_mask(self, labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal

def evaluate_EER(gf,qf,g_pids,q_pids, threshold = 3, bestThreshold = 0.5, max_rank=50):
    num_q, num_g = len(q_pids),len(g_pids)
    # gf=gf.numpy()
    # qf=qf.numpy()
        
    #labels_equal = g_pids.unsqueeze(0) == q_pids.unsqueeze(1)
    labels_equal = np.expand_dims(g_pids, 0) == np.expand_dims(q_pids, 1)
    
    labels = labels_equal.ravel()

    
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    
    D, I = euclidean_dist(gf,qf)
    I = I[:,0:50]
    D = D.ravel()#展平
    # 歸一化
    max_dist = max(D)
    pred= 1-(D/max_dist)
    print(3)
    # 計算det_curve
    # fpr_det, fnr_det, thresholds_det = det_curve(labels, pred)
    # display = DetCurveDisplay(fpr=fpr_det, fnr=fnr_det, estimator_name="SVC" )
    # display.plot()
    # plt.show()
    # plt.close()
    
    print(4)

    # # 計算 roc_curve
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(labels, pred)
    # 計算 auc
    # auc_roc = auc(fpr_roc, tpr_roc)
    # display = RocCurveDisplay(fpr=fpr_roc, tpr=tpr_roc, roc_auc=auc_roc, estimator_name='example estimator')
    # display.plot()
    # plt.show()
    print(5)
    
    # 計算 EER Threshold
    fnr_roc = 1- tpr_roc
    ex = np.nanargmin(np.absolute((fnr_roc - fpr_roc)))
    EER_threshold = thresholds_roc[ex]
    print('EER Threshold=%f' % (thresholds_roc[ex]))
    eer1 = fpr_roc[ex]
    eer2 = fnr_roc[ex]
    
    print(eer1, eer2)
    print(6)
    
    # 計算 confusion_matrix
    # final_predics = [1 if i >= EER_threshold else 0 for i in pred]
    # cm = confusion_matrix(labels, final_predics)
    # cm_display = ConfusionMatrixDisplay(cm)
    # cm_display.plot()
    # plt.show()
    print(7)    

 
    # # 計算 accuracy
    # TN, FP, FN, TP = cm.ravel()
    # accuracy = (TP + TN) / (TN + FP + FN + TP)
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # print("accuracy : %f"% accuracy, "eer : %f"% eer)
    print(8)    
    
    
    all_cmc = []
    print(D[0],I[0],g_pids[I[0][0]],q_pids[0])
    for i in range(len(I)):
        for j in range(len(I[i])):
            I[i][j]=g_pids[I[i][j]]
            I[i][j]=(I[i][j]==q_pids[i])
        I[i]=np.asarray(I[i]).astype(np.int32)
        I[i]=I[i].cumsum()
        I[i][I[i]>1]=1
        all_cmc.append(I[i])
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_q
    return all_cmc, eer1

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    #dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    
    dist = dist.t()
    
    regionDM = np.array(dist)
        
    print(1)
    
    sortIndex = np.argsort(regionDM, axis=1)
    
    print(2)
    
    return regionDM, sortIndex.astype(int)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



