import os
import sys
import argparse
import time
import datetime
import numpy as np

import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import *
from torch.optim import lr_scheduler

from tqdm import tqdm
from losses import OnlineTripletLoss, OnlineContrastiveLoss, ContrastiveLoss, OnlineContrastiveLoss, OnlineSimLoss,TripletLoss
import models
from models import model
import data_manager
from img_loader import ImageDataset,ImageDataset_aug
from torchreid.data.sampler import RandomIdentitySampler
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.torchtools import save_checkpoint
from eval_metrics import evaluate,evaluate_rank, evaluate_EER,plot_confusion_matrix

sys.path.append("./")
parser = argparse.ArgumentParser(description="Using GA-ICDNet train gait model with triplet-loss and sim-loss and reconst-loss")
# Dataset
parser.add_argument('--max-epoch', default=500, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="start epochs to run")
parser.add_argument("-j", "--workers", default=4, type=int, help="number of data loading workers(default: 4)")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--cooperative', default=0, type=int, help='whether the probe set only consists of subject with bags')
#parser.add_argument('--cooperative', default=1,type=int)
# optimization options
parser.add_argument("--train_batch", default=300, type=int)
parser.add_argument("--test_batch", default=300, type=int)
parser.add_argument("--lr", '--learning-rate', default=0.0002, type=float)
parser.add_argument("--weight-decay", default=5e-4, type=float)
parser.add_argument("--save-dir", default='save_dir', type=str)
# Architecture

# Miscs


args = parser.parse_args()
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)


def main():
    torch.manual_seed(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    print(args)
    # GPU / CPU
    device = torch.device('cuda')

    print("Initializing dataset")
    dataset = data_manager.init_dataset('../dataSet/OULP_Bag_dataset_GEI','id_list.csv',args.cooperative)

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
   
    # test/val queryLoader
    # test/val galleryLoader
    test_probeLoader = DataLoader(
        ImageDataset(dataset.test_probe, sample='dense', transform=transform_test),
        shuffle=False, batch_size=args.test_batch, drop_last=False
    )

    test_galleryLoader = DataLoader(
        ImageDataset(dataset.test_gallery, sample='dense', transform=transform_test),
        shuffle=False, batch_size=args.test_batch, drop_last=False
    )
    model = models.model.ICDNet_group_mask_mask_early_8().to(device=device)
    #model = models.model.ICDNet_mask()
    #model= nn.DataParallel(model).cuda()
    #model = models.model.icdnet().to(device=device)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    path = "C:/TANG/repository/GA-ICDNet/backup/GA_batch300300_uncooperative_trip_loss-1_sim_loss-0.1_recon_Loss-500_label_loss-0.05/save_dir"
    checkpoint = torch.load(f'{path}/model-best.pth.tar')
    test_f = open(f'{path}/t_new.txt', "w")
    model.load_state_dict(checkpoint['state_dict'])
    
    start_time = time.time()


    print("=============> Test")
    rank1, EER, correct_rate = test(model, test_probeLoader, test_galleryLoader, device, test_f,path)
    
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def test(model, queryLoader, galleryLoader, device, test_f, path, ranks=[1, 5, 10, 20]):
    with torch.no_grad():
        model.train(False)
        model.eval()
        correct = 0.
        total = 0.
        total_correct=0.
        qf, q_pids, q_bags = [], [], []
        all_predicted, all_bag = [], []
        for batch_idx, (img, pid, bag) in tqdm(enumerate(queryLoader)):
            #total += 1.0
            img = img.to(device=device, dtype=torch.float)
            bag = bag.to(device=device, dtype=torch.long)
            _, _, features,output_bag = model(img)
            #print(output_bag.shape)
            _, predicted = torch.max(output_bag, 1)
            # print('test_predicted:')
            # print(predicted)
            # print('test_label:')
            # print(bag)
            # print('================================================================')
            # print(type(predicted),type(bag))
            # exit()
            all_predicted+=predicted.tolist()
            all_bag+=bag.tolist()
            correct=(predicted == bag).sum()
            total+=img.shape[0]
            total_correct+=correct


            features = features.squeeze(0)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pid)
            q_bags.extend(bag)

        qf = torch.cat([x for x in qf])#torch.stack(qf)
        q_pids = np.asarray(q_pids)
        # q_pids = torch.tensor(q_pids)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_bags = [], [], []
        for batch_idx, (img, pid, bag) in tqdm(enumerate(galleryLoader)):
            #total += 1.0
            img = img.to(device=device, dtype=torch.float)
            bag = bag.to(device=device, dtype=torch.long)
            _, _, features,output_bag = model(img)
            _, predicted = torch.max(output_bag, 1)
            # print('test_predicted:')
            # print(predicted)
            # print('test_label:')
            # print(bag)

            all_predicted+=predicted.tolist()
            all_bag+=bag.tolist()
            correct=(predicted == bag).sum()
            total+=img.shape[0]
            total_correct+=correct

            features = features.squeeze(0)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pid)
            g_bags.extend(bag)
        gf =torch.cat([x for x in gf]) #torch.stack(gf)
        g_pids = np.asarray(g_pids)
        #g_pids = torch.tensor(g_pids)
        
        qf=qf.squeeze()
        gf=gf.squeeze() # 29102*128
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("Computing distance matrix")
        

        import matplotlib.pyplot as plt
        from sklearn.metrics import det_curve,roc_curve, DetCurveDisplay, RocCurveDisplay,confusion_matrix,ConfusionMatrixDisplay
        print(len(all_bag), len(all_predicted))
        
        cnf_matrix = confusion_matrix(all_bag, all_predicted,labels=[0, 1, 2, 3, 4, 5, 6])
        print(f's : = {cnf_matrix.sum()}')
        
        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        plt.figure()

        plot_confusion_matrix(cnf_matrix, classes=["NoCO", "FrCO","BaCO", "SmCO", "SbCO", "MuCO", "CpCO"],
                    title='Confusion matrix, without normalization')
        plt.savefig(path+"/CM.png")
        
        #cmc= evaluate_rank(gf,qf,g_pids,q_pids)
        cmc1, eer = evaluate_EER(gf,qf,g_pids,q_pids)
        '''
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()
        cmc, mAP = evaluate(distmat, q_pids, g_pids)
        '''
        
        # print("Results ----------")
        # #print("mAP: {:.1%}".format(mAP))
        # EER = 0
        # print("CMC curve")
        # for r in ranks:
        #     print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
            
        print("Results ----------")
        #print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.4%}".format(r, cmc1[r - 1]))
            test_f.write("Rank-{:<3}: {:.4%}\n".format(r, cmc1[r - 1]))
            
        
        print("\n")
        print(total_correct.float(),total)
        print("correct_rate:", total_correct.float()*1.0/total)
        test_f.write(f"\n\n============================\nEER = {eer}\ncorrect_rate = {total_correct.float()*1.0/total}\n")
        
        
        return cmc1[0], eer, total_correct.float()*1.0/total


if __name__ == '__main__':
    main()