import argparse
import numpy as np
import random
import time
import torch
import torch.optim as optim
import torch.nn as nn

# [新增] 引入 SupConLoss
from supcon_loss import SupConLoss
from loss import Loss
from dataloader import IEMOCAPRobertaDataset
from model import ECERC
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import warnings


warnings.filterwarnings("ignore")


def seed_everything(seed=2007):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    # np.random.shuffle(idx)  # shuffle for training data
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_bert_loaders(path=None, batch_size=32, num_workers=0, pin_memory=False, valid_rate=0.1):
    trainset = IEMOCAPRobertaDataset(train=True)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPRobertaDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


# [修改] 新增參數 supcon_criterion
def train_or_eval_model(model, loss_f, dataloader, train_flag=False, optimizer=None, cuda_flag=False, feature_type='text',
                        target_names=None, scheduler=None, supcon_criterion=None):
    assert not train_flag or optimizer != None
    losses, preds, labels = [], [], []
    if train_flag:
        model.train()
    else:
        model.eval()

    for step, data in enumerate(dataloader):
        if train_flag:
            optimizer.zero_grad()

        # roberta_fea: CLS embedding of last hidden layer in RoBERTa
        emo_roberta, sem_roberta, audio, vision, qmask, umask, label2 = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        if args.feature_type == "multi":
            emo_roberta = torch.cat([emo_roberta,audio, vision], dim=-1)
            sem_roberta = torch.cat([sem_roberta], dim=-1)

        # [修改] 接收兩個返回值：log_prob (分類) 和 proj_feat (對比)
        log_prob, proj_feat = model(emo_roberta, sem_roberta, qmask, umask, seq_lengths)

        label = torch.cat([label2[j][:seq_lengths[j]] for j in range(len(label2))])

        # --- [核心修改] Loss 計算部分 ---
        
        # 1. 計算分類 Loss (Cross Entropy)
        loss_ce = loss_f(log_prob, label)

        # 2. 計算對比學習 Loss (只在訓練時計算)
        if train_flag and supcon_criterion is not None:
            # proj_feat 已經是 [Total_Samples, 128]，符合 SupCon 要求
            loss_sup = supcon_criterion(proj_feat, labels=label)
            
            # 結合 Loss：建議權重 0.1 (你可以根據實驗結果調整 0.05 ~ 0.5)
            loss = loss_ce + 0.2 * loss_sup
        else:
            loss = loss_ce
        # -----------------------------

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train_flag:
            loss.backward()
            optimizer.step()
            # [修正點 1] 在這裡進行 Scheduler Step (每個 Batch 更新)
            if scheduler is not None:
                scheduler.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), float('nan'), [], []

    labels = np.array(labels)
    preds = np.array(preds)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    all_matrix = []
    all_matrix.append(metrics.classification_report(labels, preds, target_names=target_names, digits=4))
    all_matrix.append(metrics.confusion_matrix(labels, preds))

    cr = classification_report(labels, preds, target_names=target_names, digits=4)

    return avg_loss, avg_accuracy, avg_fscore, all_matrix, [labels, preds], cr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--feature_type', type=str, default='multi', help='feature type multi/text/acouf')

    parser.add_argument('--data_dir', type=str, default='../data/iemocap/iemocap_features_roberta.pkl', help='dataset dir: iemocap_features_roberta.pkl')

    parser.add_argument('--output_dir', type=str, default='.', help='saved model dir')

    parser.add_argument('--base_layer', type=int, default=1, help='the number of base model layers,1/2')

    parser.add_argument('--epochs', type=int, default=200, metavar='E', help='number of epochs')

    parser.add_argument('--patience', type=int, default=50, help='early stop')

    parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help='batch size')

    parser.add_argument('--valid_rate', type=float, default=0.1, metavar='valid_rate', help='valid rate: 0.1')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.0002, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--class_weight', action='store_true', default=True, help='use class weights')

    parser.add_argument('--Dataset', type=str, default="IEMOCAP", help='DATASET')

    args = parser.parse_args()

    epochs, batch_size, output_path, data_path, base_layer, feature_type = \
        args.epochs, args.batch_size, args.output_dir, args.data_dir, args.base_layer, args.feature_type
    cuda_flag = torch.cuda.is_available() and not args.no_cuda

    # IEMOCAP dataset
    n_classes, n_speakers, hidden_size, input_size = 6, 2, 256, None
    target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
    # perform 1/weight according to the weight of each label in training data
    class_weights = torch.FloatTensor([1.5 / 0.087178797, 1 / 0.145836136, 1.7 / 0.229786089, 1.5 / 0.148392305, 1.2 / 0.140051123, 1.5 / 0.24875555])
    # 標準化：讓權重的平均值為 1 (這樣 Loss 數值會回到 0.8~1.5 的正常範圍，不影響訓練效果，但更易讀)
    class_weights = class_weights / class_weights.mean()

    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600,
                'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    if feature_type == 'text':
        input_size = D_text
    elif feature_type == "multi":
        input_size = D_text + D_audio + D_visual
    else:
        print('Error: feature_type not set.')
        exit(0)
    
    seed_everything()
    model = ECERC(args,d_t=D_text, d_a=D_audio, d_v=D_visual,
                        base_layer=base_layer,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        n_speakers=n_speakers,
                        n_classes=n_classes,
                        cuda_flag=cuda_flag)
    
    # [新增] 初始化 SupCon Loss Criterion
    supcon_criterion = SupConLoss(temperature=0.1)

    if cuda_flag:
        print('Running on GPU')
        class_weights = class_weights.cuda()
        model.cuda()
    else:
        print('Running on CPU')

    name = 'ECERC'
    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('Running on the {} features........'.format(feature_type))

    train_loader, valid_loader, test_loader = get_IEMOCAP_bert_loaders(path=data_path, batch_size=batch_size, num_workers=0, valid_rate=args.valid_rate)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    # --- 新增第一步：加入餘弦退火調度器 ---
    # T_max 通常設定為總 Epoch 數，代表學習率會在這個週期內從最高降到最低
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 前 5 個 epoch 慢慢增加學習率
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, 
                                                    steps_per_epoch=len(train_loader), 
                                                    epochs=args.epochs)

    # loss_f = Loss(alpha=class_weights if args.class_weight else None)
    # 在 train.py 找到定義 loss_f 的地方
    # label_smoothing 建議設定在 0.1
    loss_f = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    all_test_fscore, all_test_acc = [], []
    best_epoch, best_epoch2, patience, best_eval_fscore, best_eval_loss = -1, -1, 0, 0, None
    patience2 = 0
    # [修正點 1] 用來儲存最佳 Epoch 的詳細結果 (Confusion Matrix / Classification Report)
    best_test_matrix = []
    for e in range(epochs):
        start_time = time.time()

        # [修改] 傳入 scheduler 和 supcon_criterion
        train_loss, train_acc, train_fscore, _, _ ,_= train_or_eval_model(
            model=model, loss_f=loss_f, dataloader=train_loader, train_flag=True,
            optimizer=optimizer, cuda_flag=cuda_flag, feature_type=feature_type,
            target_names=target_names, scheduler=scheduler, supcon_criterion=supcon_criterion)

        valid_loss, valid_acc, valid_fscore, _, _ ,_= train_or_eval_model(model=model, loss_f=loss_f, dataloader=valid_loader, cuda_flag=cuda_flag,
                                                                        feature_type=feature_type, target_names=target_names)
        test_loss, test_acc, test_fscore, test_metrics, _, test_cr = train_or_eval_model(model=model, loss_f=loss_f, dataloader=test_loader,
                                                                                cuda_flag=cuda_flag, feature_type=feature_type, target_names=target_names)
        all_test_fscore.append(test_fscore)
        all_test_acc.append(test_acc)

        if args.valid_rate > 0:
            eval_loss, _, eval_fscore = valid_loss, valid_acc, valid_fscore
        else:
            eval_loss, _, eval_fscore = test_loss, test_acc, test_fscore
        if e == 0 or best_eval_fscore < eval_fscore:
            patience = 0
            best_epoch, best_eval_fscore = e, eval_fscore
            # [修正點 2] 更新最佳結果矩陣
            best_test_matrix = test_metrics
        else:
            patience += 1

        if best_eval_loss is None:
            best_eval_loss = eval_loss
            best_epoch2 = 0
        else:
            if eval_loss < best_eval_loss:
                best_epoch2, best_eval_loss = e, eval_loss
                patience2 = 0
            else:
                patience2 += 1

        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore,
                       round(time.time() - start_time, 2)))

        # 這樣下一輪 (Epoch e+1) 就會使用新的學習率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {e} finished. Current LR: {current_lr:.6f}')

        # --- 新增：當找到更好的模型時，儲存起來 ---
        if e == best_epoch:
            save_path = f"{output_path}/ECERC_MODEL_best.pkl"
            torch.save(model.state_dict(), save_path)
            print(f">>> 最佳模型已儲存至: {save_path}")

        if patience >= args.patience and patience2 >= args.patience:
            break

    print('Early stoping...', patience)
    print('Eval-metric: F1, Epoch: {}, best_eval_fscore: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch, best_eval_fscore,
                                                                                                all_test_acc[best_epoch] if best_epoch >= 0 else 0,
                                                                                                all_test_fscore[best_epoch] if best_epoch >= 0 else 0))
    # [修正點 3] 印出儲存好的最佳矩陣
    print("Test Result at Best Epoch:")
    if best_test_matrix and len(best_test_matrix) > 0:
        print(best_test_matrix[0]) # 印出 Classification Report
    else:
        print("No result matrix saved.")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # 假設這是你剛剛跑出來的 confusion matrix (從 best_test_matrix[1] 拿到的)
    # 你可以手動填入，或是如果變數還在記憶體中直接用
    cm = best_test_matrix[1] 
    target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']

    # 正規化 (看百分比比較直觀)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Normalized)')
    plt.show()


