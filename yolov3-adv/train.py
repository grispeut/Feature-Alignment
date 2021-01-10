import os
import time
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from models import *
from utils.datasets import *
from utils.utils import *
from attack import PGD

def train():
    cfg = args.cfg
    data = args.data
    weights = args.weights
    epochs = args.epochs
    batch_size = args.bs
    resume = args.resume

    # adv
    adv = args.adv
    imgs_weight = args.iw
    num_steps = args.num_steps
    step_size = args.step_size
    kdfa = args.kdfa
    ssfa = args.ssfa
    beta = args.beta
    gamma = args.gamma
    kdfa_weights = args.kdfa_weights
    tod = args.tod
    kdfa_cfg = cfg

    img_size = 416
    rect = False
    multi_scale = False
    accumulate = 1
    scale_factor = 0.5
    num_workers = min([os.cpu_count(), batch_size, 16])
    path = 'weights/'
    if not os.path.exists(path):
        os.makedirs(path)
    wdir = path + os.sep  # weights dir
    last = wdir + 'last.pt'
    tb_writer = SummaryWriter()

    # Initialize
    init_seeds(seed=3)

    if multi_scale:
        img_sz_min = round(img_size / 32 / 1.5) + 1
        img_sz_max = round(img_size / 32 * 1.5) - 1
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']

    # Initialize model
    model = Darknet(cfg, arc='default').cuda().train()
    hyp = model.hyp
    # Optimizer
    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if 'Conv2d.weight' in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0

    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    del pg0, pg1

    start_epoch = 0
    if weights.endswith('.pt'):  # pytorch format
        # possible weights are 'last.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(weights)
        # load model
        chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(chkpt['model'], strict=True)

        if resume:
            if (chkpt['optimizer'] is not None):
                optimizer.load_state_dict(chkpt['optimizer'])
                start_epoch = chkpt['epoch'] + 1

        del chkpt

    elif weights.endswith('.weights'):  # darknet format
        # possible weights are 'yolov3.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        print('inherit model weights')
        if 'yolo' in weights:
            model.load_darknet_weights(weights)
            print(' inherit model weights from yolo pretrained weights')
        else:
            load_darknet_weights(model, weights)
            print(' do not inherit model weights from yolo pretrained weights')

    if adv:
        model_adv = PGD(model)
        if kdfa:
            model_t = Darknet(kdfa_cfg, arc='default').cuda().eval()
            print('inherit kdfa_weights')
            if 'yolo' in kdfa_weights:
                model_t.load_darknet_weights(kdfa_weights)
                print(' inherit model weights from yolo pretrained weights')
            else:
                load_darknet_weights(model_t, kdfa_weights)
                print(' do not inherit model weights from yolo pretrained weights')
            for param_k in model_t.parameters():
                param_k.requires_grad = False

    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=rect,  # rectangular training
                                  image_weights=False,
                                  cache_labels=True if epochs > 10 else False,
                                  cache_images=False)

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=not rect,  # Shuffle=True unless rectangular training is used
                            pin_memory=True,
                            collate_fn=dataset.collate_fn,
                            drop_last=True)
    nb = len(dataloader)
    t0 = time.time()
    print('Starting %g for %g epochs...' % (start_epoch, epochs))
    for epoch in range(start_epoch, epochs+1):  # epoch ------------------------------------------------------------------
        if epoch == int(epochs*2/3)+1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                print('param_group lr anealing: ', param_group['lr'])

        print(('\n' + '%10s' * 7) % (
            'Epoch', 'gpu_mem', 'clean', 'adv', 'kd', 'ss', 'total'))

        mloss = torch.zeros(5).cuda()  # mean losses,'clean', 'adv', 'kd', 'ss', 'total'
        loss_ss = torch.zeros(1).cuda()
        loss_kd = torch.zeros(1).cuda()
        loss_clean = torch.zeros(1).cuda()
        loss_adv = torch.zeros(1).cuda()
        loss = torch.zeros(1).cuda()

        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar

        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------

            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.cuda()
            targets = targets.cuda()

            # Multi-Scale training
            if multi_scale:
                if ni / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            if adv:
                if tod:
                    imgs_adv, loss_clean = model_adv.adv_sample_train_tod(imgs, targets, step_size=step_size,
                                                                          num_steps=num_steps,
                                                                          all_bp=True, sf=imgs_weight * scale_factor)

                    pred = model(imgs_adv, fa=False)
                    loss_adv, loss_items = compute_loss(pred, targets, model)
                    loss_adv *= (1 - imgs_weight)

                else:
                    imgs_adv, ssfa_out, loss_clean = model_adv.adv_sample_train(imgs, targets, step_size=step_size,
                                                                                  all_bp=True, sf=imgs_weight*scale_factor, num_steps=num_steps)
                    pred, fa_out = model(imgs_adv, fa=True)
                    fa_out_norm = F.normalize(fa_out, dim=1)
                    loss_adv, loss_items = compute_loss(pred, targets, model)
                    loss_adv *= (1 - imgs_weight)

                    if kdfa:
                        kdfa_out = model_t(imgs, fa=True, only_fa=True)
                        kdfa_out_norm = F.normalize(kdfa_out, dim=1)
                        kd_sim = torch.einsum('nc,nc->n', [fa_out_norm, kdfa_out_norm])
                        kd_sim.data.clamp_(-1., 1.)
                        loss_kd = (1. - kd_sim).mean().view(-1) * beta

                    if ssfa:
                        ssfa_out_norm = F.normalize(ssfa_out, dim=1)
                        ss_sim = torch.einsum('nc,nc->n', [fa_out_norm, ssfa_out_norm])
                        ss_sim.data.clamp_(-1., 1.)
                        loss_ss = (1-ss_sim).mean().view(-1) * gamma
            else:
                pred = model(imgs, fa=False)
                loss_adv, loss_items = compute_loss(pred, targets, model)


            loss_kd *= scale_factor
            loss_ss *= scale_factor
            loss_adv *= scale_factor
            loss_items = torch.cat((loss_clean, loss_adv, loss_kd, loss_ss,
                                    (loss_clean + loss_adv + loss_kd + loss_ss))).detach()
            loss = loss_adv + loss_kd + loss_ss
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                break
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            script = ('%10s' * 2 + '%10.3g' * 5) % (
                '%g/%g' % (epoch, epochs), '%.3gG' % mem, *mloss)
            pbar.set_description(script)
            # end batch ------------------------------------------------------------------------------------------------


        # Write Tensorboard results
        x = list(mloss.cpu().numpy())  # + list(results) + list([thre]) + list([prune_ratio]) + list([par_prune_ratio])
        titles = ['Loss_clean', 'Loss_adv', 'Loss_kd', 'Loss_ss', 'Train_loss']
        for xi, title in zip(x, titles):
            tb_writer.add_scalar(title, xi, epoch)


        chkpt = {'epoch': epoch,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict()
                 }
        torch.save(chkpt, last)
        if epoch > 0 and (epoch) % 5 == 0:
            torch.save(chkpt, wdir + 'backup%g.pt' % epoch)
            if epoch == epochs:
                convert(cfg=cfg, weights=wdir + 'backup%g.pt' % epoch)
        del chkpt
        time_consume = '%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600)
        print(time_consume)

        # end epoch ----------------------------------------------------------------------------------------------------

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO-V3 network')
    parser.add_argument('--r', dest='resume', default=False, action="store_true", help='resume checkpoint or not')
    parser.add_argument('--data', type=str, default='data/voc.data', help='training dataset')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-voc.cfg', help='cfg file')
    parser.add_argument('--weights', type=str, default='weights/voc_pretrained.weights')
    parser.add_argument('--bs', type=int, default=64, help='batch_size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')

    parser.add_argument('--adv', default=False, action="store_true")
    parser.add_argument('--iw', type=float, default=0.5, help='imgs_weight')
    parser.add_argument('--step_size', type=float, default=0.01, help='attack step size')
    parser.add_argument('--num_steps', type=int, default=1, help='attack num_steps')
    parser.add_argument('--kdfa', default=False, action="store_true", help='Knowledge-Distilled Feature Alignment')
    parser.add_argument('--ssfa', default=False, action="store_true", help='Self-Supervised Feature Alignment')
    parser.add_argument('--beta', type=float, default=10., help='weight for kdfa')
    parser.add_argument('--gamma', type=float, default=10., help='weight for ssfa')
    parser.add_argument('--kdfa_weights', type=str, default='weights/voc_pretrained.weights')
    parser.add_argument('--tod', default=False, action="store_true", help='using task oriented domain')

    args = parser.parse_args()
    print(args)
    train()
