import torch
import torch.nn as nn
import os
import json
from torchstat import stat
from torchinfo import summary
import copy
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.loss_msn import init_msn_loss
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import math
from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()

    exit()
    clf.fit(train_features, train_labels)

    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def run_net(args, config, train_writer=None, val_writer=None):

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
                                                               builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader) = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get(
        'extra_train') else (None, None)

    # build model
    base_model = builder.model_builder(config.model)

    if args.use_gpu:
        base_model.to(args.local_rank)

    n_batches = len(train_dataloader)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # test flops
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config, n_batches)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # # -- init losses
    msn = init_msn_loss(
        num_views=config.data.focal_views + config.data.rand_views,
        tau=config.criterion.temperature,
        me_max=config.criterion.me_max,
        return_preds=True)

    def one_hot(targets, num_classes, smoothing=config.data.label_smoothing):
        off_value = smoothing / num_classes
        on_value = 1. - smoothing + off_value
        targets = targets.long().view(-1, 1).to(device)
        return torch.full((len(targets), num_classes), off_value, device=device).scatter_(1, targets, on_value)

    # -- make prototypes
    prototypes, proto_labels = None, None
    num_proto = config.criterion.num_proto
    output_dim = config.model.transformer_config.output_dim_fc
    if num_proto > 0:
        with torch.no_grad():
            prototypes = torch.empty(num_proto, output_dim)
            _sqrt_k = (1. / output_dim) ** 0.5
            torch.nn.init.uniform_(prototypes, -_sqrt_k, _sqrt_k)
            prototypes = torch.nn.parameter.Parameter(prototypes).to(device)

            # -- init prototype labels
            proto_labels = one_hot(torch.tensor([i for i in range(num_proto)]), num_proto)

        prototypes.requires_grad = True
        logger.info(f'Created prototypes: {prototypes.shape}')
        logger.info(f'Requires grad: {prototypes.requires_grad}')

    # trainval
    # training
    base_model.zero_grad()

    # -- sharpening schedule
    _increment_T = (config.criterion.final_sharpen - config.criterion.start_sharpen) / (
                n_batches * config.max_epoch * 1.25)
    sharpen_scheduler = (config.criterion.start_sharpen + (_increment_T * i) for i in
                         range(int(n_batches * config.max_epoch * 1.25) + 1))

    time_mean = []
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(range(5))

        num_iter = 0

        base_model.train()  # set model to training mode

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)

            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)
            elif dataset_name.startswith('ScanNet'):
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints


            anchor_views, target_views, all_re_loss = base_model(points)
            target_views = target_views.detach()

            T = next(sharpen_scheduler)

            # Step 3. compute msn loss with me-max regularization
            (ploss, me_max, ent, logs, _) = msn(
                T=T,
                use_sinkhorn=config.criterion.use_sinkhorn,
                use_entropy=config.criterion.use_ent,
                anchor_views=anchor_views,
                target_views=target_views,
                proto_labels=proto_labels,
                prototypes=prototypes)

            loss_cls = ploss \
                       + config.criterion.memax_weight * me_max \
                       + config.criterion.ent_weight * ent

            all_re_loss = all_re_loss * 1000
            loss_1 = loss_cls + all_re_loss

            vis_patch_loss = all_re_loss.clone().detach()
            rec_patch_loss = all_re_loss.clone().detach()

            loss_1.backward()


            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss_1 = dist_utils.reduce_tensor(loss_1, args)
                losses.update(
                    [loss_1.item(), loss_cls.item(), all_re_loss.item(), vis_patch_loss.item(), rec_patch_loss.item()])
            else:
                losses.update(
                    [loss_1.item(), loss_cls.item(), all_re_loss.item(), vis_patch_loss.item(), rec_patch_loss.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss_1', loss_1.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/loss_cls', loss_cls.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/all_re_loss', all_re_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/vis_patch_loss', vis_patch_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/rec_patch_loss', rec_patch_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if (idx % 20 == 0) and idx > 1:
                a = losses.val()
                print_log(
                    '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %.4f loss_cls = %.4f all_re_loss = %.4f  vis_patch_loss = %.4f rec_patch_loss = %.4f lr = %.6f' %
                    (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                     losses.val(0), losses.val(1), losses.val(2), losses.val(3), losses.val(4),
                     optimizer.param_groups[0]['lr']), logger=logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/loss_cls', losses.avg(1), epoch)
            train_writer.add_scalar('Loss/Epoch/all_re_loss', losses.avg(2), epoch)
            train_writer.add_scalar('Loss/Epoch/vis_patch_loss', losses.avg(3), epoch)
            train_writer.add_scalar('Loss/Epoch/rec_patch_loss', losses.avg(4), epoch)

        time_mean.append(epoch_end_time - epoch_start_time)

        print_log(
            '[Training] EPOCH: %d EpochTime = %.3f (s) meanTime = %.3f (s)  Losses = %.4f loss_cls = %.4f all_re_loss = %.4f vis_patch_loss = %.4f rec_patch_loss = %.4f' %
            (epoch, epoch_end_time - epoch_start_time, np.mean(time_mean), losses.avg(0), losses.avg(1), losses.avg(2),
             losses.avg(3), losses.avg(4)), logger=logger)


        if False and epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config,
                               logger=logger)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())

        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        # if args.distributed:
        #     train_features = dist_utils.gather_tensor(train_features, args)
        #     train_label = dist_utils.gather_tensor(train_label, args)
        #     test_features = dist_utils.gather_tensor(test_features, args)
        #     test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(),
                               test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass