import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import math
import cv2
import numpy as np
from pointnet2_ops import pointnet2_utils
import sys
import numpy
from numpy import *
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def compute_loss(loss_1, loss_2, config, niter, train_writer):
    '''
    compute the final loss for optimization
    For dVAE: loss_1 : reconstruction loss, loss_2 : kld loss
    '''
    start = config.kldweight.start
    target = config.kldweight.target
    ntime = config.kldweight.ntime

    _niter = niter - 10000
    if _niter > ntime:
        kld_weight = target
    elif _niter < 0:
        kld_weight = 0.
    else:
        kld_weight = target + (start - target) *  (1. + math.cos(math.pi * float(_niter) / ntime)) / 2.

    if train_writer is not None:
        train_writer.add_scalar('Loss/Batch/KLD_Weight', kld_weight, niter)

    loss = loss_1 + kld_weight * loss_2

    return loss

def get_temp(config, niter):
    if config.get('temp') is not None:
        start = config.temp.start
        target = config.temp.target
        ntime = config.temp.ntime
        if niter > ntime:
            return target
        else:
            temp = target + (start - target) *  (1. + math.cos(math.pi * float(niter) / ntime)) / 2.
            return temp
    else:
        return 0 

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss1', 'Loss2'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            temp = get_temp(config, n_itr)


            ret = base_model(points, temperature = temp, hard = False)

            loss_1, loss_2 = base_model.module.get_loss(ret, points)

            _loss = compute_loss(loss_1, loss_2, config, n_itr, train_writer)

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss_1 = dist_utils.reduce_tensor(loss_1, args)
                loss_2 = dist_utils.reduce_tensor(loss_2, args)
                losses.update([loss_1.item() * 1000, loss_2.item() * 1000])
            else:
                losses.update([loss_1.item() * 1000, loss_2.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss_1', loss_1.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_2', loss_2.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Temperature', temp, n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if config.scheduler.type != 'function':
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_2', losses.avg(1), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 5:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)   
    if train_writer is not None:  
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            ret = base_model(inp = points, hard=True, eval=True)
            coarse_points = ret[0]
            dense_points = ret[1]

            sparse_loss_l1 =  ChamferDisL1(coarse_points, points)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, points)
            dense_loss_l1 =  ChamferDisL1(dense_points, points)
            dense_loss_l2 =  ChamferDisL2(dense_points, points)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            _metrics = Metrics.get(dense_points, points)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            vis_list = [0, 1000, 1600, 1800, 2400, 3400]
            if val_writer is not None and idx in vis_list: #% 200 == 0:
                input_pc = points.squeeze().detach().cpu().numpy()
                input_pc = misc.get_ptcloud_img(input_pc)
                val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

                sparse = coarse_points.squeeze().cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse)
                val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')

                dense = dense_points.squeeze().cpu().numpy()
                dense_img = misc.get_ptcloud_img(dense)
                val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
       
        
            if (idx+1) % 2000 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    config.model.cls_blocks = int(args.cls_blocks)
    base_model = builder.model_builder(config.model)
    # builder.load_model(base_model, args.ckpts, logger = logger)  # finetnue model
    # base_model.load_model_from_ckpt(args.ckpts)  #  pretrain model
    base_model = base_model.cuda()

    # tsne
    # pre_train_tsne(base_model, args, config)

    # test_vis_mask pic
    vis_shapeNet(base_model, test_dataloader, args, config, logger=logger)
    # vis_scans(base_model, test_dataloader, args, config, logger=None)

def show_recon_pc(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)


    base_model = builder.model_builder(config.model)

    ckpt = torch.load(args.ckpts)
    base_model.load_state_dict(ckpt, strict=False)

    # base_model.load_model_from_ckpt(args.ckpts)  #  pretrain model
    base_model = base_model.cuda()

    print("start show_recon_pc....")

    # test_vis_mask pic
    vis_shapeNet(base_model, test_dataloader, args, config, logger=logger)
    # vis_scans(base_model, test_dataloader, args, config, logger=None)


def vis_scans(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()  # set model to eval mode


    for class_object in range(15):
        imgnum = 0
        class_object=0
        with torch.no_grad():
            for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
                # import pdb; pdb.set_trace()
                # if taxonomy_ids[0] not in useful_cate:
                #     continue

                dataset_name = config.dataset.test._base_.NAME

                # if class_object==data[1].item():
                #     if imgnum>10:
                #         continue
                # else:
                #     class_object = data[1].item()
                #     imgnum =0
                # if class_object==14:
                #     return
                if   data[1].item() != 9:
                    continue
                else:
                    imgnum = imgnum + 1

                    if imgnum > 30:
                        return



                points = data[0].cuda()

                npoints =1024
                if npoints == 1024:
                    point_all = 1200
                elif npoints == 2048:
                    point_all = 2400
                elif npoints == 4096:
                    point_all = 4800
                elif npoints == 8192:
                    point_all = 8192


                fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
                fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()



                _, masked_pc, re_Pc = base_model(points, noaug=False, vis_flag=1)


                data_path = f'./vis_scan/'+str(class_object)+"_"+str(imgnum)
                if not os.path.exists(data_path):
                    os.makedirs(data_path)

                points = points.squeeze().detach().cpu().numpy()
                #
                np.savetxt(os.path.join(data_path, 'gt.txt'), points, delimiter=';')
                points_img = misc.get_ptcloud_img(points, points)
                points_img = points_img[145:655, 145:675, :]
                # orign_img.append(points_img[150:650,150:675,:])

                masked_pc_image = []
                for i in range(3):
                    dense_points = masked_pc[i]
                    dense_points = dense_points.squeeze().detach().cpu().numpy()
                    np.savetxt(os.path.join(data_path, 'dense_points.txt'), dense_points, delimiter=';')

                    # dense_points = dense_points + points * 0

                    dense_points_img = misc.get_ptcloud_img(dense_points, points)
                    masked_pc_image.append(dense_points_img[145:655, 145:675, :])
                img_masked_pc = np.concatenate(masked_pc_image, axis=0)

                re_Pc_image = []
                for i in range(3):
                    dense_points = re_Pc[i]
                    dense_points = dense_points.squeeze().detach().cpu().numpy()
                    np.savetxt(os.path.join(data_path, 'dense_points.txt'), dense_points, delimiter=';')
                    dense_points_img = misc.get_ptcloud_img(dense_points, points)
                    re_Pc_image.append(dense_points_img[145:655, 145:675, :])
                img_re_Pc = np.concatenate(re_Pc_image, axis=0)

                list1 = []
                list1.append(img_masked_pc)
                list1.append(img_re_Pc)

                img = np.concatenate(list1, axis=1)
                img_path = os.path.join(data_path, f'all.jpg')
                cv2.imwrite(img_path, img)

                img_path = os.path.join(data_path, f'orign.jpg')
                cv2.imwrite(img_path, points_img)



def vis_shapeNet(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode
    target = './vis'
    useful_cate = [
        "02691156",
        "02818832",
        "04379243",
        "04099429",
        "03948459",
        "03790512",
        "03642806",
        "03467517",
        "03261776",
        "03001627",
        "02958343",
        "03759954"
    ]
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # import pdb; pdb.set_trace()
            if  taxonomy_ids[0] not in useful_cate:
                continue
    
            dataset_name = config.dataset.test._base_.NAME

            if  taxonomy_ids[0] not in useful_cate:
                continue

            if taxonomy_ids[0] == "02691156":
                view_roll, view_pitch= 90, 135
            elif taxonomy_ids[0] == "04379243":
                view_roll, view_pitch = 30, 30
            elif taxonomy_ids[0] == "03642806":
                view_roll, view_pitch = 30, -45
            elif taxonomy_ids[0] == "03467517":
                view_roll, view_pitch = 0, 90
            elif taxonomy_ids[0] == "03261776":
                view_roll, view_pitch = 0, 75
            elif taxonomy_ids[0] == "03001627":
                view_roll, view_pitch = 30, -45
            else:
                view_roll, view_pitch = 0, 0



            if dataset_name == 'ShapeNet':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')


            _, masked_pc, re_Pc = base_model(points, noaug = False, vis_flag=1)

            final_image = []


            data_path = args.vis_path+f'/{taxonomy_ids[0]}_{idx}'
            # data_path = f'./vis_ShapeNet_9pic_block/{taxonomy_ids[0]}_{idx}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)



            points = points.squeeze().detach().cpu().numpy()
            #
            np.savetxt(os.path.join(data_path,'gt.txt'), points, delimiter=';')
            points_img = misc.get_ptcloud_img(points, points,view_roll, view_pitch)
            points_img = points_img[145:655, 145:675, :]


            masked_pc_image = []
            for i in range(len(masked_pc)):
                dense_points = masked_pc[i]
                dense_points = dense_points.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path,'dense_points.txt'), dense_points, delimiter=';')

                # dense_points = dense_points + points * 0

                dense_points_img = misc.get_ptcloud_img(dense_points, points,view_roll, view_pitch)
                masked_pc_image.append(dense_points_img[145:655, 145:675, :])
            img_masked_pc = np.concatenate(masked_pc_image, axis=0)

            re_Pc_image = []
            for i in range(len(masked_pc)):
                dense_points = re_Pc[i]
                dense_points = dense_points.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path, 'dense_points.txt'), dense_points, delimiter=';')
                dense_points_img = misc.get_ptcloud_img(dense_points,points,view_roll, view_pitch)
                re_Pc_image.append(dense_points_img[145:655, 145:675, :])
            img_re_Pc = np.concatenate(re_Pc_image, axis=0)



            list1 =[]
            list1.append(img_masked_pc)
            list1.append(img_re_Pc)

            img = np.concatenate(list1, axis=1)
            img_path = os.path.join(data_path, f'all.jpg')
            cv2.imwrite(img_path, img)

            img_path = os.path.join(data_path, f'orign.jpg')
            cv2.imwrite(img_path, points_img)

            if idx >500:
                break

        return 



def draw_tsne(features, labels, pic_path):
    '''
    features:(N*m) N*m ，N: data_num m:data_dim
    label:(N)
    '''

    tsne = TSNE(n_components=2, init='pca', random_state=0)

    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    latent = features
    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    # print('tsne_features的shape:', tsne_features.shape)
    # plt.scatter(tsne_features[:, 0], tsne_features[:, 1],markers =',')  # 将对降维的特征进行可视化
    # plt.show()

    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:, 0]
    df["comp-2"] = tsne_features[:, 1]

    sns.scatterplot( x= "comp-1", y="comp-2",hue=df.y.tolist(),markers =',',
                     palette=sns.color_palette("hls", class_num),
                    data=df).set(title="Bearing data T-SNE projection unsupervised")


    plt.draw()
    plt.savefig(pic_path+".png")
    plt.savefig(pic_path + ".pdf")
    plt.close()

def conver_label(label, label_list):
    label_dict = {"02691156": 1, "02747177": 2, "02773838": 3, "02801938": 4, "02808440": 5, "02818832": 6, "02828884": 7, "02843684": 8, "02871439": 9,"02876657": 10,  "02880940": 11,"02924116": 12,"02933112": 13, "02942699": 14, "02946921": 15, "02954340": 16,"02958343": 17, "02992529": 18, "03001627": 19, "03046257": 20,  "03085013": 21, "03207941": 22, "03211117":23,"03261776": 24,"03325088": 25,"03337140": 26,"03467517": 27,"03513137": 28,"03593526": 29,"03624134": 30,"03636649": 31,"03642806": 32,"03691459": 33,"03710193": 34,"03759954": 35,"03761084": 36,"03790512": 37,"03797390": 38 , "03928116": 39,"03938244": 40,"03948459":41,"03991062": 42,"04004475": 43,"04074963": 44,"04090263": 45,  "04099429": 46,"04225987": 47, "04256520": 48, "04330267": 49, "04379243": 50, "04401088": 51, "04460130": 52,"04468005": 53,"04530566": 54,"04554684": 55}


    for i in label:
        label_list.append(label_dict[i])

    return  label_list


def pre_train_tsne(base_model, args, config):


    config.dataset.train.others.bs = config.total_bs
    config.dataset.val.others.bs = config.total_bs
    config.dataset.test.others.bs = config.total_bs
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model.eval()
    label_list =[]
    test_features = []
    test_label = []

    dataset_name = config.dataset.test._base_.NAME

    with torch.no_grad():
        if dataset_name == 'ShapeNet':

            npoints = config.dataset.train.others.npoints


            for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
                points = data.cuda()
                label = taxonomy_ids

                points = misc.fps(points, npoints)
                assert points.size(1) == npoints
                feature = base_model(points, noaug=True)

                label_list = conver_label(label, label_list)


                feature = feature.reshape(feature.shape[0],-1)
                test_features.append(feature.detach())
                # test_label.append(target)

            test_features = torch.cat(test_features, dim=0)
            test_features = test_features.data.cpu().numpy()
            test_label = np.array( label_list)

        else:
            for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):


                points = data[0].cuda()
                label = data[1].cuda()

                points = data[0].cuda()

                npoints = 1024
                if npoints == 1024:
                    point_all = 1200
                elif npoints == 2048:
                    point_all = 2400
                elif npoints == 4096:
                    point_all = 4800
                elif npoints == 8192:
                    point_all = 8192

                fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
                fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                features = base_model(points, tsneFlag=True)
                target = label.view(-1)

                test_features.append(features.detach())
                test_label.append(target.detach())

            test_features = torch.cat(test_features, dim=0)
            test_features = test_features.data.cpu().numpy()

            test_label = torch.cat(test_label, dim=0)
            test_label =  test_label.data.cpu().numpy()


    draw_tsne(test_features, test_label, './tsne_pic/tsne_MMN40_scatch_cls_max_mean_last')

