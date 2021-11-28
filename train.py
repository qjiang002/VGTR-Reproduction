import argparse
import datetime
import json
import random
import time
import math

import numpy as np
from pathlib import Path
from numpy.core.fromnumeric import choose

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, validate


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=1e-4, type=float)
    parser.add_argument('--lr_visu_cnn', default=1e-4, type=float)
    parser.add_argument('--lr_visu_tra', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--lr_power', default=0.9,
                        type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False,
                        action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='vgtr_decay', type=str)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--alphaL1', default=5, type=int)
    parser.add_argument('--alphaLiou', default=2, type=int)

    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")

    # Model parameters
    parser.add_argument('--model_name', type=str, default='vgtr', choices=[
                        'transVG', 'vgtr'], help="Name of model to be exploited.")

    # DETR parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine',
                        'learned'), help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=2, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', default=True, action='store_true')

    parser.add_argument('--imsize', default=512, type=int, help='image size')
    parser.add_argument('--emb_size', default=256, type=int,
                        help='fusion module embedding dimensions')

    # Transformers in two branches
    parser.add_argument('--bert_enc_num', default=12, type=int)
    parser.add_argument('--detr_enc_num', default=6, type=int)

    # Language encoder choice
    parser.add_argument('--model_enc_lang', default='vgtr', choices=[
                        'bert', 'bilstm', 'vgtr'], help='Choose the model to be used as language encoder')
    parser.add_argument('--embedding_path', default='../../datasets/glove_embeddings/glove.6B.300d.txt',
                        help='Specify the path to the word embedding file to be loaded')
    parser.add_argument('--embedding_dim', default=256, type=int,
                        help='Dimension of the word embeddings loaded.')
    parser.add_argument('--bilstm_layers', default=4, type=int,
                        help='Number of Bi-LSTM layers.')
    parser.add_argument('--bilstm_out_dim', default=4, type=int,
                        help='Dimension of output features of the Bi-LSTM encoder.')
    parser.add_argument('--bilstm_dropout', default=0.3, type=float,
                        help='Dropout to use for the Bi-LSTM model.')
    parser.add_argument('--bilstm_hidden_dim', default=512, type=int,
                        help='Size of hidden state of LSTM')
    parser.add_argument('--lr_bilstm', default=1e-4, type=float)

    # Visual encoder choice
    parser.add_argument('--model_enc_vis', default='vgtr', choices=[
                        'detr', 'resnet', 'vgtr'], help='Choose the model to be used as visual encoder')
    parser.add_argument('--no_pre_trained_resnet',
                        action="store_true", help='Use pre-trained resnet?')
    parser.add_argument('--resnet_size', default='50', type=str, choices=[
                        '50', '101'], help='The size of the resnet to be used as visual encoder')
    parser.add_argument('--lr_resnet', default=1e-4, type=float)
    parser.add_argument('--vgtr_visual_out_dim', default=256, type=int, help='Number of VGTR visual output features.')

    # Dynamic convolution choices
    parser.add_argument('--use_dconv', action="store_true",
                        help='Replace all convolution layers with dynamic convolution?')
    parser.add_argument('--dconv_weight_criteria', default='direct_proj', choices=[
                        'direct_proj', 'proj_sim'], help='How do we calculate the weights for each candidate kernel?')
    parser.add_argument('--dconv_candidate_count', default='5', type=int, help='Number of candidate kernels to use')
    parser.add_argument('--dconv_places', default='first_in_block', type=str, choices=['first_in_block', 'last_in_block', 'both'], help='Which conv layers do we replace with dconv?')
    parser.add_argument('--dconv_7x7', action="store_true", help="Do we replace the first 7x7 kernel with dconv?")



    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='../../datasets/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='../../datasets/data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='unc', type=str,
                        help='referit/unc/unc+/gref/gref_umd')
    parser.add_argument('--max_query_len', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--dataset_fraction', default=1, type=float,
                        help='What is the fraction of dataset you want to train the model on? e.g. 0.33 means training only on ~1/3 of the training set.')


    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs/VGTR2',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        '--detr_model', default=None, type=str, help='detr model')
    parser.add_argument(
        '--bert_model', default=None, type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False,
                        action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = build_model(args)

    # build dataset
    dataset_train = build_dataset('train', args)
    dataset_val = build_dataset('val', args)

    # note certain dataset does not have 'test' set:
    # 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    # dataset_test  = build_dataset('test', args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # Assign embeddings to the model and send model to GPU
    if args.model_enc_lang == 'bilstm' or args.model_enc_lang == 'vgtr':
        corpus = dataset_train.corpus
        model.textmodel.assign_embedding(
            corpus.embedding, corpus.num_embeddings, corpus.embedding_dim)
    model.to(device)
    print(model)

    # Distributed training
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    visu_cnn_param = [p for n, p in model_without_ddp.named_parameters() if (
        ("visumodel" in n) and ("backbone" in n) and p.requires_grad)]
    visu_tra_param = [p for n, p in model_without_ddp.named_parameters() if (
        ("visumodel" in n) and ("backbone" not in n) and p.requires_grad)]
    text_tra_param = [p for n, p in model_without_ddp.named_parameters() if (
        ("textmodel" in n) and p.requires_grad)]
    rest_param = [p for n, p in model_without_ddp.named_parameters() if (
        ("visumodel" not in n) and ("textmodel" not in n) and p.requires_grad)]

    param_list = [{"params": rest_param},
                  {"params": visu_cnn_param, "lr": args.lr_visu_cnn},
                  {"params": visu_tra_param, "lr": args.lr_visu_tra},
                  {"params": text_tra_param, "lr": args.lr_bert},
                  ]
    visu_param = [p for n, p in model_without_ddp.named_parameters(
    ) if "visumodel" in n and p.requires_grad]
    text_param = [p for n, p in model_without_ddp.named_parameters(
    ) if "textmodel" in n and p.requires_grad]
    rest_param = [p for n, p in model_without_ddp.named_parameters() if (
        ("visumodel" not in n) and ("textmodel" not in n) and p.requires_grad)]

    # using RMSProp or AdamW
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            param_list, lr=args.lr, weight_decay=args.weight_decay)
        if args.model_name=='vgtr':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError('Lr scheduler type not supportted ')

    # using polynomial lr scheduler or half decay every 10 epochs or step
    if args.lr_scheduler == 'poly':
        def lr_func(epoch): return (1 - epoch / args.epochs) ** args.lr_power
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'halfdecay':
        def lr_func(epoch): return 0.5 ** (epoch // (args.epochs // 10))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'cosine':
        def lr_func(epoch): return 0.5 * \
            (1. + math.cos(math.pi * epoch / args.epochs))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'vgtr_decay':
        def lr_func(epoch): 
            if epoch == 60 or epoch == 90:
                return 0.1
            else:
                return 1
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    else:
        raise ValueError('Lr scheduler type not supportted ')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    elif args.detr_model is not None:
        checkpoint = torch.load(args.detr_model, map_location='cpu')

        if args.use_dconv:
            state_dict = {}
            loaded_dict = checkpoint['model']
            for k, v in model_without_ddp.visumodel.state_dict().items(): # for each param in the new model, search for loaded dict for matching keys
                split = k.split('.')
                if split[-1] == 'candidate_kernel_weight':
                    split[-1] = 'weight'
                    key_to_load = '.'.join(split)
                    if key_to_load not in loaded_dict:
                        continue
                    value_to_load = loaded_dict[key_to_load]
                    state_dict[k] = value_to_load.unsqueeze(0).repeat(args.dconv_candidate_count, 1, 1, 1, 1)

            for k, v in loaded_dict.items(): # for each loaded param, see if can fit in the new model
                if k in state_dict:
                    continue
                state_dict[k] = v
            missing_keys, unexpected_keys = model_without_ddp.visumodel.load_state_dict(state_dict, strict=False)
        
        else:
            missing_keys, unexpected_keys = model_without_ddp.visumodel.load_state_dict(
                checkpoint['model'], strict=False)
        print('Missing keys when loading detr model:')
        print(missing_keys)
        print('Unexpected keys when loading detr model:')
        print(unexpected_keys)


    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(str(args) + "\n")

    print("Start training")
    start_time = time.time()
    best_accu = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            args, model, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()

        val_stats = validate(args, model, data_loader_val, device)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'validation_{k}': v.item() for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(str(log_stats) + "\n")

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 10 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(
                    output_dir / f'checkpoint{epoch:04}.pth')
            if val_stats['accu'] > best_accu:
                checkpoint_paths.append(output_dir / 'best_checkpoint.pth')
                best_accu = val_stats['accu']

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_accu': val_stats['accu']
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'TransVG training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
