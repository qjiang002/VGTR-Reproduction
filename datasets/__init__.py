# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

import datasets.transforms as T
from .data_loader import TransVGDataset


def make_transforms(args, image_set, is_onestage=False):
    imsize = args.imsize
    
    if is_onestage:
        normalize = Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize)
        ])
        return normalize

    if image_set == 'train':
        if args.model_name=='vgtr':
            return T.Compose([
                T.RandomResize([imsize], with_long_side=True),
                T.VgtrAugmentation(imsize),
                T.ToTensor(),
                T.NormalizeAndPad(size=imsize, mean_padding=True)
            ])

        scales = []
        if args.aug_scale:
            for i in range(7):
                scales.append(imsize - 32 * i)
        else:
            scales = [imsize]

        if args.aug_crop:
            crop_prob = 0.5
        else:
            crop_prob = 0.
    
        return T.Compose([
            T.RandomSelect(
                T.RandomResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600], with_long_side=True),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales),
                ]),
                p=crop_prob
            ),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.GaussianBlur(aug_blur=args.aug_blur),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize, aug_translate=args.aug_translate)
        ])


    if image_set in ['val', 'test', 'testA', 'testB']:
        return T.Compose([
            T.RandomResize([imsize], with_long_side=True),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize, mean_padding=True)
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset(split, args):
    use_lstm = (args.model_enc_lang == 'bilstm' or args.model_enc_lang == 'vgtr')

    return TransVGDataset(data_root=args.data_root,
                        split_root=args.split_root,
                        dataset=args.dataset,
                        split=split,
                        transform=make_transforms(args, split),
                        max_query_len=args.max_query_len,
                        lstm=use_lstm,
                        embedding_dim=args.embedding_dim,
                        dataset_fraction=args.dataset_fraction)
