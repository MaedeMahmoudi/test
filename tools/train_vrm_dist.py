import sys

sys.path.append('..')

import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict, tinyimagenet200_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.models as tm

from timm.models import create_model

def init_distributed_mode(args): # from SDD
    '''initilize DDP
    '''
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}, local rank:{args.gpu}, world size:{args.world_size}", flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )

def main(cfg, resume, opts, distribution_arsg):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)

    init_distributed_mode(distribution_arsg)
    use_cuda = torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset_strong(cfg)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        elif cfg.DATASET.TYPE == "tinyimagenet200":
            model_student = tinyimagenet200_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        elif cfg.DATASET.TYPE.startswith("tinyimagenet200"):
            if cfg.DISTILLER.TEACHER == 'deit':
                model_teacher = create_model('deit_base_distilled_patch16_384', pretrained=False, drop_path_rate=0.1)
                for param in model_teacher.parameters():
                    param.requires_grad = False
                model_teacher.reset_classifier(num_classes=200)
                model_teacher.load_state_dict(torch.load('../download_ckpts/tinyimagenet_teachers/deit_base_distilled_384.pth')['model_state_dict'])
                model_student = tinyimagenet200_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
            elif cfg.DISTILLER.TEACHER == 'vit':
                model_teacher = create_model('vit_large_patch16_384', pretrained=False, drop_path_rate=0.1)
                for param in model_teacher.parameters():
                    param.requires_grad = False
                model_teacher.reset_classifier(num_classes=200)
                vit_model_prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..//download_ckpts/tinyimagenet_teachers/")
                vit_model_path = vit_model_prefix + 'vit_large_384'
                model_teacher.load_state_dict(torch.load(vit_model_path)['model_state_dict'])
                model_student = tinyimagenet200_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
            elif cfg.DISTILLER.TEACHER == 'swin':
                model_teacher = create_model('swin_large_patch4_window7_384', pretrained=False, drop_path_rate=0.1)
                for param in model_teacher.parameters():
                    param.requires_grad = False
                model_teacher.reset_classifier(num_classes=200)
                swin_model_prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..//download_ckpts/tinyimagenet_teachers/")
                swin_model_path = swin_model_prefix + 'swin_large_384'
                model_teacher.load_state_dict(torch.load(swin_model_path)['model_state_dict'])
                model_student = tinyimagenet200_model_dict[cfg.DISTILLER.STUDENT][0](
                    num_classes=num_classes
                )
            else:
                net, pretrain_model_path = tinyimagenet200_model_dict[cfg.DISTILLER.TEACHER]
                assert (pretrain_model_path is not None), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
                model_teacher = net(num_classes=num_classes)
                model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
                model_student = tinyimagenet200_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        if cfg.DISTILLER.TYPE == "CRD": # not executed
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else: # executed
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
    
    # distiller = torch.nn.DataParallel(distiller.cuda())
    distiller = torch.nn.SyncBatchNorm.convert_sync_batchnorm(distiller)
    distiller = distiller.to(device)

    distiller = DDP(distiller, device_ids=[distribution_arsg.gpu], find_unused_parameters=True)

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--local_rank', type=int, help='local rank, will passed by ddp')
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.local_rank = args.local_rank
    cfg.distributation = True
    cfg.freeze()
    main(cfg, args.resume, args.opts, args)