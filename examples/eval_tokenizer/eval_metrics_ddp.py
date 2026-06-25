import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from scipy import linalg
from tqdm import tqdm
from PIL import Image

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import lpips
from examples.eval_tokenizer.inception import InceptionV3

from transformers import AutoTokenizer, SiglipTextModel

from models.wintok import WinTok
from utils.config import Args
from data.dataset import build_clip_transforms
from utils.zero_shot_classifier import build_zero_shot_classifier
from utils.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES


class FlatImageFolderDataset(torch.utils.data.Dataset):
    """Read images from a directory (recursively) and return label=0 for all."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        paths = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in self.IMG_EXTS:
                    paths.append(os.path.join(dirpath, fn))
        self.paths = sorted(paths)

        if len(self.paths) == 0:
            raise ValueError(f"No images found under: {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
    

def init_distributed():
    """
    单机多卡 NCCL 初始化：
    - 必须在 init_process_group 之前 set_device
    - local_rank 从环境变量读取
    """
    if dist.is_initialized():
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank


def is_main_process():
    return dist.get_rank() == 0


def all_gather_np(array: np.ndarray):
    """把任意 shape 的 numpy array gather 到 rank0，返回 rank0拼接后的 ndarray。"""
    local_size = torch.tensor([array.shape[0]], device="cuda")
    size_list  = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)
    sizes = [int(v.item()) for v in size_list]

    # flatten 传输，最后再 reshape
    flat = torch.from_numpy(array).cuda()
    # 先 gather 尺寸最大的 tensor，其他 pad
    max_size = max(sizes) * array.shape[1]
    padding  = max_size - flat.numel()
    flat_padded = torch.cat([flat.flatten(), torch.zeros(padding, device="cuda")], dim=0)

    gather_list = [torch.empty_like(flat_padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, flat_padded)

    if is_main_process():
        chunks = []
        for g, sz in zip(gather_list, sizes):
            chunks.append(g[: sz * array.shape[1]].view(sz, array.shape[1]).cpu().numpy())
        return np.concatenate(chunks, axis=0)
    return None


def load_model(args):
    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_cfg = Args()
    model_cfg.load_state_dict(ckpt['args'])
    model = WinTok(model_cfg)
    
    sd = ckpt['trainer']['unitok']
    model.load_state_dict(sd)
    return model.eval()


class VQWrapper(torch.nn.Module):
    """把 encode→decode 封装成 forward，适配 DDP。"""
    def __init__(self, model):
        super().__init__()
        self.g = model

    def forward(self, x):

        with torch.no_grad():
            
            encoder_out = self.g.encode(x)
            quant = encoder_out['quant']

            decoder_out = self.g.decode(quant)

            outdict = {**encoder_out, **decoder_out}
        return outdict


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def count_vocab_from_indices(indices: torch.Tensor, vocab_size: int, num_codebooks: int) -> torch.Tensor:
    """Count vocab usage for VectorQuantizerM indices.

    indices: (B, num_codebooks, L) with per-codebook ids in [0, vocab_size/num_codebooks).
    Returns counts: (vocab_size,) int64.
    """
    if indices.dtype != torch.long:
        indices = indices.long()
    if indices.dim() != 3 or indices.shape[1] != num_codebooks:
        raise ValueError(f"Expected indices shape (B, {num_codebooks}, L), got {tuple(indices.shape)}")
    if vocab_size % num_codebooks != 0:
        raise ValueError(f"vocab_size ({vocab_size}) must be divisible by num_codebooks ({num_codebooks})")

    sub_vocab = vocab_size // num_codebooks
    offsets = (torch.arange(num_codebooks, device=indices.device, dtype=torch.long) * sub_vocab).view(1, -1, 1)
    global_ids = (indices + offsets).reshape(-1)
    return torch.bincount(global_ids, minlength=vocab_size)


def print_vocab_stats(counts: torch.Tensor, topk: int = 50, title: str = "Vocab usage"):
    """Pretty print vocab usage statistics on rank0."""
    counts = counts.detach().cpu().long()
    total = int(counts.sum().item())
    used = int((counts > 0).sum().item())
    vocab_size = int(counts.numel())
    coverage = 100.0 * used / max(vocab_size, 1)

    print(f"\n=========  {title}  =========")
    print(f"Total tokens counted     : {total}")
    print(f"Vocab size               : {vocab_size}")
    print(f"Used tokens              : {used} ({coverage:.2f}%)")

    if total > 0:
        p = (counts.float() / total)
        nz = p > 0
        entropy = float((-p[nz] * torch.log(p[nz])).sum().item())
        ppl = float(torch.exp(torch.tensor(entropy)).item())
        print(f"Entropy (nats)           : {entropy:.6f}")
        print(f"Perplexity               : {ppl:.2f}")

    k = min(int(topk), vocab_size)
    if k > 0 and total > 0:
        vals, idx = torch.topk(counts, k=k)
        print(f"\nTop-{k} tokens:")
        for i, (tid, c) in enumerate(zip(idx.tolist(), vals.tolist()), start=1):
            pct = 100.0 * c / total
            print(f"{i:>3d}. id={tid:<6d} count={c:<10d} ({pct:6.3f}%)")


def main(args):
    
    local_rank = init_distributed()              # 先初始化分布式
    device = torch.device(f"cuda:{local_rank}")

    # 1. dataset
    _, val_transform = build_clip_transforms(args)
    use_mscoco = args.mscoco is not None
    if use_mscoco:
        if not os.path.exists(args.mscoco):
            raise FileNotFoundError(f"--mscoco path not found: {args.mscoco}")
        dataset = FlatImageFolderDataset(args.mscoco, transform=val_transform)
    else:
        if args.imagenet_val is None:
            raise ValueError("You must provide either --mscoco <dir> or --imagenet_val <dir>.")
        if not os.path.exists(args.imagenet_val):
            raise FileNotFoundError(f"--imagenet_val path not found: {args.imagenet_val}")
        dataset = datasets.ImageFolder(args.imagenet_val, transform=val_transform)

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.local_bs,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )

    # 2. model & metrics
    base_model = load_model(args)
    wrapper = VQWrapper(base_model).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        wrapper, device_ids=[device], output_device=device, broadcast_buffers=False
    )

    # Optional vocab usage counting
    vocab_counts = None
    if getattr(args, "vocab_stats", False):
        vocab_size = int(wrapper.g.codebook_size)
        num_codebooks = int(wrapper.g.quantizer.num_codebooks)
        vocab_counts = torch.zeros(vocab_size, dtype=torch.long, device=device)

    # If only collecting vocab stats, skip the expensive metrics/models.
    # If MSCOCO path is provided (and exists), skip classification accuracy metrics.
    do_cls_metrics = (not use_mscoco)

    if not getattr(args, "vocab_only", False):
        text_encoder = SiglipTextModel.from_pretrained(args.model).to(device).eval()
        text_encoder = torch.nn.parallel.DistributedDataParallel(
            text_encoder, device_ids=[device], output_device=device, broadcast_buffers=False
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        # inception / lpips 也做 DDP，只是 forward 不梯度
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        inception = InceptionV3([block_idx]).to(device).eval()

        lpips_alex = lpips.LPIPS(net='alex').to(device).eval()
        lpips_vgg  = lpips.LPIPS(net='vgg').to(device).eval()

        classifier = None
        if do_cls_metrics:
            if is_main_process():
                print(f"Building zero-shot classifier with {len(IMAGENET_CLASSNAMES)} classes...")

            classifier = build_zero_shot_classifier(
                    text_encoder.module,  # 用 .module 访问原模型
                    tokenizer=tokenizer,
                    classnames=IMAGENET_CLASSNAMES,
                    templates=OPENAI_IMAGENET_TEMPLATES,
                    num_classes_per_batch=10,
                    device=device,
                )
    
    # 3. 统计量
    img_cnt = torch.tensor(0, dtype=torch.long, device=device)

    if not getattr(args, "vocab_only", False):
        ## for recon
        ssim_value = torch.tensor(0.0, device=device)
        psnr_value = torch.tensor(0.0, device=device)
        lpips_alex_sum = torch.tensor(0.0, device=device)
        lpips_vgg_sum = torch.tensor(0.0, device=device)
        iter_cnt = torch.tensor(0, dtype=torch.long, device=device)

        feats_ref, feats_rec = [], []

        ## for classification
        if do_cls_metrics:
            top1 = torch.tensor(0, dtype=torch.long, device=device)
            top5 = torch.tensor(0, dtype=torch.long, device=device)
            total_samples = torch.tensor(0, dtype=torch.long, device=device)

    # 4. 推理
    with torch.no_grad():
        for images, labels in tqdm(dataloader, disable=not is_main_process()):
            images = images.to(device)
            targets = labels.long().to(device)
            batch_size = images.size(0)
            img_cnt += batch_size

            if getattr(args, "vocab_stats", False):
                # NOTE: img_to_idx will run an extra encoder forward; use --vocab_only to avoid extra metrics cost.
                indices = model.module.g.img_to_idx(images)
                vocab_counts += count_vocab_from_indices(indices, vocab_counts.numel(), model.module.g.quantizer.num_codebooks)

            if getattr(args, "vocab_only", False):
                continue

            out_dict = model(images)
            rec = out_dict['recon'].clamp(-1, 1)

            # LPIPS
            lpips_alex_sum += lpips_alex(images, rec).sum()
            lpips_vgg_sum  += lpips_vgg (images, rec).sum()

            # FID features
            feats_ref.append(inception((images + 1) / 2)[0].squeeze(-1).squeeze(-1).cpu().numpy())
            feats_rec.append(inception((rec    + 1) / 2)[0].squeeze(-1).squeeze(-1).cpu().numpy())

            # PSNR / SSIM
            img_np = ((images + 1) / 2).mul_(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rec_np = ((rec    + 1) / 2).mul_(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            ssim_sum = torch.tensor(0.0, device=device)
            psnr_sum = torch.tensor(0.0, device=device)
            for a, b in zip(img_np, rec_np):
                ssim_sum  += ssim_loss(a, b, data_range=255, channel_axis=-1)
                psnr_sum  += psnr_loss(a, b)
            ssim_value += ssim_sum / batch_size
            psnr_value += psnr_sum / batch_size
            iter_cnt += 1

            # Accuracy (skip when using MSCOCO image folder)
            if do_cls_metrics:
                image_features = out_dict['pooler_out']
                image_features = F.normalize(image_features, dim=-1)
                logits = 100.0 * image_features @ classifier.T
                pred = logits.topk(5, dim=-1).indices
                top1 += (pred[:, 0] == targets).sum()
                top5 += (pred == targets.unsqueeze(1)).any(dim=1).sum()
                total_samples += batch_size

    # 5. gather
    if getattr(args, "vocab_stats", False):
        dist.reduce(vocab_counts, dst=0, op=dist.ReduceOp.SUM)

    if not getattr(args, "vocab_only", False):
        lpips_t = torch.stack([lpips_alex_sum, lpips_vgg_sum, ssim_value, psnr_value, img_cnt.float(), iter_cnt.float()])
        dist.reduce(lpips_t, dst=0, op=dist.ReduceOp.SUM)

        feats_ref = all_gather_np(np.concatenate(feats_ref, axis=0))
        feats_rec = all_gather_np(np.concatenate(feats_rec, axis=0))

        if do_cls_metrics:
            # Classification metrics: reduce across all GPUs
            cls_metrics = torch.stack([top1.float(), top5.float(), total_samples.float()])
            dist.reduce(cls_metrics, dst=0, op=dist.ReduceOp.SUM)

    if is_main_process():
        if getattr(args, "vocab_stats", False):
            print_vocab_stats(vocab_counts, topk=getattr(args, "vocab_topk", 50), title="Vocab usage (global)")

        if not getattr(args, "vocab_only", False):
            lpips_alex_sum, lpips_vgg_sum, ssim_value, psnr_value, img_cnt, iter_cnt = lpips_t.tolist()
            lpips_alex_val = lpips_alex_sum / img_cnt
            lpips_vgg_val  = lpips_vgg_sum  / img_cnt
            ssim_val       = ssim_value       / iter_cnt
            psnr_val       = psnr_value       / iter_cnt

            mu1, mu2 = feats_ref.mean(0), feats_rec.mean(0)
            sigma1   = np.cov(feats_ref, rowvar=False)
            sigma2   = np.cov(feats_rec, rowvar=False)
            fid_val  = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

            if do_cls_metrics:
                global_top1, global_top5, global_total = cls_metrics.tolist()

                final_acc1 = (global_top1 / global_total) * 100
                final_acc5 = (global_top5 / global_total) * 100

            print("\n=========  Final Results  =========")
            total_evaluated = int(global_total) if do_cls_metrics else int(img_cnt)
            print(f"Total samples evaluated  : {total_evaluated}")
            print(f"FID            : {fid_val:.4f}")
            print(f"LPIPS (alex)   : {lpips_alex_val:.4f}")
            print(f"LPIPS (vgg)    : {lpips_vgg_val :.4f}")
            print(f"SSIM           : {ssim_val      :.4f}")
            print(f"PSNR           : {psnr_val      :.4f}")
            if do_cls_metrics:
                print(f"Top-1 Accuracy          : {final_acc1:.2f}%")
                print(f"Top-5 Accuracy          : {final_acc5:.2f}%")
    dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='')
    # model
    parser.add_argument('--model', type=str, default='')
    # data
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--dataset_type', type=str, default='webdataset')
    parser.add_argument('--imagenet_train', type=str, default=None)
    parser.add_argument('--imagenet_val', type=str, default=None)
    parser.add_argument('--imagenet_v2', type=str, default=None)
    parser.add_argument('--mscoco', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--resize_ratio', type=float, default=1.125)
    parser.add_argument('--hflip', type=bool, default=False)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--use_aug', type=bool, default=False)
    # optimization
    parser.add_argument('--local_bs', type=int, default=64)
    # vocab usage stats
    parser.add_argument('--vocab_stats', action='store_true', help='Count per-token vocab usage (DDP-reduced)')
    parser.add_argument('--vocab_topk', type=int, default=50, help='Print top-k most frequent tokens')
    parser.add_argument('--vocab_only', action='store_true', help='Only run vocab counting, skip other metrics')

    args = parser.parse_args()
    main(args)

