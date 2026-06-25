import os
import torch
import argparse
from torchvision.transforms import transforms

from models.wintok import WinTok
from utils.config import Args

from data.dataset import normalize_01_into_pm1, pil_load


def main(args):

    # load model
    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['trainer']['unitok']

    unitok_cfg = Args()
    unitok_cfg.load_state_dict(ckpt['args'])
    model = WinTok(unitok_cfg)
    model.load_state_dict(sd)
    model.to('cuda')
    model.eval()

    args.dtype = torch.bfloat16
    args.device = "cuda"
    preprocess_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ])
    
    # load images
    vis_imgs = []
    for img in os.listdir(args.dir_raw):
        img = os.path.join(args.dir_raw, img)
        img = pil_load(img, args.img_size)
        vis_imgs.append(preprocess_val(img))
    vis_imgs = torch.stack(vis_imgs, dim=0).to(args.device, non_blocking=True)

    # reconstruct images
    os.makedirs(args.dir_recon, exist_ok=True)
    with torch.inference_mode():
        recs = model.img_to_reconstructed_img(vis_imgs).clamp(-1, 1)

    # save both the original and reconstructed images
    for img, path in zip(vis_imgs, os.listdir(args.dir_raw)):
        filename = os.path.basename(path)
        img_denorm = img.add(1).mul_(0.5).clamp_(0, 1)
        img_pil = transforms.ToPILImage()(img_denorm.cpu())
        img_pil.save(os.path.join(args.dir_recon, os.path.splitext(filename)[0] + ".png"))
    for rec, path in zip(recs, os.listdir(args.dir_raw)):
        filename = os.path.basename(path)
        rec_denorm = rec.add(1).mul_(0.5).clamp_(0, 1)
        rec_pil = transforms.ToPILImage()(rec_denorm.cpu())
        rec_pil.save(os.path.join(args.dir_recon, os.path.splitext(filename)[0] + "_recon.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='path_to_ckpt')
    parser.add_argument('--dir_raw', type=str, default='./assets/vis_imgs')
    parser.add_argument('--dir_recon', type=str, default='./rec_imgs')
    parser.add_argument('--img_size', type=int, default=256)

    args = parser.parse_args()
    main(args)

