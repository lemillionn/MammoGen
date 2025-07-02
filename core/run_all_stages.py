import os, glob, json, csv, warnings, argparse, torch
from PIL import Image
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler

from core import config
from core.evaluate import evaluate_all
from core.image_io import save_image_tensor
from modules.train.train_cyclegan import train_cyclegan
from modules.train.train_ddpm import train_ddpm
from modules.translator.generator import Generator
from modules.refiner.ddpm_refine import DDPMRefiner

def find_image(path, exts=('.png','.jpg','.jpeg')):
    if os.path.isfile(path): return path
    base, _ = os.path.splitext(path)
    for e in exts:
        cand = base + e
        if os.path.isfile(cand):
            return cand
    return None

def append_eval(csv_path, stage, fname, res):
    header = ['Stage','Filename','PSNR','SSIM','Dice','SemanticLoss']
    exists = os.path.isfile(csv_path)
    with open(csv_path,'a',newline='') as f:
        w = csv.writer(f)
        if not exists: w.writerow(header)
        w.writerow([
            stage, fname,
            res.get('psnr',''),
            res.get('ssim',''),
            res.get('dice',''),
            res.get('semantic_loss','')
        ])

def run_all_stages(output_dir, mask_path, input_path, real_path, force):
    cfg    = config.load_config()
    device = torch.device(cfg['cyclegan']['device'])
    os.makedirs(output_dir, exist_ok=True)

    # 1) CycleGAN
    ckpt_dir = cfg['paths']['checkpoints']; os.makedirs(ckpt_dir,exist_ok=True)
    ckpts = sorted(glob.glob(f"{ckpt_dir}/G_epoch_*.pth"))
    if force or not ckpts:
        print("[run_all_stages] Training CycleGAN from scratch…")
        train_cyclegan()
        ckpts = sorted(glob.glob(f"{ckpt_dir}/G_epoch_*.pth"))
    best_ckpt = ckpts[-1]
    print(f"[run_all_stages] Using CycleGAN checkpoint {best_ckpt}")

    # 2) DDPM
    ddpm_dir = cfg['paths']['ddpm_model']; os.makedirs(ddpm_dir,exist_ok=True)
    if force or not os.listdir(ddpm_dir):
        print("[run_all_stages] Training DDPM from scratch…")
        train_ddpm()
    print(f"[run_all_stages] Using DDPM model in {ddpm_dir}")

    # 3) Input
    inp = find_image(input_path)
    if not inp: raise RuntimeError(f"No such input: {input_path}")
    print(f"[run_all_stages] Input image: {inp}")

    # 4) Ground truth (DM→CM)
    real = None
    if real_path:
        real = find_image(real_path)
    if not real:
        bm = os.path.basename(inp)
        bm_cm = bm.replace("_DM_","_CM_")
        candidate = os.path.join(cfg['paths']['real_cesm'], bm_cm)
        real = find_image(candidate)
    if real:
        print(f"[run_all_stages] Ground truth: {real}")
    else:
        warnings.warn("No paired ground truth; skipping metrics")

    # 5) Transform
    tf = transforms.Compose([
        transforms.Resize((cfg['cyclegan']['image_size'],)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    img = Image.open(inp).convert('RGB')
    x = tf(img).unsqueeze(0).to(device)

    # clear CSV
    log_csv = os.path.join(output_dir,"full_eval.csv")
    if os.path.exists(log_csv): os.remove(log_csv)

    # 6) CycleGAN → fake_low
    G = Generator().to(device)
    G.load_state_dict(torch.load(best_ckpt,map_location=device))
    G.eval()
    with torch.no_grad():
        fake_low = G(x)
    cyc_path = os.path.join(output_dir,"synthetic_cyclegan.png")
    save_image_tensor(fake_low, cyc_path)
    if real:
        r1 = evaluate_all(real, cyc_path, mask_path, device)
        append_eval(log_csv,"CycleGAN",os.path.basename(inp),r1)
    else:
        print("[run_all_stages] Skipping CycleGAN metrics")

    # 7) DDPM → refined
    fake_low_gray = fake_low[:, 0:1, :, :]  # Convert RGB to single channel

    refiner = DDPMRefiner(
        model=UNet2DModel.from_pretrained(cfg['ddpm']['model_id']).to(device),
        scheduler=DDPMScheduler.from_pretrained(cfg['ddpm']['model_id']),
        device=device
    )
    ref_path = os.path.join(output_dir,"synthetic_refined.png")
    refined = refiner.refine(fake_low_gray)

    # Convert tensor to PIL Image before saving
    refined_img = refined.squeeze()
    if refined_img.ndim == 3:
        refined_img = refined_img.squeeze(0)

    from torchvision.transforms import ToPILImage
    ToPILImage()(refined_img).save(ref_path)

    if real:
        r2 = evaluate_all(real, ref_path, mask_path, device)
        append_eval(log_csv,"DDPM",os.path.basename(inp),r2)
    else:
        print("[run_all_stages] Skipping DDPM metrics")

    # 8) JSON
    summary = {"cyclegan":r1,"ddpm":r2} if real else {}
    with open(os.path.join(output_dir,"evaluation.json"),"w") as f:
        json.dump(summary,f,indent=2)

    print(f"[run_all_stages] Done → outputs in {output_dir}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--real",   default=None)
    p.add_argument("--mask",   default=None)
    p.add_argument("--out-dir",default="data/outputs/final")
    p.add_argument("--force",  action="store_true")
    args = p.parse_args()

    run_all_stages(
        output_dir=args.out_dir,
        mask_path=args.mask,
        input_path=args.input,
        real_path=args.real,
        force=args.force
    )
