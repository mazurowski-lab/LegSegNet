## Import libraries
import json
import cv2
import torch

from pathlib import Path
import nibabel as nib
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

## Global variables
HU_LO = -200.0
HU_HI = 200.0
IMAGE_SIZE = 256

CLASS_DICT = {1: "SAT", 2: "SM", 3: "IAT", 4: "Bone"}
CLASS_COLOR_DICT = {
    1: (139, 35, 35),
    2: (70, 105, 150),
    3: (210, 120, 30),
    4: (240, 210, 120)
}

## Input your checkpoint folder here.
MODEL_FOLDER = "   "

def hu_to_uint8(img_slice):
    x = np.clip(img_slice.astype(np.float32), HU_LO, HU_HI)
    x = (x - HU_LO) / (HU_HI - HU_LO)
    x = x * 255.0
    return x.astype(np.uint8)


def color_overlay(img, mask, alpha=0.55):
    img_3c = np.stack([img] * 3, axis=-1).astype(np.float32)
    img_overlay = img_3c.copy()
    for c, rgb in CLASS_COLOR_DICT.items():
        sel = (mask == c)
        if sel.any():
            img_overlay[sel] = (1 - alpha) * img_3c[sel] + alpha * np.array(rgb)
    return img_overlay.clip(0, 255).astype(np.uint8)


def _patch_plans_check(plans_path):
    plans = json.loads(plans_path.read_text())
    changed = False
    for cfg in plans.get("configurations", {}).values():
        sp = cfg.get("spacing")
        if sp and any(s is None for s in sp):
            cfg["spacing"] = [999.0 if s is None else s for s in sp]
            changed = True
    if changed:
        plans_path.write_text(json.dumps(plans, indent=4))


def load_nifti_volume(nifti_path):
    img = nib.load(str(nifti_path))

    vol = img.get_fdata().astype(np.float32)
    codes = nib.aff2axcodes(img.affine)
    z_axis = next(i for i, c in enumerate(codes) if c in ("S", "I"))
    vol = np.moveaxis(vol, z_axis, 0).astype(np.float32)
    sp = np.linalg.norm(img.affine[:3, :3], axis=0)
    spacing = (
        float(sp[z_axis]),
        float(sp[(z_axis + 1) % 3]),
        float(sp[(z_axis + 2) % 3]),
    )
    return vol, codes, spacing


class LegCTSegmenter:
    def __init__(
            self, 
            model_folder=MODEL_FOLDER,
            fold=0, 
            checkpoint_name="checkpoint_best.pth",
            device= None,
            use_mirroring=None,
            tile_step_size=None
        ):

        self.model_folder = Path(model_folder)
        _patch_plans_check(self.model_folder / "plans.json")

        ## Enable CPU option
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        
        is_cpu = device.type == "cpu"
        if use_mirroring is None:
            use_mirroring = not is_cpu

        ## Depending on usrs' patience
        if tile_step_size is None:
            tile_step_size = 1.0 if is_cpu else 0.5

        ## Initialize the predictor instance
        self.predictor = nnUNetPredictor(
            tile_step_size=tile_step_size, 
            use_gaussian=True,
            use_mirroring=use_mirroring,
            perform_everything_on_device=True, 
            device=device,
            verbose=False, 
            verbose_preprocessing=False, 
            allow_tqdm=False,
        )

        ## Load the checkpoint weight
        self.predictor.initialize_from_trained_model_folder(
            str(self.model_folder),
            use_folds=(fold,),
            checkpoint_name=checkpoint_name,
        )

        ## Load preprocessor
        self.preprocessor = self.predictor.configuration_manager.preprocessor_class(verbose=False)

        self.device = device
        self.is_cpu = is_cpu
        print(f"[LegSegNet] device={device}  use_mirroring={use_mirroring}  tile_step_size={tile_step_size}")

    @torch.inference_mode()
    def predict_slice(self, img_slice):
        if img_slice.shape != (IMAGE_SIZE, IMAGE_SIZE):
            img_slice = cv2.resize(img_slice, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

        ## Preprocessing data
        data = img_slice.astype(np.float32)[None, None]
        seg = np.zeros_like(data, dtype=np.int8)
        props = {"spacing": (999, 1, 1)}
        preprocessed_data, _, props2 = self.preprocessor.run_case_npy(
            data, 
            seg, 
            props,
            self.predictor.plans_manager,
            self.predictor.configuration_manager,
            self.predictor.dataset_json,
        )

        ## Prediction
        logits = self.predictor.predict_logits_from_preprocessed_data(torch.from_numpy(preprocessed_data).float()).cpu()

        ## Postprocessing
        if logits.ndim == 4:
            logits = logits[:, 0]
        
        pred_crop = logits.argmax(0).byte().numpy()
        bbox = props2["bbox_used_for_cropping"]
        if len(bbox) == 3:
            _, (h0, h1), (w0, w1) = bbox
        else:
            (h0, h1), (w0, w1) = bbox
        
        full = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        full[h0:h1, w0:w1] = pred_crop
        return full

    def predict_raw_slice(self, slice):
        img_slice = hu_to_uint8(slice)
        img = cv2.resize(img_slice, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        raw_img  = cv2.resize(slice.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        mask = self.predict_slice(img)
        return img, mask, raw_img

    def predict_volume(self, vol_hu, z_lo=0, z_hi=None, progress_cb=None):
        if z_hi is None:
            z_hi = vol_hu.shape[0] - 1
        z_lo = max(0, int(z_lo))
        z_hi = min(int(z_hi), vol_hu.shape[0] - 1)
        Z_sel = z_hi - z_lo + 1

        imgs  = np.zeros((Z_sel, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        masks = np.zeros((Z_sel, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        hus   = np.zeros((Z_sel, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        
        ## Iteratively predict all slices within the volume
        for i, z in enumerate(range(z_lo, z_hi + 1)):
            img_slice, mask, raw_img = self.predict_raw_slice(vol_hu[z])
            imgs[i]  = img_slice
            masks[i] = mask
            hus[i]   = raw_img
            if progress_cb is not None:
                progress_cb((i + 1) / Z_sel, desc=f"Slice {i+1}/{Z_sel}")
    
        return imgs, masks, hus


def compute_body_composition(masks, ct_hu, sx_mm, sy_mm, dz_mm):
    area_mm2 = sx_mm * sy_mm
    eps = 1e-9
    name = {1: "SAT", 2: "SM", 3: "IAT", 4: "Bone"}

    out, n_per, sum_per = {}, {}, {}
    for c, cn in name.items():
        sel = masks == c
        n   = int(sel.sum())
        s   = float(ct_hu[sel].sum()) if n else 0.0
        out[f"V_{cn}"]  = n * area_mm2 * dz_mm / 1000.0
        out[f"mu_{cn}"] = s / n if n else float("nan")
        n_per[c]   = n
        sum_per[c] = s
    
    out["V_Fat"]     = out["V_SAT"] + out["V_IAT"]
    out["R_IAT_SAT"] = out["V_IAT"] / (out["V_SAT"] + eps)
    out["R_SM_Fat"]  = out["V_SM"]  / (out["V_SAT"] + out["V_IAT"] + eps)
    n_fat            = n_per[1] + n_per[3]
    out["mu_Fat"]    = (sum_per[1] + sum_per[3]) / n_fat if n_fat else float("nan")

    return out


def make_side_view(raw_vol, projection="max"):
    ## Normalize raw image
    vol_min, vol_max = float(raw_vol.min()), float(raw_vol.max())
    if vol_max > vol_min:
        vol_norm = (raw_vol - vol_min) / (vol_max - vol_min)
        vol = (vol_norm * 255.0).astype(np.uint8)
    else:
        vol = np.zeros_like(raw_vol, dtype=np.uint8)

    if projection == "mean":
        side = vol.mean(axis=2).astype(np.uint8)
    else:
        side = vol.max(axis=2)

    # Detect feet
    body = (raw_vol > -100).sum(axis=(1, 2))
    chunk = max(20, len(body) // 10)
    front = body[:chunk].mean()
    back = body[-chunk:].mean()

    flipped = back > front
    if flipped:
        side = side[::-1]
    return side, flipped
