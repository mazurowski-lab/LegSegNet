## Import libraries
import os
import cv2
import torch

from pathlib import Path
import gradio as gr
import nibabel as nib
import numpy as np

from inference import (
    LegCTSegmenter,
    CLASS_DICT,
    CLASS_COLOR_DICT,
    IMAGE_SIZE,
    hu_to_uint8,
    color_overlay,
    load_nifti_volume,
    make_side_view,
    compute_body_composition,
)

CORONAL_DISPLAY_MAX_H = 200

## Input your results saving folder here
OUTPUT_DIR = Path("   ")

def get_segmenter():
    return LegCTSegmenter()

def predict_png(img_slice):
    ## The input CT image slice should be:
    ## 1. Clip to [-200, 200]
    ## 2. Normalize to [0, 255]
    img_slice = np.asarray(img_slice)
    if img_slice.ndim == 3:
        img_slice = cv2.cvtColor(img_slice, cv2.COLOR_RGB2GRAY)
    if img_slice.shape != (IMAGE_SIZE, IMAGE_SIZE):
        img_slice = cv2.resize(img_slice, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    
    mask = get_segmenter().predict_slice(img_slice)

    overlay = color_overlay(img_slice, mask, alpha=0.55)
    summary = "  •  ".join(f"**{CLASS_DICT[c]}**: {int((mask == c).sum())} px" for c in (1, 2, 3, 4))
    mask_path = OUTPUT_DIR / "leg_ct_seg_mask.png"
    
    cv2.imwrite(str(mask_path), mask)
    return overlay, str(mask_path), summary


def draw_marker_lines(img_slice, y_pos):
    out = img_slice.copy()
    line_color = (41, 128, 185)

    for y in y_pos:
        cv2.line(out, (0, y), (out.shape[1] - 1, y), line_color, thickness=2)
        
        ## Add circle to the left and right ends of the line
        cv2.circle(out, (4, y), 4, line_color, -1)
        cv2.circle(out, (out.shape[1] - 5, y), 4, line_color, -1)

    return out


def disp_to_z(y, state):
    z = int(round(y / state["z_to_y"]))
    if state["flipped"]:
        z = state["Z"] - 1 - z
    return max(0, min(state["Z"] - 1, z))


def upload_nifti(file, state):
    state = {}
    vol, codes, spacing_mm = load_nifti_volume(file.name)
    side, flipped = make_side_view(vol, projection="max")
    Z, W = side.shape

    ## This is to show the coronal preview of the volume data
    if Z > CORONAL_DISPLAY_MAX_H:
        scale = CORONAL_DISPLAY_MAX_H / Z
        new_w = max(1, int(W * scale))
        disp = cv2.resize(side, (new_w, CORONAL_DISPLAY_MAX_H), interpolation=cv2.INTER_AREA)
        state["z_to_y"] = CORONAL_DISPLAY_MAX_H / Z
    else:
        disp = side
        state["z_to_y"] = 1.0
    
    disp_rgb = np.stack([disp] * 3, axis=-1)
    state.update(
        vol_hu=vol, 
        side_disp_rgb=disp_rgb, 
        Z=Z, 
        flipped=flipped,
        codes=codes, 
        spacing_mm=spacing_mm
    )
    info = (f"**Shape:** {vol.shape}  \n"
            f"**Voxel:** {spacing_mm[0]:.2f}×{spacing_mm[1]:.2f}×{spacing_mm[2]:.2f} mm")
    
    ## Return to match the Gradio pipeline
    return disp_rgb, [], info, "—", "—", state


## Clicks to draw line to select the range to predict
def click_side(clicks, state, event: gr.SelectData):
    if not state or state.get("vol_hu") is None:
        return gr.update(), clicks, "—", "—"

    y_click = int(event.index[1])
    clicks = list(clicks) + [y_click]
    if len(clicks) > 2:
        clicks = clicks[-2:]
    
    out = draw_marker_lines(state["side_disp_rgb"], clicks)
    zs = sorted(disp_to_z(y, state) for y in clicks)
    z_lo_text = str(zs[0]) if len(zs) >= 1 else "—"
    z_hi_text = str(zs[-1]) if len(zs) >= 2 else "—"

    return out, clicks, z_lo_text, z_hi_text


def reset_clicks(state):
    if not state or state.get("side_disp_rgb") is None:
        return None, [], "—", "—"
    
    return state["side_disp_rgb"].copy(), [], "—", "—"


def run_volume(clicks, state, progress=gr.Progress(track_tqdm=False)):
    if not state or state.get("vol_hu") is None:
        return None, None, "Upload a NIfTI volume first."
    if len(clicks) != 2:
        return None, None, "Click two points on the side view to set Z range."
    
    zs = sorted(disp_to_z(y, state) for y in clicks)
    z_lo, z_hi = zs[0], zs[1]

    ## Load prediction network
    seg = get_segmenter()
    progress(0.0, desc=f"Running on z={z_lo}..{z_hi}")
    imgs, masks, hus = seg.predict_volume(state["vol_hu"], z_lo, z_hi, progress_cb=progress)
    
    ## Save prediction as NIFTI file
    out_nifti = OUTPUT_DIR / "leg_ct_seg_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.moveaxis(masks, 0, -1).astype(np.uint8), np.eye(4)), str(out_nifti))

    n_sample = min(15, len(imgs))
    idxs = np.linspace(0, len(imgs) - 1, n_sample).astype(int)
    tiles = [color_overlay(imgs[i], masks[i], alpha=0.55) for i in idxs]
    tile_h, tile_w = tiles[0].shape[:2]
    
    ## Show 15 evenly split slices
    cols = 5
    rows = 3
    grid_line_width = 2
    grid_color = (60, 60, 60)
    grid_h = rows * tile_h + (rows + 1) * grid_line_width
    grid_w = cols * tile_w + (cols + 1) * grid_line_width
    grid = np.full((grid_h, grid_w, 3), grid_color, dtype=np.uint8)

    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        y0 = grid_line_width + r * (tile_h + grid_line_width)
        x0 = grid_line_width + c * (tile_w + grid_line_width)
        grid[y0:y0+tile_h, x0:x0+tile_w] = t

    # Body composition
    z_mm, h_mm, w_mm = state["spacing_mm"]
    in_plane_h = state["vol_hu"].shape[1]
    in_plane_w = state["vol_hu"].shape[2]
    sx_256 = h_mm * (in_plane_h / IMAGE_SIZE)
    sy_256 = w_mm * (in_plane_w / IMAGE_SIZE)
    bc = compute_body_composition(masks, hus, sx_256, sy_256, z_mm)

    md  = [f"**Range** z = {z_lo} – {z_hi}  ({z_hi - z_lo + 1} slices)  "
            f"&nbsp;•&nbsp; voxel ≈ {sx_256:.2f} × {sy_256:.2f} × {z_mm:.2f} mm"]
    md += [""]
    md += ["| Tissue | Volume (cm\u00B3) | Attenuation \u03BC (HU) |",
            "|---|---:|---:|",
            f"| SAT  | {bc['V_SAT']:.1f}  | {bc['mu_SAT']:.1f} |",
            f"| SM   | {bc['V_SM']:.1f}   | {bc['mu_SM']:.1f} |",
            f"| IAT  | {bc['V_IAT']:.1f}  | {bc['mu_IAT']:.1f} |",
            f"| Bone | {bc['V_Bone']:.1f} | {bc['mu_Bone']:.1f} |"]
    md += [""]
    md += ["| Derived metric | Value |",
            "|---|---:|",
            f"| V_Fat = V_SAT + V_IAT          | {bc['V_Fat']:.1f} cm\u00B3 |",
            f"| R_IAT/SAT = V_IAT / V_SAT      | {bc['R_IAT_SAT']:.3f} |",
            f"| R_SM/Fat  = V_SM / V_Fat       | {bc['R_SM_Fat']:.3f} |"]
    info = "\n".join(md)
    return grid, str(out_nifti), info

## Color legend
LEGEND_HTML = """
<div style="display:flex; gap:10px;">
    <b>Classes:</b>
    <span style="background:rgb(139,35,35); color:white; padding:4px 8px; border-radius:4px;">SAT</span>
    <span style="background:rgb(70,105,150); color:white; padding:4px 8px; border-radius:4px;">SM</span>
    <span style="background:rgb(210,120,30); color:white; padding:4px 8px; border-radius:4px;">IAT</span>
    <span style="background:rgb(240,210,120); color:black; padding:4px 8px; border-radius:4px;">Bone</span>
</div>
"""

CUSTOM_CSS = """
.gradio-container {
    font-family: Arial, sans-serif;
}
"""

def device_banner_html():
    """Show GPU or CPU mode banner"""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"""
        <div style="padding:8px; background:#e8f5e9; color:#2e7d32; border-radius:8px;">
            <b>GPU mode:</b> {name}
        </div>
        """
    else:
        return """
        <div style="padding:8px; background:#fff8e1; color:#f57c00; border-radius:8px;">
            <b>CPU mode:</b> no GPU detected
        </div>
        """


def build_ui():
    with gr.Blocks(title="LegSegNet") as demo:
        gr.Markdown(
            "# LegSegNet\nSegment **SAT**, **SM**, **IAT**, and **Bone** from CT slices or volumes using a pre-trained nnU-Net 2D model."
        )

        gr.HTML(device_banner_html())
        gr.HTML(LEGEND_HTML)

        with gr.Tab("Single Slice (PNG)"):
            gr.Markdown(
                "Upload a CT slice PNG (HU-windowed to **[-200, 200]**, rescaled to **[0, 255] uint8**). "
                "Any resolution is accepted and will be resized to 256×256.")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    png_input = gr.Image(
                        label="Input CT Slice", 
                        image_mode="L",
                        type="pil", 
                        height=300
                    )
                    png_run = gr.Button(
                        "Run Segmentation", 
                        variant="primary", 
                        size="lg"
                    )
                with gr.Column(scale=1):
                    png_overlay = gr.Image(label="Segmentation Overlay", height=300)
                    png_mask_file = gr.File(label="Download Mask (PNG)")
                    png_summary = gr.Markdown()

            ## Add function to click button
            png_run.click(
                predict_png, 
                inputs=png_input,
                outputs=[png_overlay, png_mask_file, png_summary]
            )

        with gr.Tab("3D Volume (NIfTI)"):
            state = gr.State({})
            clicks = gr.State([])
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    nifti_input = gr.File(label="Upload NIfTI", file_types=[".nii", ".gz"])
                    volume_info = gr.Markdown(value="", elem_id="vol-info")
                    side_view = gr.Image(
                        label="Coronal (click 2 pts)",
                        type="numpy", 
                        height=220, 
                        interactive=False
                    )
                    with gr.Row():
                        z_lo_text = gr.Textbox(label="Z Lo", value="—", interactive=False)
                        z_hi_text = gr.Textbox(label="Z Hi", value="—", interactive=False)
                    with gr.Row():
                        reset_btn = gr.Button("Reset", size="sm")
                        run_btn = gr.Button("Run", variant="primary", size="sm")
                        stop_btn = gr.Button("Stop", variant="stop", size="sm")

                with gr.Column(scale=3):
                    nifti_overlay = gr.Image(label="Segmentation", height=350, show_label=False)

                with gr.Column(scale=2):
                    nifti_summary = gr.Markdown(value="*Run inference to see metrics*")
                    nifti_mask_file = gr.File(label="Download Mask")

            nifti_input.change(
                upload_nifti, inputs=[nifti_input, state],
                outputs=[
                    side_view, 
                    clicks, 
                    volume_info,
                    z_lo_text, 
                    z_hi_text, 
                    state
                ])
            
            side_view.select(
                click_side, 
                inputs=[clicks, state],
                outputs=[side_view, clicks, z_lo_text, z_hi_text])
            reset_btn.click(
                reset_clicks, 
                inputs=state,
                outputs=[side_view, clicks, z_lo_text, z_hi_text])
            
            ## Link the run_volume function to button
            run_event = run_btn.click(
                run_volume, 
                inputs=[clicks, state],
                outputs=[nifti_overlay, nifti_mask_file, nifti_summary],
                show_progress="minimal")
            
            ## Stop prediction if need to reset
            stop_btn.click(fn=None, cancels=[run_event])
            
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    build_ui().launch(
        server_name="127.0.0.1", 
        server_port=port, 
        share=False,
        css=CUSTOM_CSS, 
        theme=gr.themes.Soft()
    )
