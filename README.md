# LegSegNet

LegSegNet is a Gradio app for lower extremity CT tissue segmentation and body composition analysis. 

It uses a pretrained 2D nnU-Net model to segment four tissue compartments:

- **SAT**: subcutaneous adipose tissue
- **SM**: skeletal muscle
- **IAT**: inter/intramuscular adipose tissue
- **Bone**

LegSegNet is designed for a practical system: upload a CT slice or NIfTI volume, run segmentation, and export masks and quantitative tissue measurements.

![System overview PDF](assets/pipeline.png)

![3D segmentation view PDF](assets/3D_view.png)

## Run

Install the packages and run:

```bash
pip install -r requirements.txt
python app.py
```

The app automatically detects whether CUDA is available. With a GPU, it uses the full inference settings with mirroring and 0.5 tile overlap (Can be customized for patient user). 

On CPU, it disables mirroring and uses `tile_step_size=1.0`, which is faster but may slightly reduce accuracy.

## 2D Slice Inference

Use the **Single Slice (PNG)** tab for one 2D CT slice.

The input PNG should be:

1. Clip HU to `[-200, 200]`
2. Scale to `[0, 255]` as 8-bit grayscale

LegSegNet accepts different image sizes, and resizes the slice to `256 x 256` before inference.

Outputs:

- segmentation overlay
- downloadable mask
- tissue quantification

## 3D Volume Inference

Use the **3D Volume (NIfTI)** tab for `.nii` or `.nii.gz` CT volumes.

The system:

1. Loads the volume and creates a coronal preview of the legs
2. Lets user click two points to choose the axial range
3. Runs slice-by-slice nnUNet inference on the selected range
4. Returns an overlay grid, body composition measurements, and a downloadable 3D mask NIfTI

For volume inputs, LegSegNet reports tissue volume and mean CT attenuation for SAT, SM, IAT, and bone.

## Model Weights

The LegSegNet model is available at: [LegSegNet weights](https://huggingface.co/GogoChen/LegSegNet)

Download following files and put them under `model` folder as follow:

```text
model/
|-- plans.json
|-- dataset.json
|-- fold_0/
    |-- checkpoint_best.pth
```

If your model files are stored somewhere else, update `MODEL_FOLDER` in `inference.py`.

## Citation

If you find LegSegNet useful, please cite the manuscript:

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). See the [LICENSE](LICENSE) file for details.
