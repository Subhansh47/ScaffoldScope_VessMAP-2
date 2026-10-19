# ScaffoldScope: VessMAP starter project

A local starter app for vessel segmentation and basic morphology review. Includes a compact PyTorch U-Net, training script, Streamlit interface, and connected-component/skeleton measurements.

## Important limits
- This is a **starter project**, not a trained or validated scientific model. No model weights or VessMAP image data are included.
- Read the VessMAP release documentation and confirm its license, citation, image/mask format, and annotation meaning before use.
- Every training image must have a corresponding binary vessel-mask annotation. Never use an image as its own mask.
- The model segments pixels. It does **not** reconstruct missing borders or infer biologically valid vessel connections.
- Measurements are approximate and require validation against expert annotations before scientific use.

## macOS + VS Code setup
1. Extract the ZIP and open the `ScaffoldScope_VessMAP` folder in VS Code.
2. Choose **Terminal → New Terminal**.
3. Run `chmod +x setup_mac.sh train_mac.sh start_app_mac.sh`.
4. Run `./setup_mac.sh`. This creates `.venv` and installs packages. Internet access is required. On Apple Silicon, pip should choose a compatible PyTorch build; if installation fails, share the full error before trying other commands.
5. Prepare data as described below.
6. Train with `./train_mac.sh 20` (20 epochs; CPU). Training can take a while. The best validation-loss checkpoint is saved to `models/scaffoldscope_unet.pt`.
7. Launch with `./start_app_mac.sh` and open the URL printed in Terminal, normally `http://localhost:8765`.

If you already installed requirements in `.venv`, you can skip setup and go to dataset preparation.

## Dataset preparation
Download VessMAP from https://zenodo.org/records/10045265 and inspect its release documentation. This project does not automatically download or convert the dataset because its archive structure and annotation convention must be checked rather than guessed.

Create these folders at the project root:
- `data/vessmap/images/` for original microscopy images
- `data/vessmap/masks/` for corresponding binary masks

Each pair must have the same filename stem, though extensions may differ, e.g. `sample_01.tif` and `sample_01.png`. Masks should have vessel pixels white and background black. Use only true annotations, not images copied as masks. The current loader resizes to 256×256. At least two valid pairs are required to train with a validation split.

## macOS commands
```bash
source .venv/bin/activate
python train.py --epochs 20 --batch-size 2 --size 256 --device cpu
python -m streamlit run app.py --server.port 8765
```

## Windows
Windows users can use `1_setup_cpu.bat`, `2_train.bat`, and `3_start_app.bat`. These `.bat` files are not for macOS.

## App measurements
The app reports foreground area, connected components, skeleton endpoints, and a rough skeleton-pixel length. Pixel-size conversion is only meaningful when the image calibration is known. Skeleton branch points and pixel geometry make this length an approximation, not a calibrated vessel centerline measurement.
