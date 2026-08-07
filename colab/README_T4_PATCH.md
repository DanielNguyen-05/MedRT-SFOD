# MedRT-SFOD T4 patch

Merge this folder into the root of the existing `MedRT-SFOD` project stored on Google Drive.

It adds/restores:
- `colab/00_MedRT_SFOD_T4_End_to_End.ipynb`
- `scripts/YOLO26/train_source_supervised.py`
- `scripts/YOLO26/stage0_stage1_adabn_rc_yolo26_v2.py`
- `scripts/YOLO26/stage2_rtsfod_yolo26.py`
- current `rasp_pruning.py`
- current `stage2_rasp_rtsfod_yolo26.py`

The notebook runs GPU stages from Drive-backed project data while copying code/dataset to `/content/MedRT-SFOD` for faster T4 access. Outputs/checkpoints are written to `<Drive project>/runs_t4`.

Before running, set Colab Runtime to T4 GPU. In the notebook config cell, change only `DRIVE_PROJECT` if the folder is not `/content/drive/MyDrive/MedRT-SFOD`.
