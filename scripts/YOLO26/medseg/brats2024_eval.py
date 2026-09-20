"""O2O-only predictions and original-grid, per-region 3D BraTS evaluation."""
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

from brats2024_data import NAMES, write_json


def regions_from_result(result, shape):
    """Resolve overlapping native instances by confidence; labels remain 1..4."""
    output = np.zeros(shape, dtype=np.uint8)
    if result.masks is None:
        return output
    masks = result.masks.data.detach().cpu().numpy() > .5
    if masks.shape[1:] != tuple(shape):
        raise ValueError(f'Prediction geometry mismatch: {masks.shape[1:]} != {shape}')
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    conf = result.boxes.conf.detach().cpu().numpy()
    if len(masks) != len(classes) or not set(classes).issubset(set(NAMES)):
        raise ValueError('Invalid prediction class/mask alignment')
    # Stable tie-breaking by original detection order.
    order = sorted(range(len(conf)), key=lambda i: (conf[i], -i))
    for i in order:
        output[masks[i]] = classes[i] + 1
    return output


def binary_metrics(pred, gt, spacing):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    both_empty = not (tp + fp + fn)
    out = dict(tp=tp, fp=fp, fn=fn, gt_voxels=tp + fn, pred_voxels=tp + fp,
               dice=2 * tp / (2 * tp + fp + fn) if not both_empty else 1.,
               iou=tp / (tp + fp + fn) if not both_empty else 1.,
               recall=tp / (tp + fn) if tp + fn else None,
               precision=tp / (tp + fp) if tp + fp else None,
               both_empty=both_empty, one_empty=bool(pred.any()) != bool(gt.any()))
    if both_empty:
        out.update(asd_mm=0., hd95_mm=0.)
    elif out['one_empty']:
        # Surface distances are undefined; do not hide misses with a fabricated score.
        out.update(asd_mm=None, hd95_mm=None)
    else:
        structure = generate_binary_structure(3, 1)
        ps = pred & ~binary_erosion(pred, structure=structure, border_value=0)
        gs = gt & ~binary_erosion(gt, structure=structure, border_value=0)
        distances = np.concatenate((distance_transform_edt(~gs, sampling=spacing)[ps],
                                    distance_transform_edt(~ps, sampling=spacing)[gs]))
        out.update(asd_mm=float(distances.mean()), hd95_mm=float(np.percentile(distances, 95)))
    return out


def summarize(rows):
    out = {}
    for name in NAMES.values():
        region = [r['regions'][name] for r in rows]
        patients = defaultdict(list)
        for r in rows:
            patients[r['patient_id']].append(r['regions'][name])
        summary = dict(cases=len(region), patients=len(patients),
                       both_empty_cases=sum(r['both_empty'] for r in region),
                       undefined_surface_cases=sum(r['one_empty'] for r in region),
                       gt_present_cases=sum(r['gt_voxels'] > 0 for r in region))
        for metric in ('dice', 'iou', 'asd_mm', 'hd95_mm', 'recall', 'precision'):
            values = [r[metric] for r in region if r[metric] is not None]
            patient_values = [np.mean([r[metric] for r in visits if r[metric] is not None])
                              for visits in patients.values() if any(r[metric] is not None for r in visits)]
            summary[metric] = dict(case_mean=float(np.mean(values)) if values else None,
                                   case_std=float(np.std(values)) if values else None,
                                   valid_cases=len(values),
                                   patient_mean=float(np.mean(patient_values)) if patient_values else None)
        present = [r['dice'] for r in region if r['gt_voxels'] > 0]
        summary['dice_gt_present_mean'] = float(np.mean(present)) if present else None
        out[name] = summary
    return out


def predict_case(model, root, modality, case, cfg):
    cid, role = case['case_id'], case['role']
    paths = [str(Path(root) / modality / 'images' / role / f'{cid}_z{z:03d}.png') for z in case['slices']]
    shape = (case['shape'][1], case['shape'][0])
    volume = np.zeros((*shape, len(paths)), dtype=np.uint8)
    for start in range(0, len(paths), cfg['batch']):
        batch = paths[start:start + cfg['batch']]
        predictions = model.predict(source=batch, imgsz=cfg['imgsz'], batch=cfg['batch'],
                                    device=cfg['device'], conf=cfg['evaluation']['confidence'],
                                    retina_masks=True, verbose=False, stream=True, max_det=300)
        for offset, (_, result) in enumerate(zip(batch, predictions, strict=True)):
            volume[..., start + offset] = regions_from_result(result, shape)
    return volume


def evaluate(model, root, manifest, modality, role, cfg, output, checkpoint):
    if role not in ('source_val', 'target_test'):
        raise ValueError('Evaluation restricted to source_val or sealed target_test')
    records = [r for r in manifest['cases'] if r['role'] == role]
    rows = []
    for i, case in enumerate(records):
        pred = predict_case(model, root, modality, case, cfg)
        with np.load(Path(root) / 'evaluation_masks' / role / f"{case['case_id']}.npz") as data:
            gt = data['mask']
        if pred.shape != gt.shape:
            raise ValueError('Prediction/GT volume mismatch')
        sx, sy, sz = case['spacing_mm']
        spacing = (sy, sx, sz * manifest['stride'])
        rows.append(dict(case_id=case['case_id'], patient_id=case['patient_id'],
                         regions={name: binary_metrics(pred == cls + 1, gt == cls + 1, spacing)
                                  for cls, name in NAMES.items()}))
        if i % 10 == 0:
            print(f'{modality}/{role}: {i + 1}/{len(records)}', flush=True)
    report = dict(checkpoint=str(checkpoint), modality=modality, role=role,
                  split_sha256=manifest['split_sha256'], smoke=manifest['smoke'],
                  protocol='3D voxel metrics on original grid; per-class confidence-resolved O2O instances',
                  confidence=cfg['evaluation']['confidence'],
                  surface_protocol='6-connected inner surfaces, pooled bidirectional distances in physical mm',
                  empty_policy='both empty: Dice=1, ASD=HD95=0; one empty: Dice=0, ASD/HD95=null; report counts',
                  sampling='subsampled slices; smoke only' if manifest['smoke'] else 'all axial slices',
                  not_official_brats_challenge_metrics=True, summary=summarize(rows), cases=rows)
    write_json(output, report)
    return report


def predict_external(model, root, manifest, modality, cfg, output):
    if manifest['stride'] != 1:
        raise ValueError('NIfTI export requires all slices; use full preparation')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for case in manifest['cases']:
        if case['role'] != 'external_unlabeled':
            continue
        pred = predict_case(model, root, modality, case, cfg).transpose(1, 0, 2)
        nib.save(nib.Nifti1Image(pred, np.asarray(case['affine'])), output / f"{case['case_id']}-seg.nii.gz")
    write_json(output / 'prediction_protocol.json', dict(modality=modality, split_sha256=manifest['split_sha256'],
                                                        ground_truth_available=False, confidence=cfg['evaluation']['confidence']))
