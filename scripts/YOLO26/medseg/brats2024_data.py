"""Patient-disjoint BraTS2024 GLI preparation for MedRT-SFSeg modality shifts."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import cv2
import nibabel as nib
import numpy as np
import yaml

MODALITIES = ('t1n', 't1c', 't2w', 't2f')
NAMES = {0: 'NETC', 1: 'SNFH', 2: 'ET', 3: 'RC'}
ROLES = ('source_train', 'source_val', 'target_train', 'target_test')
FOLDERS = ('training_data1_v2', 'training_data_additional')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def discover(root):
    records = []
    for folder in (*FOLDERS, 'validation_data'):
        directory = Path(root) / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for case in sorted(directory.iterdir()):
            if not case.is_dir():
                continue
            cid = case.name
            if not cid.startswith('BraTS-GLI-'):
                raise ValueError(f'Unexpected non-GLI directory: {case}')
            suffixes = MODALITIES + (() if folder == 'validation_data' else ('seg',))
            for suffix in suffixes:
                path = case / f'{cid}-{suffix}.nii.gz'
                if not path.is_file():
                    raise FileNotFoundError(path)
            records.append(dict(case_id=cid, patient_id=cid.rsplit('-', 1)[0],
                                directory=str(case.relative_to(root)), folder=folder))
    if len({r['case_id'] for r in records}) != len(records):
        raise ValueError('Duplicate GLI case IDs')
    return records


def split_patients(records, fractions, seed):
    if len(fractions) != 4 or any(f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError('Four positive role fractions must sum to 1')
    labeled = [r for r in records if r['folder'] in FOLDERS]
    patients = sorted({r['patient_id'] for r in labeled})
    random.Random(seed).shuffle(patients)
    counts = [int(len(patients) * f) for f in fractions[:3]]
    counts.append(len(patients) - sum(counts))
    if min(counts) < 1:
        raise ValueError('Too few patients for four disjoint partitions')
    assignment = {}
    start = 0
    for role, count in zip(ROLES, counts):
        assignment.update({p: role for p in patients[start:start + count]})
        start += count
    out = [dict(r, role=assignment[r['patient_id']] if r['folder'] in FOLDERS else 'external_unlabeled')
           for r in records]
    validate_manifest({'cases': out})
    return out


def validate_manifest(manifest):
    records = manifest['cases']
    if len({r['case_id'] for r in records}) != len(records):
        raise ValueError('Duplicate case IDs')
    ownership = {}
    for r in records:
        if r['patient_id'] != r['case_id'].rsplit('-', 1)[0]:
            raise ValueError('Patient ID does not match case prefix')
        if r['role'] not in (*ROLES, 'external_unlabeled'):
            raise ValueError('Unknown data role')
        if ownership.setdefault(r['patient_id'], r['role']) != r['role']:
            raise ValueError(f"Patient leakage: {r['patient_id']}")
    if not set(ROLES).issubset({r['role'] for r in records}):
        raise ValueError('Every experimental role must be nonempty')


def make_plan(root, config, smoke=False):
    records = split_patients(discover(root), config['split']['fractions'], config['seed'])
    full_counts = {role: {'cases': sum(r['role'] == role for r in records),
                         'patients': len({r['patient_id'] for r in records if r['role'] == role})}
                   for role in (*ROLES, 'external_unlabeled')}
    if smoke:
        # Sample by ID only; never inspect target labels to choose images.
        selected = []
        for role in (*ROLES, 'external_unlabeled'):
            candidates = [r for r in records if r['role'] == role]
            random.Random(f"{config['seed']}:{role}").shuffle(candidates)
            selected.extend(candidates[:1])
        records = sorted(selected, key=lambda r: r['case_id'])
    signature = dict(version=1, cases=records, modalities=MODALITIES, names=NAMES,
                     stride=32 if smoke else 1, seed=config['seed'], fractions=config['split']['fractions'],
                     normalization='nonzero volume percentiles 0.5/99.5 -> uint8; zero background',
                     polygons='external contours per raw label; holes approximated, exact evaluation mask retained')
    return dict(signature, split_sha256=digest(signature), smoke=smoke, prepared=False,
                full_counts=full_counts, selected_counts=dict(Counter(r['role'] for r in records)))


def normalize(volume):
    x = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('MRI has NaN/Inf')
    nonzero = x != 0
    out = np.zeros(x.shape, dtype=np.uint8)
    if nonzero.any():
        lo, hi = np.percentile(x[nonzero], [0.5, 99.5])
        if hi > lo:
            out[nonzero] = np.rint(np.clip((x[nonzero] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return out


def polygons(mask):
    """One external polygon per component and class; retain original masks for scoring."""
    rows, omitted = [], 0
    h, w = mask.shape
    for raw_label in range(1, 5):
        contours, _ = cv2.findContours((mask == raw_label).astype(np.uint8), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points = contour.reshape(-1, 2)
            if len(points) < 3 or cv2.contourArea(contour) <= 0:
                omitted += 1
                continue
            xy = points.astype(float) / [w, h]
            rows.append(f'{raw_label - 1} ' + ' '.join(f'{v:.7f}' for v in xy.flatten()))
    return rows, omitted


def prepare(root, dst, plan):
    root, dst = Path(root), Path(dst)
    manifest_file = dst / 'manifest.json'
    if manifest_file.exists():
        old = json.loads(manifest_file.read_text())
        if old['split_sha256'] != plan['split_sha256']:
            raise ValueError('Prepared directory belongs to a different split/configuration')
        if old['prepared']:
            verify_prepared(dst, old)
            print(f'Prepared data already complete: {dst}', flush=True)
            return old
        completed = {r['case_id']: r for r in old['cases'] if r.get('exported')}
    else:
        if dst.exists() and any(dst.iterdir()):
            raise FileExistsError(dst)
        completed = {}
    manifest = dict(plan, cases=[completed.get(r['case_id'], r) for r in plan['cases']])
    write_json(manifest_file, manifest)
    for idx, record in enumerate(manifest['cases']):
        if record.get('exported'):
            continue
        cid, role = record['case_id'], record['role']
        case_dir = root / record['directory']
        reference = nib.load(case_dir / f'{cid}-t1n.nii.gz')
        shape = reference.shape
        if len(shape) != 3:
            raise ValueError(f'Expected 3D MRI: {cid}')
        zs = list(range(0, shape[2], plan['stride']))
        for modality in MODALITIES:
            img = nib.load(case_dir / f'{cid}-{modality}.nii.gz')
            if img.shape != shape or not np.allclose(img.affine, reference.affine, rtol=0, atol=1e-4):
                raise ValueError(f'Geometry mismatch: {cid}/{modality}')
            data = normalize(np.asanyarray(img.dataobj))
            directory = dst / modality / 'images' / role
            directory.mkdir(parents=True, exist_ok=True)
            for z in zs:
                # Store one grayscale modality. Loaders replicate it into 3 identical channels.
                if not cv2.imwrite(str(directory / f'{cid}_z{z:03d}.png'), data[:, :, z].T):
                    raise IOError(f'PNG write failed: {cid}/{modality}/{z}')
        omitted = 0
        if role in ('source_train', 'source_val', 'target_test'):
            img = nib.load(case_dir / f'{cid}-seg.nii.gz')
            if img.shape != shape or not np.allclose(img.affine, reference.affine, rtol=0, atol=1e-4):
                raise ValueError(f'Mask geometry mismatch: {cid}')
            mask = np.asanyarray(img.dataobj)
            if not set(np.unique(mask)).issubset({0, 1, 2, 3, 4}):
                raise ValueError(f'Unexpected mask label: {cid}')
            mask = mask.astype(np.uint8)
            gt_path = dst / 'evaluation_masks' / role / f'{cid}.npz'
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(gt_path, mask=mask[:, :, zs].transpose(1, 0, 2))
            if role != 'target_test':
                for z in zs:
                    rows, small = polygons(mask[:, :, z].T)
                    omitted += small
                    for modality in MODALITIES:
                        directory = dst / modality / 'labels' / role
                        directory.mkdir(parents=True, exist_ok=True)
                        (directory / f'{cid}_z{z:03d}.txt').write_text('\n'.join(rows) + ('\n' if rows else ''))
        # No target_train segmentation file is ever opened or exported.
        manifest['cases'][idx] = dict(record, exported=True, shape=list(shape), slices=zs,
                                      spacing_mm=list(map(float, reference.header.get_zooms())),
                                      affine=reference.affine.tolist(), omitted_contours=omitted)
        write_json(manifest_file, manifest)
        print(f'Prepared {idx + 1}/{len(manifest["cases"])}: {cid} ({role})', flush=True)
    manifest['prepared'] = True
    write_json(manifest_file, manifest)
    verify_prepared(dst, manifest)
    return manifest


def verify_prepared(dst, manifest):
    dst = Path(dst)
    validate_manifest(manifest)
    if not manifest.get('prepared'):
        raise ValueError('Dataset conversion is incomplete')
    for role in (*ROLES, 'external_unlabeled'):
        rows = [r for r in manifest['cases'] if r['role'] == role]
        expected = {f"{r['case_id']}_z{z:03d}" for r in rows for z in r['slices']}
        for modality in MODALITIES:
            actual = {p.stem for p in (dst / modality / 'images' / role).glob('*.png')}
            if actual != expected:
                raise ValueError(f'Image manifest mismatch: {modality}/{role}')
            label_dir = dst / modality / 'labels' / role
            if role in ('source_train', 'source_val'):
                if {p.stem for p in label_dir.glob('*.txt')} != expected:
                    raise ValueError(f'Label manifest mismatch: {modality}/{role}')
            elif label_dir.exists():
                raise ValueError(f'Target labels must not be present in image dataset: {label_dir}')
        if role in ('source_train', 'source_val', 'target_test'):
            if {p.stem for p in (dst / 'evaluation_masks' / role).glob('*.npz')} != {r['case_id'] for r in rows}:
                raise ValueError(f'Evaluation mask manifest mismatch: {role}')
    if (dst / 'evaluation_masks' / 'target_train').exists():
        raise ValueError('Target adaptation masks must not be exported')


def source_yaml(dst, modality, output):
    # Intentionally contains neither target paths nor any test split.
    config = dict(path=str((Path(dst) / modality).resolve()), train='images/source_train',
                  val='images/source_val', names=NAMES)
    Path(output).write_text(yaml.safe_dump(config, sort_keys=False))
    return str(output)
