import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/YOLO26/medseg'))
from brats2024_data import (ROLES, MODALITIES, split_patients, validate_manifest, prepare,
                            verify_prepared, normalize, polygons)
from brats2024_eval import binary_metrics, regions_from_result
from brats2024_experiments import check_stage, mark_complete
from brats2024_train import MRITargetDataset
from durr_multiclass import (generate_multiclass_pseudo, build_multiclass_pseudo_batch,
                             multiclass_durr_losses, select_batch)
from durr_seg import compute_directional_routing_loss, compute_rescue_loss, compute_safe_hallucination_loss


def route(shape=(8, 8), empty=False):
    zero = torch.zeros(shape)
    one = torch.ones(shape)
    return dict(directional_target=one * .8, directional_weight=one, signed_delta=one * .2,
                rescue_mask=one, rescue_weight=one, safe_bg_mask=one, teacher_empty=empty)


class DataProtocolTests(unittest.TestCase):
    def test_repeated_visits_never_cross_roles(self):
        rows = [dict(case_id=f'BraTS-GLI-{p:05d}-{v:03d}', patient_id=f'BraTS-GLI-{p:05d}',
                     folder='training_data1_v2') for p in range(30) for v in (100, 101)]
        a = split_patients(rows, [.45, .1, .3, .15], 29)
        self.assertEqual(a, split_patients(list(reversed(rows)), [.45, .1, .3, .15], 29)[::-1])
        validate_manifest({'cases': a})
        corrupted = copy.deepcopy(a)
        corrupted[0]['role'] = next(r for r in ROLES if r != corrupted[0]['role'])
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_manifest({'cases': corrupted})

    def test_prepare_never_opens_adaptation_or_external_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, dst = Path(temp) / 'raw', Path(temp) / 'prepared'
            records = []
            for i, role in enumerate((*ROLES, 'external_unlabeled')):
                cid = f'BraTS-GLI-{i:05d}-100'
                case = raw / cid; case.mkdir(parents=True)
                volume = np.arange(16 * 16 * 4, dtype=np.float32).reshape(16, 16, 4)
                mask = np.zeros(volume.shape, dtype=np.uint8)
                for cls in range(1, 5):
                    mask[cls * 2:cls * 2 + 2, 2:6] = cls
                for modality in MODALITIES:
                    nib.save(nib.Nifti1Image(volume, np.eye(4)), case / f'{cid}-{modality}.nii.gz')
                if role in ('source_train', 'source_val', 'target_test'):
                    nib.save(nib.Nifti1Image(mask, np.eye(4)), case / f'{cid}-seg.nii.gz')
                records.append(dict(case_id=cid, patient_id=cid.rsplit('-', 1)[0], directory=cid, role=role))
            plan = dict(cases=records, stride=1, split_sha256='synthetic', smoke=False, prepared=False)
            # Adaptation/external masks do not even exist: successful preparation proves image-only access.
            result = prepare(raw, dst, plan)
            self.assertTrue(result['prepared'])
            self.assertFalse((dst / 'evaluation_masks/target_train').exists())
            with np.load(dst / 'evaluation_masks/target_test' / f"{records[3]['case_id']}.npz") as data:
                self.assertEqual(set(np.unique(data['mask'])), {0, 1, 2, 3, 4})
            (dst / 't1n/labels/target_train').mkdir()
            with self.assertRaisesRegex(ValueError, 'Target labels'):
                verify_prepared(dst, result)

    def test_mri_augmentation_preserves_single_modality(self):
        gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
        image = np.repeat(gray[..., None], 3, axis=2)
        for _ in range(10):
            strong = MRITargetDataset.strong_photo(image)
            np.testing.assert_array_equal(strong[..., 0], strong[..., 1])
            np.testing.assert_array_equal(strong[..., 0], strong[..., 2])
        self.assertEqual(normalize(np.zeros((3, 3, 3))).sum(), 0)

    def test_all_label_ids_survive_polygon_conversion(self):
        mask = np.zeros((20, 20), np.uint8)
        for cls in range(1, 5):
            mask[cls * 3:cls * 3 + 2, 3:8] = cls
        rows, omitted = polygons(mask)
        self.assertEqual({int(r.split()[0]) for r in rows}, {0, 1, 2, 3})
        self.assertEqual(omitted, 0)

    def test_completed_stage_checks_protocol_and_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'stage'
            self.assertTrue(check_stage(out, {'seed': 29}))
            (out / 'weights.pt').write_bytes(b'checkpoint')
            mark_complete(out, ['weights.pt'])
            self.assertFalse(check_stage(out, {'seed': 29}))
            with self.assertRaisesRegex(ValueError, 'Changed'):
                check_stage(out, {'seed': 30}, True)
            (out / 'weights.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed/missing'):
                check_stage(out, {'seed': 29})


class ModelAndMetricTests(unittest.TestCase):
    def test_pseudo_semantics_use_class_ids_and_confidence(self):
        labels = [torch.tensor([[0., 0., 8., 8., .9, 3.], [0., 0., 8., 8., .6, 1.]])]
        masks = [torch.ones(2, 8, 8)]
        batch = build_multiclass_pseudo_batch(labels, masks, (1, 3, 8, 8), 4)
        self.assertTrue((batch['sem_masks'] == 3).all())
        self.assertEqual(batch['cls'].tolist(), [3, 1])

    def test_multiclass_routing_never_matches_other_class(self):
        # Class 0 O2O and coincident class 1 O2M: class 1 must be rescued,
        # never treated as a witness for class 0. Decode exactly once.
        oo = torch.tensor([[[0., 0., 16., 16., .9, 0., 8.]]])
        om = torch.tensor([[[0., 0., 16., 16., .95, 1., 8.]]])
        proto = torch.ones(1, 1, 8, 8)
        with patch('durr_multiclass.decode_teacher', return_value=(oo, om, proto)) as decode:
            labels, masks, _, _, routes = generate_multiclass_pseudo(
                None, torch.zeros(1, 3, 16, 16), 4, [0.] * 4, min_mask_pixels=1)
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(set(labels[0][:, 5].tolist()), {0., 1.})
        self.assertEqual(float(routes[0][0]['directional_weight'].sum()), 0.)
        self.assertGreater(float(routes[0][1]['rescue_mask'].sum()), 0.)
        self.assertTrue(routes[0][2]['teacher_empty'])

    def test_binary_loss_compatibility_and_class_gradient_isolation(self):
        sem = torch.zeros(1, 1, 8, 8, requires_grad=True)
        outputs = {'one2many': {'proto': (torch.zeros(1, 2, 8, 8), sem)}}
        r = route()
        new = multiclass_durr_losses(outputs, [[r]])
        for i, fn in enumerate((compute_directional_routing_loss, compute_rescue_loss, compute_safe_hallucination_loss)):
            torch.testing.assert_close(new[i], fn(outputs, [r])[0])
        sem4 = torch.zeros(1, 4, 8, 8, requires_grad=True)
        outputs['one2many']['proto'] = (torch.zeros(1, 2, 8, 8), sem4)
        routes = [route() for _ in range(4)]
        for r in routes[:3]:
            r['directional_weight'].zero_(); r['rescue_mask'].zero_()
        dl, _, _, _ = multiclass_durr_losses(outputs, [routes])
        dl.backward()
        self.assertEqual(float(sem4.grad[:, :3].abs().sum()), 0.)
        self.assertGreater(float(sem4.grad[:, 3].abs().sum()), 0.)

    def test_real_yolo26s_four_class_native_and_durr_backward(self):
        from ultralytics.nn.tasks import SegmentationModel
        from smoke_stage2_mt_seg import ensure_seg_loss_args
        from segmard_seg import compute_segmard_loss
        from stage2_dense_sfseg_durr import SegmentInputFeatureHook
        torch.set_num_threads(4)
        model = SegmentationModel('yolo26s-seg.yaml', nc=4, verbose=False)
        ensure_seg_loss_args(model, 2)
        model.train()
        hook = SegmentInputFeatureHook(model)
        outputs = model(torch.randn(2, 3, 64, 64))
        mh, mw = outputs['one2many']['proto'][0].shape[-2:]
        labels = [torch.tensor([[4., 4., 60., 60., .9, float(c)] for c in range(4)]), torch.zeros(0, 6)]
        masks = [torch.zeros(4, mh, mw), torch.zeros(0, mh, mw)]
        for c in range(4):
            masks[0][c, c * 2:c * 2 + 2, 2:8] = 1
        batch = build_multiclass_pseudo_batch(labels[:1], masks[:1], (1, 3, 64, 64), 4)
        criterion = model.init_criterion()
        captured = []
        handle = criterion.one2many.bcedice_loss.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[1].detach().clone()))
        full_batch = build_multiclass_pseudo_batch(labels, masks, (2, 3, 64, 64), 4)
        criterion(outputs, full_batch)
        self.assertTrue(captured, 'Native semantic supervision was not exercised')
        self.assertEqual(float(captured[0][1].sum()), 0., 'Blank source slice must not become NETC foreground')
        handle.remove()
        native = criterion(select_batch(outputs, [0]), batch)[0].sum()
        cfg = yaml.safe_load((ROOT / 'configs/experiments/brats2024_sfseg.yaml').read_text())
        mard, _ = compute_segmard_loss([f[:1] for f in hook.latest], labels[:1], masks[:1],
                                       64, 64, argparse.Namespace(**cfg['adaptation']['segmard']))
        routes = [[route((mh, mw), empty=i == 1) for _ in range(4)] for i in range(2)]
        dl, rl, hl, _ = multiclass_durr_losses(outputs, routes, student_threshold=.1, area_threshold=.01)
        loss = native + .05 * mard + .1 * (dl + rl + hl)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grad = model.model[-1].proto.semseg[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue((grad.flatten(1).abs().sum(1) > 0).all())
        hook.close()

    def test_surface_distance_uses_physical_spacing_and_explicit_empty_policy(self):
        gt = np.zeros((7, 7, 7), bool); gt[2, 3, 3] = True
        pred = np.zeros_like(gt); pred[3, 3, 3] = True
        metrics = binary_metrics(pred, gt, (2., 1., 1.))
        self.assertEqual(metrics['asd_mm'], 2.)
        self.assertEqual(metrics['hd95_mm'], 2.)
        self.assertEqual(metrics['dice'], 0.)
        empty = np.zeros_like(gt)
        self.assertIsNone(binary_metrics(empty, gt, (1., 1., 1.))['asd_mm'])
        self.assertEqual(binary_metrics(empty, empty, (1., 1., 1.))['dice'], 1.)


if __name__ == '__main__':
    unittest.main()
