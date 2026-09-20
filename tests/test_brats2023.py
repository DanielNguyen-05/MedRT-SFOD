"""Protocol checks: group leakage, replacement precedence and geometry fidelity."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import nibabel as nib
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/YOLO26/medseg/prepare_brats2023.py"
spec = importlib.util.spec_from_file_location("brats_prepare", SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
run_spec = importlib.util.spec_from_file_location("brats_run", SCRIPT.with_name("run_brats2023.py"))
runner = importlib.util.module_from_spec(run_spec)
run_spec.loader.exec_module(runner)


class BraTSProtocolTest(unittest.TestCase):
    def test_evaluation_uses_original_paths_when_loader_renames_images(self):
        class EmptyModel:
            def predict(self, source, **kwargs):
                return iter(SimpleNamespace(path=f"image{i}.jpg", masks=None) for i in range(len(source)))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gt_masks/test").mkdir(parents=True)
            cid = "BraTS-GLI-00001-000"
            mask = np.zeros((10, 10), np.uint8)
            mask[2:4, 2:4] = 255
            cv2.imwrite(str(root / "gt_masks/test" / f"{cid}_z000.png"), mask)
            cv2.imwrite(str(root / "gt_masks/test" / f"{cid}_z001.png"), np.zeros_like(mask))
            manifest = {"cases": [dict(case_id=cid, patient_id="BraTS-GLI-00001", domain="GLI", split="test", slices=[0, 1])]}
            args = SimpleNamespace(batch=2, imgsz=128, device="cpu", conf=0.25)
            result = runner.evaluate(EmptyModel(), root, manifest, "test", args)
            self.assertEqual(result["cases"][0]["fn"], 4)
            self.assertEqual(result["cases"][0]["negative_slices"], 1)
            self.assertEqual(result["overall"]["patient_macro"]["dice"], 0.0)

    def test_overlap_and_equal_patient_weighting(self):
        self.assertEqual(runner.scores(0, 0, 0)["dice"], 1.0)
        self.assertEqual(runner.scores(0, 3, 0)["dice"], 0.0)
        self.assertEqual(runner.scores(0, 0, 3)["dice"], 0.0)
        self.assertAlmostEqual(runner.scores(3, 1, 2)["dice"], 2 / 3)
        rows = [dict(patient_id="A", **runner.scores(5, 0, 0)),
                dict(patient_id="A", **runner.scores(5, 0, 0)),
                dict(patient_id="B", **runner.scores(0, 0, 5))]
        summary = runner.summarize(rows)
        self.assertEqual(summary["patient_macro"]["dice"], 0.5)
        self.assertEqual(summary["patients"], 2)

    def test_repeated_scans_stay_together_and_order_does_not_matter(self):
        records = [dict(case_id=f"BraTS-{d}-{p:05d}-{visit:03d}",
                        patient_id=f"BraTS-{d}-{p:05d}", domain=d)
                   for d in ("GLI", "MEN", "PED") for p in range(20) for visit in range(2)]
        split = prep.split_patients(records, 29, 0.15, 0.15)
        reversed_split = prep.split_patients(records[::-1], 29, 0.15, 0.15)
        self.assertEqual({r["case_id"]: r["split"] for r in split},
                         {r["case_id"]: r["split"] for r in reversed_split})
        groups = [{r["patient_id"] for r in split if r["split"] == s} for s in prep.SPLITS]
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        self.assertEqual([len(g) for g in groups], [42, 9, 9])

    def test_fix_overrides_original_and_unlabeled_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder, cid, labeled in [("BraTS-MEN-Train", "BraTS-MEN-00001-000", True),
                                         ("BraTS-MEN-TRAIN-FIX-V4", "BraTS-MEN-00001-000", True),
                                         ("ValidationData", "BraTS-MEN-00002-000", False)]:
                case = root / folder / cid
                case.mkdir(parents=True)
                for m in (*prep.MODALITIES, "t1n", *(("seg",) if labeled else ())):
                    (case / f"{cid}-{m}.nii.gz").touch()
            records, audit = prep.discover(root)
            self.assertEqual(len(records), 1)
            self.assertIn("FIX-V4", records[0]["directory"])
            self.assertEqual(len(audit["men_fixed_replacements"]), 1)
            self.assertEqual(audit["unlabeled_cases_by_domain"], {"MEN": 1})

    def test_export_preserves_masks_negative_slices_and_channel_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cid = "BraTS-GLI-00001-000"
            case = root / cid
            case.mkdir()
            seg = np.zeros((8, 10, 3), dtype=np.uint8)
            seg[2:6, 3:8, 1] = 3
            seg[3, 4, 1] = 0  # Original GT must retain a hole even if polygons cannot.
            affine = np.diag([1.0, 2.0, 3.0, 1.0])
            nib.save(nib.Nifti1Image(seg, affine), case / f"{cid}-seg.nii.gz")
            volumes = []
            for i, modality in enumerate(prep.MODALITIES):
                volume = np.arange(240, dtype=np.float32).reshape(8, 10, 3)
                volume = np.roll(volume, i, axis=0)
                volumes.append(volume)
                nib.save(nib.Nifti1Image(volume, affine), case / f"{cid}-{modality}.nii.gz")
            for kind in ("images", "labels", "gt_masks"):
                (root / "out" / kind / "train").mkdir(parents=True)
            row = prep.export_case(dict(case_id=cid, patient_id="BraTS-GLI-00001", domain="GLI",
                                        directory=str(case), split="train"), root / "out", 1)
            self.assertEqual(row["slices"], [0, 1, 2])
            self.assertEqual(row["negative_slices"], 2)
            self.assertEqual(row["spacing_mm"], [1.0, 2.0, 3.0])
            gt = cv2.imread(str(root / "out/gt_masks/train" / f"{cid}_z001.png"), 0)
            np.testing.assert_array_equal(gt > 0, seg[:, :, 1].T > 0)
            rgb = cv2.imread(str(root / "out/images/train" / f"{cid}_z001.png"))[:, :, ::-1]
            for i in range(3):
                np.testing.assert_array_equal(rgb[:, :, i], prep.normalize(volumes[i])[:, :, 1].T)
            self.assertEqual((root / "out/labels/train" / f"{cid}_z000.txt").read_text(), "")
            # A physical mismatch must fail instead of silently using the wrong mask.
            nib.save(nib.Nifti1Image(volumes[0], np.eye(4)), case / f"{cid}-t1c.nii.gz")
            with self.assertRaisesRegex(ValueError, "geometry mismatch"):
                prep.export_case(dict(case_id=cid, directory=str(case), split="train"), root / "out", 1)


if __name__ == "__main__":
    unittest.main()
