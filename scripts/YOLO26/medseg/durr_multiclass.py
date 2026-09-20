"""Class-wise extension of the existing DURR method; no deployment modules."""
from collections import defaultdict

import torch
from ultralytics.utils.ops import xyxy2xywh

from durr_seg import (generate_durr_pseudo_masks, compute_directional_routing_loss,
                      compute_rescue_loss, compute_safe_hallucination_loss)


def select_batch(outputs, indices):
    """Slice nested Segment26 training outputs without another Student forward."""
    if isinstance(outputs, torch.Tensor):
        return outputs[indices]
    if isinstance(outputs, dict):
        return {k: select_batch(v, indices) for k, v in outputs.items()}
    if isinstance(outputs, (tuple, list)):
        return type(outputs)(select_batch(v, indices) for v in outputs)
    raise TypeError(f'Unexpected Segment26 output: {type(outputs)}')


@torch.no_grad()
def decode_teacher(teacher, images):
    teacher.eval()
    (o2o, proto), branches = teacher(images, augment=False, visualize=False)
    if isinstance(proto, (tuple, list)):
        proto = proto[0]
    head = teacher.model[-1]
    o2m = head.postprocess(head._inference(branches['one2many']).permute(0, 2, 1))
    return o2o, o2m, proto


@torch.no_grad()
def generate_multiclass_pseudo(teacher, images, num_classes, thresholds, **kwargs):
    decoded = decode_teacher(teacher, images)
    o2o, o2m, proto = decoded
    labels, masks, reliabilities = [[] for _ in images], [[] for _ in images], [[] for _ in images]
    routes = [[] for _ in images]
    stats = defaultdict(float)
    for cls in range(num_classes):
        filtered = ([r[r[:, 5].long() == cls] for r in o2o],
                    [r[r[:, 5].long() == cls] for r in o2m], proto)
        ls, ms, rs, counts, rt = generate_durr_pseudo_masks(
            teacher, images, decoded_predictions=filtered, reliability_threshold=thresholds[cls], **kwargs)
        for key, value in counts.items():
            stats[key] += float(value)
        for i in range(len(images)):
            labels[i].append(ls[i]); masks[i].append(ms[i]); reliabilities[i].append(rs[i])
            routes[i].append(rt[i])
    for i in range(len(images)):
        labels[i] = torch.cat(labels[i])
        masks[i] = torch.cat(masks[i])
        reliabilities[i] = torch.cat(reliabilities[i])
        order = labels[i][:, 4].argsort(descending=True)
        labels[i], masks[i], reliabilities[i] = labels[i][order], masks[i][order], reliabilities[i][order]
    return labels, masks, reliabilities, dict(stats), routes


def build_multiclass_pseudo_batch(labels, masks, shape, num_classes):
    """Native per-instance targets + class-index semantic map; confidence resolves overlaps."""
    device = labels[0].device
    mh, mw = masks[0].shape[-2:]
    semantic = torch.zeros((len(labels), mh, mw), dtype=torch.long, device=device)
    indices, classes, boxes, all_masks = [], [], [], []
    norm = torch.tensor([shape[3], shape[2], shape[3], shape[2]], device=device)
    for i, (rows, imasks) in enumerate(zip(labels, masks, strict=True)):
        if len(rows) != len(imasks):
            raise ValueError('Pseudo instance/mask count mismatch')
        if not len(rows):
            continue
        cls = rows[:, 5].long()
        if (cls < 0).any() or (cls >= num_classes).any():
            raise ValueError('Pseudo class outside model label space')
        # Lower confidence first: strongest instance owns each overlapping pixel.
        for j in rows[:, 4].argsort():
            semantic[i][imasks[j] > 0.5] = cls[j]
        indices.append(torch.full((len(rows),), i, dtype=torch.long, device=device))
        classes.append(cls)
        boxes.append(xyxy2xywh(rows[:, :4]) / norm)
        all_masks.append(imasks.float())
    if not boxes:
        return None
    return dict(batch_idx=torch.cat(indices), cls=torch.cat(classes), bboxes=torch.cat(boxes),
                masks=torch.cat(all_masks), sem_masks=semantic)


def multiclass_durr_losses(outputs, routes, **hall_kwargs):
    """Apply the original binary DURR losses to their own semantic channels.

    Mean over classes keeps the binary formulation's scale. Rescue and safe
    background decisions are class-specific, including on otherwise nonempty images.
    """
    proto, sem = outputs['one2many']['proto']
    nc = sem.shape[1]
    if any(len(r) != nc for r in routes):
        raise ValueError('Route/semantic class count mismatch')
    sums = [sem.sum() * 0 for _ in range(3)]
    stats = defaultdict(float)
    for cls in range(nc):
        view = {'one2many': {'proto': (proto, sem[:, cls:cls + 1])}}
        class_routes = [r[cls] for r in routes]
        for j, (fn, kwargs) in enumerate(((compute_directional_routing_loss, {}),
                                         (compute_rescue_loss, {}),
                                         (compute_safe_hallucination_loss, hall_kwargs))):
            loss, counts = fn(view, class_routes, **kwargs)
            sums[j] = sums[j] + loss / nc
            for key, value in counts.items():
                stats[key] += value
    return (*sums, dict(stats))
