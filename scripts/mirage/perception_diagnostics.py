"""Candidate proposal/mask decoding and diagnostic native matching. No names or OCR."""
import numpy as np
from PIL import Image, ImageDraw


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def decode(outputs, meta):
    raw = outputs['detections'][0]
    result = {}
    for kind, channel, limit in [('objects', 0, 64), ('text', 1, 32)]:
        heat = sigmoid(raw[..., channel])
        pad = np.pad(heat, 1, constant_values=-np.inf)
        peak = np.maximum.reduce([pad[i:i+64, j:j+64] for i in range(3) for j in range(3)])
        valid = ((np.arange(64)[:, None] * 8 < meta['h']) & (np.arange(64)[None, :] * 8 < meta['w']))
        cells = np.argwhere((heat == peak) & (heat >= 0.3) & valid)
        order = sorted(cells.tolist(), key=lambda cell: (-float(heat[tuple(cell)]), cell[0], cell[1]))[:limit]
        proposals = []
        for i, j in order:
            width, height = abs(float(raw[i, j, 2+2*channel])), abs(float(raw[i, j, 3+2*channel]))
            cx, cy = j + float(raw[i, j, 6+2*channel]), i + float(raw[i, j, 7+2*channel])
            box = np.array([cx-width/2, cy-height/2, cx+width/2, cy+height/2]) / 64
            box[[0, 2]] = box[[0, 2]].clip(0, meta['w']/512)
            box[[1, 3]] = box[[1, 3]].clip(0, meta['h']/512)
            if 'coordinate_scale' in meta:
                scale = np.asarray(meta['coordinate_scale'], dtype=np.float64)
                if scale.shape != (4,) or not np.isfinite(scale).all() or not (scale > 0).all():
                    raise ValueError('invalid explicit coordinate inverse')
            else:
                scale = np.array([512/meta['r']/meta['W'], 512/meta['r']/meta['H']] * 2)
            normalized = box * scale
            proposal = {'bbox': (normalized * 1000).round().astype(int).tolist(), 'score': float(heat[i, j]), 'cell': [i, j]}
            if kind == 'objects':
                logit = np.einsum('r,rhw->hw', outputs['coefficients'][0, :, i, j], outputs['prototypes'][0])
                centers = (np.arange(128) + 0.5)/128
                inside = ((centers[:, None] >= box[1]) & (centers[:, None] <= box[3]) &
                          (centers[None, :] >= box[0]) & (centers[None, :] <= box[2]))
                inside[min(127, int((box[1]+box[3])*64)), min(127, int((box[0]+box[2])*64))] = True
                mask = (sigmoid(logit) * inside) >= 0.5
                rows = np.minimum(127, ((np.arange(128)+0.5)/128 * meta['h']/4).astype(int))
                cols = np.minimum(127, ((np.arange(128)+0.5)/128 * meta['w']/4).astype(int))
                proposal['_mask'] = mask[np.ix_(rows, cols)]
            proposals.append(proposal)
        result[kind] = proposals
    return result


def box_iou(a, b):
    if not all(np.isfinite(v) for v in list(a) + list(b)):
        raise ValueError('nonfinite box')
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = lambda x: max(0, x[2]-x[0]) * max(0, x[3]-x[1])
    union = area(a)+area(b)-intersection
    return intersection/union if union else 0.0


def polygon_mask(polygons):
    canvas = Image.new('1', (128, 128))
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        if len(polygon) < 6 or len(polygon) % 2:
            raise ValueError('invalid polygon')
        draw.polygon([(polygon[i] * 128/1000, polygon[i+1] * 128/1000) for i in range(0, len(polygon), 2)], fill=1)
    return np.asarray(canvas, dtype=bool)


def compare_native(proposals, native, *, compare_scores=True):
    """Greedy one-to-one IoU matching >=0.5. No semantic or accuracy acceptance."""
    report = {'status': 'DIAGNOSTIC_NOT_PARITY_ACCEPTANCE', 'matching': 'greedy maximum box IoU >= 0.5; one-to-one, names ignored',
              'mask_grid': '128x128 original-image coordinates; native polygon union vs thresholded candidate nearest samples',
              'score_scope': 'COMPARED' if compare_scores else 'NOT_REQUESTED_GEOMETRY_ONLY',
              'object_names_agreement': None, 'ocr_text_agreement': None, 'groups': {}}
    for kind in ('objects', 'text'):
        candidate, original = proposals[kind], native[kind]
        possible = sorted([(box_iou(a['bbox'], b['bbox']), i, j) for i, a in enumerate(candidate) for j, b in enumerate(original)], reverse=True)
        used_a, used_b, pairs = set(), set(), []
        for iou, i, j in possible:
            if iou < 0.5 or i in used_a or j in used_b:
                continue
            used_a.add(i); used_b.add(j)
            pair = {'candidate': i, 'native': j, 'bbox_iou': iou,
                    'score_absolute_difference': abs(candidate[i]['score']-original[j]['score']) if compare_scores else None,
                    'mask_iou': None, 'mask_status': 'NOT_AVAILABLE'}
            if kind == 'objects' and original[j].get('mask'):
                try:
                    mask = polygon_mask(original[j]['mask'])
                    predicted = candidate[i]['_mask']
                    union = np.count_nonzero(mask | predicted)
                    pair['mask_iou'] = float(np.count_nonzero(mask & predicted)/union) if union else None
                    pair['mask_status'] = 'MEASURED' if union else 'EMPTY_UNION'
                except ValueError as error:
                    pair.update(mask_status='INVALID_NATIVE_POLYGON', mask_error=str(error), native_mask=original[j]['mask'])
            pairs.append(pair)
        report['groups'][kind] = {'candidate_count': len(candidate), 'native_count': len(original),
                                  'matched_count': len(pairs), 'unmatched_candidate': len(candidate)-len(pairs),
                                  'unmatched_native': len(original)-len(pairs), 'pairs': pairs}
    return report
