#!/usr/bin/env python3
"""Fit operating points on development partitions; report previously observed holdouts separately.
No images, model weights, or private paths are emitted in the public report.
"""
import argparse, collections, hashlib, json, pathlib
import numpy as np


def multiply(a, b):
    # Some Accelerate versions retain status flags despite finite results.
    with np.errstate(all='ignore'):
        result = a @ b
    if not np.isfinite(result).all():
        raise ArithmeticError('Non-finite matrix result')
    return result


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or (norm < 1e-12).any():
        raise ValueError('Invalid embedding')
    return x / norm


def load(path):
    rows = json.loads(pathlib.Path(path).read_text())['rows']
    return [r for r in rows if r['encoder'] == 'siglip2' and r['view'] == 'original']


def gallery(rows, classes):
    result = []
    for label in classes:
        groups = collections.defaultdict(list)
        for row in rows:
            if row['split'] == 'gallery' and row['knowledgeID'] == label:
                groups[row.get('groupID', row['imageID'])].append(row)
        groups = [sorted(v, key=lambda r: r['imageID']) for _, v in sorted(groups.items())]
        order = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
        result.extend(order[:min(10, 300 // len(classes))])
    return result


def fit(rows, classes):
    training = gallery(rows, classes)
    x = unit([r['vector'] for r in training])
    labels = np.array([classes.index(r['knowledgeID']) for r in training])
    # Same parent/group/class weighting as production for original-only, distinct sources.
    weights = np.array([len(training) / (len(classes) * int((labels == c).sum())) for c in labels])
    mean = np.average(x, axis=0, weights=weights)
    target = np.eye(len(classes))[labels]
    target_mean = np.average(target, axis=0, weights=weights)
    centered = (x - mean) * np.sqrt(weights[:, None])
    rhs = (target - target_mean) * np.sqrt(weights[:, None])
    regularization = max(np.square(centered).sum() / weights.sum(), 1e-6)
    matrix = multiply(centered, centered.T) + np.eye(len(x)) * regularization
    with np.errstate(all='ignore'):
        solution = np.linalg.solve(matrix, rhs)
    if np.linalg.norm(multiply(matrix, solution) - rhs) > 1e-6 * max(1, np.linalg.norm(rhs)):
        raise ArithmeticError('Invalid Ridge solution')
    coefficients = multiply(centered.T, solution)
    center = unit(np.average(x, axis=0, weights=weights)[None])[0]
    def scores(q):
        if len(classes) == 1:
            return np.maximum(0, 2 - 2 * multiply(q, center[:, None]))
        return -(multiply(q - mean, coefficients) + target_mean)
    train_scores = scores(x)
    if len(classes) > 1:
        ordered = np.sort(train_scores, axis=1)
        separation = max(float(np.median([np.median((ordered[:, 1] - ordered[:, 0])[labels == c]) for c in range(len(classes))])), 1e-6)
    else:
        separation = 1
    return training, x, labels, scores, separation


def predictions(rows, classes, queries):
    training, x, labels, scores, separation = fit(rows, classes)
    q = unit([r['vector'] for r in queries])
    s = scores(q)
    order = np.argsort(s, axis=1, kind='stable')
    best = order[:, 0]
    distances = np.maximum(0, 2 - 2 * multiply(q, x.T))
    native = np.stack([distances[:, labels == c].min(1) for c in range(len(classes))], axis=1)
    native_best = native[np.arange(len(q)), best]
    expected = np.array([classes.index(r['knowledgeID']) if r['knowledgeID'] in classes else -1 for r in queries])
    gap = (s[np.arange(len(q)), order[:, 1]] - s[np.arange(len(q)), best]) / separation if len(classes) > 1 else np.full(len(q), np.inf)
    return dict(known=expected >= 0, correct=best == expected, distance=native_best,
                score=s[np.arange(len(q)), best], gap=gap, classes=len(classes),
                prototype=len(classes) == 1, labels=[r['knowledgeID'] for r in queries])


def metrics(pred, maximum, margin):
    eligible = (pred['distance'] <= maximum) & (pred['score'] <= (maximum if pred['prototype'] else 0))
    accepted = eligible & (pred['gap'] + 1e-9 >= margin)
    def macro(values, known):
        ids = sorted({label for label, k in zip(pred['labels'], pred['known']) if bool(k) == known})
        return float(np.mean([values[np.array([label == c for label in pred['labels']])].mean() for c in ids])) if ids else 0
    return dict(knownImages=int(pred['known'].sum()), unknownImages=int((~pred['known']).sum()),
                top1=macro(pred['correct'], True), knownSingleAcceptRate=macro(accepted, True),
                correctSingleAcceptRate=macro(accepted & pred['correct'], True),
                unknownSingleFalseAcceptRate=macro(accepted, False), unknownCandidateRate=macro(eligible, False))


def cells(rows, calibration):
    all_classes = sorted({r['knowledgeID'] for r in rows if r['split'] == 'gallery'})
    unknown_ids = sorted({r['knowledgeID'] for r in rows if r['split'] == 'unknown'})
    unknown_ids = set(unknown_ids[:max(1, len(unknown_ids)//2)] if calibration else unknown_ids[max(1, len(unknown_ids)//2):])
    result = []
    for count in sorted({1, min(3, len(all_classes)), min(10, len(all_classes)), len(all_classes)}):
        for seed in [20261108, 20261109]:
            rng = np.random.default_rng(seed)
            classes = sorted(rng.choice(all_classes, count, replace=False))
            queries = [r for r in rows if (r['split'] == ('validation' if calibration else 'test') and r['knowledgeID'] in classes)
                       or (r['split'] == 'unknown' and r['knowledgeID'] in unknown_ids)]
            if queries:
                result.append(predictions(rows, classes, queries))
    return result


def aggregate(cells, maximum, margin):
    values = [metrics(c, maximum, margin) for c in cells]
    keys = ['top1', 'knownSingleAcceptRate', 'correctSingleAcceptRate', 'unknownSingleFalseAcceptRate', 'unknownCandidateRate']
    return {key: float(np.mean([v[key] for v in values])) for key in keys}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development', nargs='+', required=True)
    p.add_argument('--holdout', nargs='*', default=[])
    p.add_argument('--output', required=True)
    a = p.parse_args()
    development = [load(f) for f in a.development]
    calibration = [c for rows in development for c in cells(rows, True)]
    # Search only development calibration. Known-correct coverage and unknown
    # false acceptance have equal weight at the standard preset. Candidate
    # leakage is penalized too because ambiguous contexts may reach the VLM.
    distances = np.unique(np.r_[0, np.quantile(np.concatenate([c['distance'] for c in calibration]), np.linspace(0, 1, 81)), 4])
    margin_points = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
    best = None
    for margin in margin_points:
        for maximum in distances:
            e = aggregate(calibration, maximum, margin)
            utility = e['correctSingleAcceptRate'] - e['unknownSingleFalseAcceptRate'] - .25 * e['unknownCandidateRate']
            key = (utility, -e['unknownSingleFalseAcceptRate'], e['correctSingleAcceptRate'], -maximum)
            if best is None or key > best[0]:
                best = (key, float(maximum), margin, e)
    margin = best[2]
    limits = []
    for penalty in [8, 4, 2, 1, .5]:
        candidates = []
        for maximum in distances:
            e = aggregate(calibration, maximum, margin)
            utility = e['correctSingleAcceptRate'] - penalty * (e['unknownSingleFalseAcceptRate'] + .25 * e['unknownCandidateRate'])
            candidates.append(((utility, -e['unknownSingleFalseAcceptRate'], e['correctSingleAcceptRate'], -maximum), float(maximum)))
        limits.append(max(candidates)[1])
    report = dict(version=2, modelRevision='75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2',
                  metric='squared-l2-unit-embedding', developmentSourceNames=[pathlib.Path(f).stem for f in a.development],
                  holdoutSourceNames=[pathlib.Path(f).stem for f in a.holdout], referenceImagesPerClass=10, ridgeRegularization=1,
                  maximumDistances=limits, competitionMarginMultiplier=margin/.1,
                  selectionObjective='Correct known single acceptance minus unknown single acceptance and 0.25 * unknown candidate leakage; margin selected at equal error cost; distance penalties 8/4/2/1/0.5 (user-selected standard cost 2)',
                  sourcePartitions=[dict(galleryImages=sum(r['split']=='gallery' for r in rows),knownCalibrationImages=sum(r['split']=='validation' for r in rows),knownCheckImages=sum(r['split']=='test' for r in rows),unknownImages=sum(r['split']=='unknown' for r in rows)) for rows in development],
                  calibration=[aggregate(calibration, t, margin) for t in limits],
                  developmentHoldout=[aggregate([c for rows in development for c in cells(rows, False)], t, margin) for t in limits],
                  previouslyObservedFinalHoldout=[],
                  limitations=['Previously observed feature data, not a new sealed independent study',
                               'Development unknown identities split into calibration and holdout; validation selects, test only reports',
                               'One-class Prototype and multi-class Ridge, original images only, balanced across dataset and inventory size',
                               'No iPhone live-input, latency, heat or final VLM evaluation',
                               'Candidate leakage is an upper bound at the native/score gate before ambiguous-candidate delivery count checks',
                               'Feature vectors re-extracted on macOS with the released FP16 model and PIL-compatible production preprocessing'])
    for filename in a.holdout:
        rows = load(filename)
        held = cells(rows, False)
        report['previouslyObservedFinalHoldout'].append([aggregate(held, t, margin) for t in limits])
    pathlib.Path(a.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
