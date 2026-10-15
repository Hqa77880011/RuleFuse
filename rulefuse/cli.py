import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB


def minimal_conflicts(assertions, allowed):
    conflicts = []
    sensors = list(assertions)
    for count in range(1, len(sensors) + 1):
        for subset in itertools.combinations(sensors, count):
            if any(set(old).issubset(subset) for old in conflicts):
                continue
            possible = allowed.copy()
            for sensor in subset:
                possible &= assertions[sensor]
            if not possible.any():
                conflicts.append(subset)
    return conflicts


def fit(args):
    data = np.load(args.data)
    x, y, mask = data['x'], data['y'].astype(int), data['mask'].astype(bool)
    if x.ndim != 3 or mask.shape != x.shape[:2]:
        raise ValueError('x must be N,S,D and mask N,S')
    ontology = json.loads(Path(args.ontology).read_text())
    worlds = np.array(ontology['worlds'], dtype=int)
    if y.min() < 0 or y.max() >= len(worlds):
        raise ValueError('y indexes ontology worlds')
    train, val = data['split'] == 'train', data['split'] == 'val'
    state = {'ontology': ontology, 'mean': [], 'variance': [], 'prior': [], 'temperature': [], 'reliability': []}
    for sensor in range(x.shape[1]):
        selected = train & mask[:, sensor]
        if set(y[selected]) != set(range(len(worlds))):
            raise ValueError('Every represented world requires training observations for each sensor')
        model = GaussianNB(var_smoothing=1e-5).fit(x[selected, sensor], y[selected])
        state['mean'].append(model.theta_.tolist())
        state['variance'].append(model.var_.tolist())
        state['prior'].append(model.class_prior_.tolist())
        selected = val & mask[:, sensor]
        if not selected.any():
            raise ValueError('Each sensor needs observed validation examples')
        logits = model.predict_log_proba(x[selected, sensor])
        objective = lambda temperature: float(np.mean(logsumexp(logits / temperature, axis=1) - logits[np.arange(len(logits)), y[selected]] / temperature))
        temperature = minimize_scalar(objective, bounds=(.1, 10), method='bounded').x
        state['temperature'].append(float(temperature))
        state['reliability'].append(float(np.mean(logits.argmax(1) == y[selected])))
    Path(args.output).write_text(json.dumps(state, indent=2))


def channel_evidence(x, state):
    means, variances = np.array(state['mean']), np.array(state['variance'])
    logits = -.5 * (np.log(2 * np.pi * variances) + (x[:, None] - means) ** 2 / variances).sum(-1)
    logits += np.log(np.array(state['prior']).clip(1e-12))
    return softmax(logits / np.array(state['temperature'])[:, None], axis=1)


def infer(evidence, available, state, variant, penalty):
    ontology = state['ontology']
    worlds = np.asarray(ontology['worlds'], dtype=int)
    reliability = np.asarray(state['reliability']) * available
    if variant == 'impute':
        evidence = evidence.copy()
        evidence[~available] = np.mean(evidence[available], axis=0) if available.any() else 1 / len(worlds)
        reliability = np.asarray(state['reliability'])
    assertions = {}
    for sensor in np.flatnonzero(available):
        symptom = worlds[evidence[sensor].argmax()]
        assertions[int(sensor)] = np.all(worlds >= symptom, axis=1) if symptom.any() else ~worlds.any(1)
    single = worlds.sum(1) <= 1
    conflicts = minimal_conflicts(assertions, single)
    weights = np.array([np.prod(reliability[list(c)]) for c in conflicts])
    hypotheses = [('physical', -1)] + [('sensor', s) for s in np.flatnonzero(available)]
    log_scores, physical_posteriors, details = [], [], []
    for origin, removed in hypotheses:
        permitted = np.ones(len(worlds), dtype=bool) if origin == 'physical' else single.copy()
        unexplained, explained = 0., 0.
        for subset, weight in zip(conflicts, weights):
            possible = permitted.copy()
            for s in subset:
                if s != removed:
                    possible &= assertions[s]
            resolved = possible.any()
            unexplained += weight * (not resolved)
            explained += weight * resolved
        r = reliability.copy()
        if removed >= 0:
            r[removed] *= .05
        if variant in ('no_provenance', 'late_fusion', 'impute'):
            r = reliability.copy()
        mass = evidence * r[:, None] + (1 - r[:, None]) / len(worlds)
        physical_logits = np.log(mass.clip(1e-12)).sum(0)
        new_conflict = np.zeros(len(worlds))
        for s, compatible in assertions.items():
            if s != removed:
                new_conflict += r[s] * ~compatible
        physical_logits -= penalty * new_conflict
        physical_logits[~permitted] = -1e9
        if variant == 'late_fusion':
            physical_logits = np.log(np.average(evidence, axis=0, weights=reliability + 1e-8).clip(1e-12))
        provenance_cost = unexplained + .1 * (removed >= 0)
        if variant in ('no_provenance', 'late_fusion', 'impute'):
            provenance_cost = 0.
        if variant == 'no_rules':
            physical_logits = np.log(mass.clip(1e-12)).sum(0)
            provenance_cost = .1 * (removed >= 0)
        log_scores.append(float(logsumexp(physical_logits) - penalty * provenance_cost))
        physical_posteriors.append(softmax(physical_logits))
        details.append({'origin': origin, 'sensor': int(removed), 'unexplained': float(unexplained),
                        'explained': float(explained), 'new_conflict': new_conflict.tolist()})
    if variant in ('no_provenance', 'late_fusion', 'impute'):
        provenance = np.zeros(len(hypotheses))
        provenance[0] = 1
    else:
        provenance = softmax(log_scores)
    posterior = provenance @ np.array(physical_posteriors)
    selected = int(provenance.argmax())
    costs = np.array(ontology['action_cost'])
    action_costs = costs @ posterior + np.asarray(ontology.get('inspection_cost', np.zeros(len(costs))))
    return {'probability': posterior.tolist(), 'provenance_probability': provenance.tolist(),
            'provenance': details[selected], 'hypotheses': details, 'ignorance': (1 - reliability).tolist(),
            'conflicts': [list(v) for v in conflicts], 'conflict_weights': weights.tolist(),
            'action': int(action_costs.argmin()), 'expected_cost': float(action_costs.min()),
            'prediction': int(posterior.argmax())}


def evaluate(args):
    data = np.load(args.data)
    state = json.loads(Path(args.model).read_text())
    worlds = np.asarray(state['ontology']['worlds'])
    records, truth = [], []
    for index in np.flatnonzero(data['split'] == 'test'):
        evidence = channel_evidence(data['x'][index], state)
        mask = data['mask'][index].astype(bool)
        record = infer(evidence, mask, state, args.variant, args.penalty)
        record['index'] = int(index)
        record['evidence'] = np.where(mask[:, None], evidence, 0).tolist()
        records.append(record)
        truth.append(int(data['y'][index]))
    truth = np.asarray(truth)
    predictions = np.array([v['prediction'] for v in records])
    probability = np.array([v['probability'] for v in records])
    confidence = probability.max(1)
    correct = predictions == truth
    ece = 0.
    for low in np.linspace(0, 1, 11)[:-1]:
        selected = (confidence > low) & (confidence <= low + .1)
        if selected.any():
            ece += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    ids = np.flatnonzero(data['split'] == 'test')
    compound = worlds[truth].sum(1) > 1
    violations, fidelity = [], []
    for record in records:
        selected = record['provenance']
        violations.append(float(np.asarray(selected['new_conflict'])[record['prediction']] > 0))
        total = sum(record['conflict_weights'])
        fidelity.append(selected['explained'] / total if total > 0 else 1.)
    actual_cost = np.array(state['ontology']['action_cost'])[[v['action'] for v in records], truth]
    actual_cost += np.asarray(state['ontology'].get('inspection_cost', np.zeros(len(state['ontology']['action_cost']))))[[v['action'] for v in records]]
    result = {'macro_f1': f1_score(truth, predictions, average='macro'),
              'multi_fault_f1': f1_score(worlds[truth][compound], worlds[predictions][compound], average='micro', zero_division=0) if compound.any() else None,
              'brier': float(np.mean(np.square(probability - np.eye(len(worlds))[truth]).sum(1))), 'ece': float(ece),
              'rule_violation_rate': float(np.mean(violations)), 'realized_maintenance_cost': float(actual_cost.mean()),
              'explanation_fidelity': float(np.mean(fidelity)),
              'explanation_size': float(np.mean([sum(len(c) for c in v['conflicts']) for v in records]))}
    if 'origin' in data:
        result['conflict_source_accuracy'] = float(np.mean([v['provenance']['origin'] == str(data['origin'][i]) for v, i in zip(records, ids)]))
    if 'bad_sensor' in data:
        selected = data['bad_sensor'][ids] >= 0
        result['sensor_localization_accuracy'] = float(np.mean(np.array([v['provenance']['sensor'] for v in records])[selected] == data['bad_sensor'][ids][selected])) if selected.any() else None
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    (out / 'cases.json').write_text(json.dumps(records, indent=2))
    plt.scatter(confidence, correct.astype(float), alpha=.25)
    plt.xlabel('Fused confidence')
    plt.ylabel('Correct physical state')
    plt.tight_layout()
    plt.savefig(out / 'confidence.png', dpi=180)
    plt.close()


def prepare(args):
    d = np.load(args.source)
    required = ['y', 'mask', 'split', 'group']
    if any(k not in d for k in required):
        raise ValueError('source requires y,mask,split,group and x or signal')
    for group in np.unique(d['group']):
        if len(np.unique(d['split'][d['group'] == group])) != 1:
            raise ValueError('Recording/bearing groups cannot cross partitions')
    if not set(d['split']) == {'train', 'val', 'test'}:
        raise ValueError('All three partitions are required')
    arrays = {k: d[k] for k in d.files if k != 'signal'}
    if 'x' not in arrays:
        signal = np.asarray(d['signal'], dtype=float)
        if signal.ndim != 3 or not np.isfinite(signal).all():
            raise ValueError('signal must contain finite windows,sensors,samples')
        rms = np.sqrt(np.mean(signal ** 2, axis=-1)).clip(1e-8)
        centered = signal - signal.mean(-1, keepdims=True)
        variance = np.mean(centered ** 2, axis=-1).clip(1e-8)
        kurtosis = np.mean(centered ** 4, axis=-1) / variance ** 2
        crest = np.max(np.abs(signal), axis=-1) / rms
        power = np.abs(np.fft.rfft(signal * np.hanning(signal.shape[-1]), axis=-1)) ** 2
        bands = np.array_split(np.arange(power.shape[-1]), 3)
        energy = [np.log1p(power[..., b].mean(-1)) for b in bands]
        arrays['x'] = np.stack([rms, kurtosis, crest, *energy], axis=-1).astype(np.float32)
    np.savez_compressed(args.output, **arrays)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('fit')
    p.add_argument('--data', required=True)
    p.add_argument('--ontology', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('evaluate')
    for name in ['data', 'model', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--variant', choices=['full', 'late_fusion', 'no_provenance', 'no_rules', 'impute'], default='full')
    p.add_argument('--penalty', type=float, default=2.)
    args = parser.parse_args()
    {'prepare': prepare, 'fit': fit, 'evaluate': evaluate}[args.command](args)
