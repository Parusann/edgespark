"""Post-hoc recalibration of the confidence head (spec section 9.4, "the fix").

Two cheap methods, both fit on a small held-out calibration set and both
*monotone* in the head's raw score, they rescale confidence without changing the
ranking of which positions look more likely to be accepted, so they can never
turn a good proposal into a bad one:

* :class:`TemperatureScaler`, one parameter ``T``. ``p_cal = sigmoid(z / T)``
  where ``z`` is the head's logit. Divides the confidence "sharpness" down
  (``T > 1``) or up (``T < 1``). This is Method A in the spec and the one that
  does most of the work in practice.

* :class:`PlattScaler`, two parameters. ``p_cal = sigmoid(a * z + b)``. A slope
  and a bias; strictly more expressive than temperature scaling, useful when
  quantization introduces a *shift* as well as a sharpness change.

Both are fit by minimising negative log-likelihood with a plain 1-D / 2-D Newton
step in ``numpy``, no scipy, no torch, so recalibration runs anywhere the metrics
do. Inputs are the head's predicted probabilities; we recover the logit
internally, so callers never have to plumb logits around.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-7


def _to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64).ravel(), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable branchwise sigmoid.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _nll(prob: np.ndarray, outcome: np.ndarray) -> float:
    prob = np.clip(prob, _EPS, 1.0 - _EPS)
    return float(-np.mean(outcome * np.log(prob) + (1 - outcome) * np.log(1 - prob)))


@dataclass
class TemperatureScaler:
    """Single-parameter temperature scaling."""

    temperature: float = 1.0
    fitted: bool = False

    def fit(self, confidence: np.ndarray, outcome: np.ndarray, max_iter: int = 100) -> TemperatureScaler:
        z = _to_logit(confidence)
        y = np.asarray(outcome, dtype=np.float64).ravel()
        # Optimise beta = 1/T by Newton's method on the logistic NLL with the
        # single feature z. beta stays > 0 so the mapping is monotone increasing.
        beta = 1.0
        for _ in range(max_iter):
            p = _sigmoid(beta * z)
            grad = float(np.mean((p - y) * z))
            hess = float(np.mean(p * (1 - p) * z * z)) + 1e-9
            step = grad / hess
            beta -= step
            beta = max(beta, 1e-3)
            if abs(step) < 1e-8:
                break
        self.temperature = 1.0 / beta
        self.fitted = True
        return self

    def transform(self, confidence: np.ndarray) -> np.ndarray:
        return _sigmoid(_to_logit(confidence) / self.temperature)

    def fit_transform(self, confidence: np.ndarray, outcome: np.ndarray) -> np.ndarray:
        return self.fit(confidence, outcome).transform(confidence)


@dataclass
class PlattScaler:
    """Two-parameter Platt scaling (logistic regression on the head's logit)."""

    slope: float = 1.0
    bias: float = 0.0
    fitted: bool = False

    def fit(self, confidence: np.ndarray, outcome: np.ndarray, max_iter: int = 100) -> PlattScaler:
        z = _to_logit(confidence)
        y = np.asarray(outcome, dtype=np.float64).ravel()
        X = np.stack([z, np.ones_like(z)], axis=1)  # [n, 2] -> [slope, bias]
        w = np.zeros(2)
        nll = _nll(_sigmoid(X @ w), y)
        for _ in range(max_iter):
            p = _sigmoid(X @ w)
            grad = X.T @ (p - y) / len(y)
            # Floor the IRLS weights before forming the Hessian. On a badly
            # over-confident head (the NF4 case) an undamped Newton step overshoots,
            # the sigmoid saturates, W = p(1-p) collapses to ~0, and the Hessian
            # degenerates to just the ridge term, sending the next step to ~1e6.
            # The floor keeps curvature information; the line search below refuses
            # any step that does not actually decrease the loss.
            W = np.clip(p * (1 - p), 1e-6, None)
            hess = (X.T * W) @ X / len(y) + 1e-6 * np.eye(2)
            step = np.linalg.solve(hess, grad)
            # Backtracking line search: shrink the step until NLL decreases.
            t = 1.0
            while t > 1e-4:
                cand = w - t * step
                cand_nll = _nll(_sigmoid(X @ cand), y)
                if cand_nll <= nll:
                    break
                t *= 0.5
            if cand_nll > nll:  # no descent direction found; converged
                break
            w, prev_nll = cand, nll
            nll = cand_nll
            if np.linalg.norm(t * step) < 1e-9 or prev_nll - nll < 1e-12:
                break
        self.slope, self.bias = float(w[0]), float(w[1])
        self.fitted = True
        return self

    def transform(self, confidence: np.ndarray) -> np.ndarray:
        return _sigmoid(self.slope * _to_logit(confidence) + self.bias)

    def fit_transform(self, confidence: np.ndarray, outcome: np.ndarray) -> np.ndarray:
        return self.fit(confidence, outcome).transform(confidence)


def fit_recalibrator(
    confidence: np.ndarray,
    outcome: np.ndarray,
    method: str = "temperature",
):
    """Fit and return a recalibrator by name.

    ``method`` is ``"temperature"`` or ``"platt"``. The returned object exposes
    ``transform`` so downstream code (the drafter's calibrated ``a_j`` path and
    the verification-length policy) is agnostic to which method was chosen.
    """
    if method == "temperature":
        return TemperatureScaler().fit(confidence, outcome)
    if method == "platt":
        return PlattScaler().fit(confidence, outcome)
    raise ValueError(f"unknown method {method!r}; expected 'temperature' or 'platt'")
