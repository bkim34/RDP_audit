"""Utility functions to audit DP algorithms using (eps, delta)-DP or GDP"""
import numpy as np
from scipy.stats import binomtest, norm
from scipy.optimize import root_scalar
from privacy_estimates import AttackResults, compute_eps_lo
import concurrent.futures
from tqdm import tqdm
import math

def compute_mu_lower_gdp(results, alpha):
    """Convert attack counts to a lower bound on mu-GDP at significance level alpha."""
    _, fpr_r = binomtest(int(results.FP), int(results.N)).proportion_ci(confidence_level=1 - 2 * alpha)
    _, fnr_r = binomtest(int(results.FN), int(results.P)).proportion_ci(confidence_level=1 - 2 * alpha)

    mu_l = norm.ppf(1 - fpr_r) - norm.ppf(fnr_r)
    if math.isnan(mu_l) or mu_l < 0:
        return 0

    return mu_l

def compute_eps_lower_gdp(results, alpha, delta):
    """Convert FPR and FNR to eps, delta using GDP at significance level alpha"""
    mu_l = compute_mu_lower_gdp(results, alpha)
    if mu_l <= 0:
        return 0

    try:
        # Step 3: convert mu-GDP to (eps, delta)-DP using Equation (6) from Tight Auditing DPML paper
        def eq6(epsilon):
            return norm.cdf(-epsilon / mu_l + mu_l / 2) - np.exp(epsilon) * norm.cdf(-epsilon / mu_l - mu_l / 2) - delta

        sol = root_scalar(eq6, bracket=[0, 50], method='brentq')
        eps_l = sol.root
    except Exception:
        eps_l = 0

    return eps_l

def compute_eps_lower_single(results, alpha, delta, method='all'):
    """Given FPR and FNR estimate epsilon lower bound using different methods at a given significance level alpha and delta
    For (eps, delta)-DP use method(s) described in https://proceedings.mlr.press/v202/zanella-beguelin23a/zanella-beguelin23a.pdf
    For GDP use method described in https://arxiv.org/pdf/2302.07956
    """
    if method == 'GDP':
        return compute_eps_lower_gdp(results, alpha, delta)
    
    method_map = {
        'zb': 'joint-beta',
        'cp': 'beta',
        'jeff': 'jeffreys'
    }

    max_eps_lo = -1
    for curr_method, curr_method_full in method_map.items():
        if method == 'all' or curr_method == method:
            try:
                curr_eps_lo = compute_eps_lo(count=results, delta=delta, alpha=alpha, method=curr_method_full)
                max_eps_lo = max(curr_eps_lo, max_eps_lo)
            except Exception:
                pass

    return max_eps_lo

def compute_roc_from_mia(scores, labels):
    """Compute threshold-wise FPR and TPR for membership inference scores."""
    scores, labels = np.array(scores), np.array(labels)
    threshs = np.sort(np.unique(scores))

    positives = labels == 1
    negatives = labels == 0
    n_pos = np.sum(positives)
    n_neg = np.sum(negatives)

    fprs = []
    tprs = []
    resultss = []
    for t in threshs:
        tp = np.sum(scores[positives] >= t)
        fp = np.sum(scores[negatives] >= t)
        fn = np.sum(scores[positives] < t)
        tn = np.sum(scores[negatives] < t)

        resultss.append((t, AttackResults(FN=fn, FP=fp, TN=tn, TP=tp)))
        fprs.append(fp / n_neg if n_neg > 0 else 0.0)
        tprs.append(tp / n_pos if n_pos > 0 else 0.0)

    return threshs, np.array(fprs), np.array(tprs), resultss

def compute_eps_lower_from_mia(scores, labels, alpha, delta, method='all', n_procs=32):
    """Compute lower bound for epsilon using privacy estimation procedure
    Step 1: For each threshold, calculate TP, FP, TN, FN and estimate epsilon lower bound using different methods at a given significance level alpha and delta
    Step 2: Output the maximum epsilon lower bound 
    """
    threshs, _, _, resultss = compute_roc_from_mia(scores, labels)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as executor, \
         tqdm(total=len(resultss), leave=False) as pbar:

        futures = {}
        for (t, curr_results) in resultss:
            futures[executor.submit(compute_eps_lower_single, curr_results, alpha, delta, method)] = t
        
        max_eps_lo, max_t = None, None
        for future in concurrent.futures.as_completed(futures):
            curr_max_eps_lo = future.result()
            t = futures[future]
            if not math.isnan(curr_max_eps_lo) and (max_eps_lo is None or curr_max_eps_lo > max_eps_lo):
                max_eps_lo = curr_max_eps_lo
                max_t = t
            pbar.update(1)
    
    return max_t, max_eps_lo

def compute_mu_lower_from_mia(scores, labels, alpha, n_procs=32):
    """Compute a lower bound for mu-GDP from MIA scores by sweeping thresholds."""
    threshs, _, _, resultss = compute_roc_from_mia(scores, labels)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as executor, \
         tqdm(total=len(resultss), leave=False) as pbar:

        futures = {}
        for (t, curr_results) in resultss:
            futures[executor.submit(compute_mu_lower_gdp, curr_results, alpha)] = t

        max_mu_lo, max_t = None, None
        for future in concurrent.futures.as_completed(futures):
            curr_max_mu_lo = future.result()
            t = futures[future]
            if not math.isnan(curr_max_mu_lo) and (max_mu_lo is None or curr_max_mu_lo > max_mu_lo):
                max_mu_lo = curr_max_mu_lo
                max_t = t
            pbar.update(1)

    return max_t, max_mu_lo

def estimate_eps(scoress, alpha=0.1, delta=0, method='all', n_procs=32):
    """Choose optimal threshold on the entire test set (e.g., GDP where choosing threshold doesn't matter)"""
    _, eps_l = compute_eps_lower_from_mia(scoress[:, 0], scoress[:, 1], alpha=alpha, delta=delta, method=method,
        n_procs=n_procs)
    return eps_l
