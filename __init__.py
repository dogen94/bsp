#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-

# Standard Library modules

# Third Party modules
import numpy as np
import math as mt
import copy 
import scipy.special as scisp
import scipy.stats as scist
import scipy.interpolate as scint
import matplotlib.pyplot as plt

# CAPE modules
import cape.attdb.rdb as rdb

"""
What do I do for the problem of hyper-rectangle volume vanishing to zero as
the dimensions increase? It sort of makes sense that the probability of any
lineload appearing vanishes to zero since there's 101 dimensions to draw
from that one specific lineload is not likely. Use the log probability? at
least helps with stability, maybe try to scale the input values to [0,1] now
we talking about copula transforms again.

could save memory at the cost of run time if we only save the best option from
each dimension? then compare the best options from each dim. That seems like a
good idea for sure.

masks probably the most at risk to explode if you have huge # of data pts.
Might have to redesign to have another array with index of points that are
contained in each domain instead of masks, probably the smarter idea tbh, that
way as you get more cuts the size of each would decrease dynamically so it
doesn't blow up at all.
"""


class GaussianKde(scist.gaussian_kde):
    # Ridge pad for covariance matrix
    EPSILON = 1e-10

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            # Get covariance
            self._data_covariance = np.atleast_2d(np.cov(self.dataset,
                rowvar=1, bias=False, aweights=self.weights))
            # Pad the diagonal of covariance with epsilon
            self._data_covariance += self.EPSILON * np.eye(
                len(self._data_covariance))
            # Re-set the inverse covariance w/o factor
            self._data_inv_cov = np.linalg.inv(self._data_covariance)
        # Re-set covariance
        self.covariance = self._data_covariance * self.factor**2
        # Re-set inverse covariance
        self.inv_cov = self._data_inv_cov / self.factor**2
        # Take cholesky decomp of cov
        L = np.linalg.cholesky(self.covariance * 2 * np.pi)
        # Re-set the normalization factor
        self._norm_factor = 2*np.log(np.diag(L)).sum()
        self.log_det = 2*np.log(np.diag(L)).sum()


class Bsp(object):
    def __init__(self, data, **kw):
        self._data = data
        # NxM (N dims, M samples)
        self.ndims, self.ndata = data.shape
        # Domain covering all data space
        domain = np.array([np.min(data, axis=1), np.max(data, axis=1)]).T
        # Initialize list of all domains
        self.domains = [domain]
        # Initialize list of # of pts in each domain
        self.counts = [self.ndata]
        # Maybe list of masks?
        self.masks = [np.ones(self.ndata, dtype=bool)]
        # Volume of domain
        self._total_volume = np.product(np.diff(domain))
        # Initialize list of volume of each domain
        self.volumes = [self._total_volume]
        # Function to evaluate partition score
        self.pscore = lambda a, b, j, n, v: -b*j + \
            (np.sum(scisp.loggamma(n + np.ones_like(n)*a)) -
             scisp.loggamma(np.sum(n + np.ones_like(n)*a))) - \
            (np.sum(scisp.loggamma(np.ones_like(n)*a)) -
             scisp.loggamma(np.sum(np.ones_like(n)*a))) - \
            np.sum(n*np.log(np.abs(v)))
        # Log conditional probability estimator
        self.c_prob = lambda n, n1, n2: n*np.log(2) + scisp.loggamma(n1) + \
                                        scisp.loggamma(n2) - scisp.loggamma(n)
        # Save first partition score for exit criteria
        self.pscore_hist = None
        # Maybe save c_prob hists?
        self.c_probs = []
        # Save domain that was just cut
        self.prev_domain_cut = None
        self.p_lim = True
        # Set some kw for searching function
        self.search_kws = {"beta": 1.0,
                        "alpha": 0.10}

    # Get probability density estimate for a point
    def __call__(self, x, cond=False, cond_dims=[], cond_x=None):

        domains = np.array(self.domains)

        # Conditional check mask
        ccheck = np.ones(len(self.domains), dtype=bool)
        if cond:
            # Check for each conditional variable
            cchecks = np.zeros((len(cond_dims), len(self.domains)),
                                dtype=bool)
            # Go through each domain
            for idom, domain in enumerate(domains):
                # Check each dimension
                for i,idim in enumerate(cond_dims):
                    # True if within bounds of idim in idom
                    cchecks[i,idom] = np.logical_and(cond_x[i] > domain[idim, 0],
                        cond_x[i] < domain[idim, 1])
            # Combine checks for each domain and see valid domains
            for icheck in range(len(cond_dims)):
                ccheck = np.logical_and(ccheck, cchecks[icheck,:])
        
        # Go through each domain
        for idom, domain in enumerate(domains[ccheck]):
            check = False
            # Check each dimension
            for idim in range(self.ndims):
                # True if within bounds of idim in idom
                check = np.logical_and(x[idim] > domain[idim, 0],
                    x[idim] < domain[idim, 1])
                # If it's not, skip the rest of the dims in domain
                if not check:
                    break
            # If check survives, give probability density est for domain
            if check:
                return (self.counts[idom]/self.volumes[idom]) * (1/self.ndata)
        # If not found in any of the domains
        return 0.0

    def partition_data(self, *a, **kw):
        # Read in kws
        search_depth = kw.pop("search_depth", 2)
        score_tol = kw.pop("score_tol", 0.05)
        cut_lim = kw.pop("cut_limit", None)
        max_search_paths = kw.pop("max_search_paths", None)
        # Set some kw for searching function
        search_kws = {"beta": 1.0,
                      "alpha": 0.25}
        while len(self.domains) < cut_lim+1:
            # Take cuts until search depth reached
            for i_depth in range(search_depth):
                # If no max search all doms
                if max_search_paths is None:
                    n_search_domains = len(self.domains)
                    # Only need search depth of 1
                    search_depth = 1
                else:
                    # If more domains than max, restrict search paths
                    if len(self.domains) > max_search_paths:
                        n_search_domains = max_search_paths
                    else:
                        # Keep searching all until too many domains
                        n_search_domains = len(self.domains)
                # Current number of cuts
                search_kws["ncuts"] = len(self.domains)
                # Cut score array for each domain
                cut_scores = np.zeros(n_search_domains)
                domain_dims = self.domains[0].shape
                # Best candidate cuts for domain
                candidate_domains = np.zeros((n_search_domains, *domain_dims))
                # Compliments of the best candidates
                compliment_domains = np.zeros((n_search_domains, *domain_dims))
                # Number of points in best candidate/compliment domain
                potential_counts = np.ones((n_search_domains, 2))
                # Volume of best candidate/compliment domain
                potential_volumes = np.ones((n_search_domains, 2))
                # Mask of best candidates
                I = np.ones((n_search_domains, self.ndata), dtype=bool)
                # Mask of compliment arrays of best candidates
                I_comp = np.ones((n_search_domains, self.ndata), dtype=bool)
                # Just choose randomly for now
                drawn_domains = np.random.choice(np.arange(len(self.domains)),
                                               n_search_domains, replace=False)
                for idraw, idom in enumerate(drawn_domains):
                    domain = self.domains[idom]
                    # Search domain for best possible cut
                    bpc = self._search_domain(idom, domain, **search_kws)
                    # Save score of "best cut"
                    cut_scores[idraw] = bpc["cut_score"]
                    # Save best domain
                    candidate_domains[idraw, :, :] = bpc["candidate_domain"]
                    # Save best domain compliment
                    compliment_domains[idraw, :, :] = bpc["compliment_domain"]
                    # Save best domain/compliment counts
                    potential_counts[idraw, :] = bpc["counts"]
                    # Save best domain/compliment volume
                    potential_volumes[idraw, :] = bpc["volumes"]
                    # Save best domain mask
                    I[idraw, :] = bpc["candidate_mask"]
                    # Save best domain compliment mask
                    I_comp[idraw, :] = bpc["compliment_mask"]
                # Get winning cut index
                wdrawn = cut_scores.argmax()
                wdom = drawn_domains[wdrawn]
                # Splice in new domains
                self.domains = self.domains[:wdom] + \
                                [candidate_domains[wdrawn, :, :],
                                compliment_domains[wdrawn, :, :]] + \
                                self.domains[wdom+1:]
                # Splice in new masks
                self.masks = self.masks[:wdom] + \
                                [I[wdrawn, :],
                                I_comp[wdrawn, :]] + self.masks[wdom+1:]
                # Splice in new counts
                self.counts = self.counts[:wdom] + \
                                [potential_counts[wdrawn, 0],
                                potential_counts[wdrawn, 1]] + \
                                self.counts[wdom+1:]
                # Splice in new volumes
                self.volumes = self.volumes[:wdom] + \
                                [potential_volumes[wdrawn, 0],
                                potential_volumes[wdrawn, 1]] + \
                                self.volumes[wdom+1:]
                # Save state at search depth 1
                if i_depth == 1:
                    domain1 = self.domains[:]
                    masks1 = self.masks[:]
                    counts1 = self.counts[:]
                    volumes1 = self.volumes[:]
                # Update score for tol
                if self.pscore_hist is None:
                    self.pscore_hist = cut_scores[wdom]
            # If search depth == 1, skip else revert back to 1
            if search_depth == 1:
                pass
            else:
                # Revert back to state after 1 step and repeat
                self.domains = domain1[:]
                # Splice in new masks
                self.masks = masks1[:]
                # Splice in new counts
                self.counts = counts1[:]
                # Splice in new volumes
                self.volumes = volumes1[:]

    def _search_domain(self, *a, **kw):
        # Extract kws
        alpha = kw.get("alpha", 0.25)
        beta = kw.get("beta", 1.0)
        n_cuts = kw.get("ncuts")
        # Get index of domain we're searching
        idom = a[0]
        # Get domain we're searching
        domain = a[1]
        # Get all other volumes and counts
        _allother_vols = [*self.volumes[:idom], *self.volumes[idom+1:]]
        _allother_counts = [*self.counts[:idom], *self.counts[idom+1:]]
        # Cut score array 
        _cut_scores = np.zeros((self.ndims))
        # Other half of the cut candidates
        _compliment_domains = np.zeros((self.ndims,
            *self.domains[0].shape))
        # Candidate cuts for domain
        _candidate_domains = np.zeros((self.ndims,
            *self.domains[0].shape))
        # Volume of each candidate/compliment domain
        _potential_volumes = np.ones((self.ndims, 2))
        # Number of points in each candidate/compliment domain
        _potential_counts = np.ones((self.ndims, 2))
        # Set stack of candidate domains to domain
        _candidate_domains[:, :, :] = [domain] * self.ndims
        # Set stack of compliment domains to domain
        _compliment_domains[:, :, :] = [domain] * self.ndims
        # Midpoints of each dimension in domain
        midpoints = np.mean(domain, axis=1)
        # Mask of candidate arrays
        _I = np.ones((self.ndims, self.ndata), dtype=bool)
        # Mask of compliment arrays
        _I_comp = np.ones((self.ndims, self.ndata), dtype=bool)
        # Set masks
        mask_stack = [self.masks[idom]] * self.ndims
        _I[:, :] = np.vstack(mask_stack)
        # Go through each dimension
        for dim in range(self.ndims):
            # Split each candidate domain along one dim
            _candidate_domains[dim, dim, 1] = midpoints[dim]
            # Keep track of other "half" of split candidate
            _compliment_domains[dim, dim, 0] = midpoints[dim]
            # Calculate candidate domain volume
            _potential_volumes[dim, 0] = np.product(np.diff(
                _candidate_domains[dim, :, :]))
            # Calcualte compliment domain volume
            _potential_volumes[dim, 1] = np.product(np.diff(
                _compliment_domains[dim, :, :]))
            # Check if within the new candidate dim bounds
            _I[dim, :] = np.logical_and(_I[dim, :],
                self._data[dim, :] < midpoints[dim])
            # Compliment count is not in candidate domain
            _I_comp[dim, :] = np.logical_not(_I[dim, :])
            # And in domain mask
            _I_comp[dim, :] = np.logical_and(_I_comp[dim, :],
                self.masks[idom])
            # Count nonzero mask in candidate domain
            _potential_counts[dim, 0] = np.count_nonzero(
                _I[dim, :])
            # Count nonzero mask in compliment domain
            _potential_counts[dim, 1] = np.count_nonzero(
                _I_comp[dim, :])
            if n_cuts > 1:
                all_counts = [*_allother_counts, *_potential_counts[dim, :]]
                all_vols = [*_allother_vols, *_potential_volumes[dim, :]]
            else:
                all_counts = _potential_counts[dim, :]
                all_vols = _potential_volumes[dim, :]
            # Score each candidate cut
            _cut_scores[dim] = self.pscore(alpha, beta, n_cuts,
                all_counts, all_vols)
            # Get index of "best" cut from this domain
        bdim = _cut_scores.argmax()
        # Build dict of results for best cut in this domain
        best_cut = {"cut_score": _cut_scores[bdim],
                    "candidate_domain": _candidate_domains[bdim, :, :],
                    "compliment_domain": _compliment_domains[bdim, :, :],
                    "counts": _potential_counts[bdim, :],
                    "volumes": _potential_volumes[bdim, :],
                    "candidate_mask": _I[bdim, :],
                    "compliment_mask": _I_comp[bdim, :]}
        # Return best cut from this domain
        return best_cut

    def partition_data_pmf(self, *a, **kw):
        # Read in kws
        # search_depth = kw.pop("search_depth", 2)
        score_tol = kw.pop("score_tol", 0.05)
        cut_lim = kw.pop("cut_limit", None)
        n_partition_samples = kw.pop("n_partition_samples", 1)
        parts_tried = 1
        while len(self.domains) < cut_lim+1 and self.p_lim:
            # Partition score array for each domain
            part_scores = np.zeros(n_partition_samples)
            # Cuts
            wcuts = np.zeros((n_partition_samples, 2), dtype=int)
            # drawn bpc
            drawn_bpcs = np.zeros((n_partition_samples), dtype=dict)
            for i_sample_part in range(n_partition_samples):
                # Current number of cuts
                self.search_kws["ncuts"] = len(self.domains)
                # Probability mass functions for each sample partition
                cut_pmfs = np.zeros((self.search_kws["ncuts"], self.ndims))
                # For each domain
                bpcs = np.zeros(len(self.domains), dtype=dict)
                for idom, domain in enumerate(self.domains):
                    # Nothing special if too few domains
                    if len(self.domains) > 2:
                        # If just cut this domain last time
                        if idom == self.prev_domain_cut or \
                           idom == self.prev_domain_cut + 1:
                           # Recalculate everything for the new domains
                            bpcs[idom] = self._search_domain_pmf(idom, domain,
                                        **self.search_kws)
                            # Save score of "best cut"
                            cut_pmfs[idom, :] = bpcs[idom]["cut_score"]
                        else:
                            # Else just use the previously calculated data
                            bpcs[idom] = None
                            # Save prob from history of "best cut"
                            cut_pmfs[idom, :] = self.c_probs[idom]
                    else:
                        # Search domain for best possible cut
                        bpcs[idom] = self._search_domain_pmf(idom, domain,
                                                             **self.search_kws)
                        # Save score of "best cut"
                        cut_pmfs[idom, :] = bpcs[idom]["cut_score"]
                # Flatten the pmfs for random draw
                flat_pmfs = cut_pmfs.flatten()
                # Randomly choose a cut
                cut_draw = np.random.choice(np.arange(len(flat_pmfs)),
                                        p=flat_pmfs/np.sum(flat_pmfs))
                # Translate the random draw back to domain/index 2d array
                dom_indx, cut_indx = np.unravel_index(cut_draw, cut_pmfs.shape)
                # Save cut draw
                wcuts[i_sample_part, :] = [dom_indx, cut_indx]
                # Save domain that was cut
                self.prev_domain_cut = wcuts[i_sample_part, 0]
                # If None bpc
                if bpcs[dom_indx] is None:
                    # Re-calc that cut if it was drawn
                    drawn_bpcs[i_sample_part] = self._cut_domain_pmf(
                                                cut_indx, dom_indx,
                                                self.domains[dom_indx],
                                                **self.search_kws)
                    # Save cut draw
                    i_dom, i_cut = [0, 0]
                    wcuts[i_sample_part, -1] = i_cut
                else:
                    # Save domain data from drawn cut
                    drawn_bpcs[i_sample_part] = bpcs[dom_indx]
                    # Save cut draw
                    i_dom, i_cut = [dom_indx, cut_indx]
                # Splice in new counts
                tmp_part_counts = self.counts[:dom_indx] + \
                                [drawn_bpcs[i_sample_part]["counts"][i_cut][0],
                                 drawn_bpcs[i_sample_part]["counts"][i_cut][1]] + \
                                 self.counts[dom_indx+1:]
                # Splice in new volumes
                tmp_part_volumes = self.volumes[:dom_indx] + \
                                [drawn_bpcs[i_sample_part]["volumes"][i_cut][0],
                                 drawn_bpcs[i_sample_part]["volumes"][i_cut][1]] + \
                                 self.volumes[dom_indx+1:]
                # Calculate partition score
                part_scores[i_sample_part] = self.pscore(
                                             self.search_kws["alpha"],
                                             self.search_kws["beta"],
                                             self.search_kws["ncuts"],
                                             tmp_part_counts,
                                             tmp_part_volumes)
            # Select best partition
            wpart = np.argmax(part_scores)
            # Get dict of best partition data
            bpc = drawn_bpcs[wpart]
            # Get domain and dimension index of cut
            wcut = wcuts[wpart]
            # Splice in new domains
            self.domains = self.domains[:wcut[0]] + \
                            [bpc["candidate_domain"][wcut[-1]][:, :],
                             bpc["compliment_domain"][wcut[-1]][:, :]] + \
                             self.domains[wcut[0]+1:]
            # Splice in new masks
            self.masks = self.masks[:wcut[0]] + \
                            [bpc["candidate_mask"][wcut[-1]][:],
                             bpc["compliment_mask"][wcut[-1]][:]] + \
                            self.masks[wcut[0]+1:]
            # Splice in new counts
            self.counts = self.counts[:wcut[0]] + \
                            [bpc["counts"][wcut[-1]][0],
                             bpc["counts"][wcut[-1]][1]] + \
                             self.counts[wcut[0]+1:]
            # Splice in new volumes
            self.volumes = self.volumes[:wcut[0]] + \
                            [bpc["volumes"][wcut[-1]][0],
                             bpc["volumes"][wcut[-1]][1]] + \
                            self.volumes[wcut[0]+1:]
            # Save partition score
            curr_pscore = part_scores[wpart]

            # Save all pmf calculations
            self.c_probs = np.zeros((cut_pmfs.shape[0] + 1, cut_pmfs.shape[-1]))
            self.c_probs[:wcut[0], :] = cut_pmfs[:wcut[0], :]
            self.c_probs[wcut[0]:wcut[0]+2] = None
            self.c_probs[wcut[0]+2:] = cut_pmfs[wcut[0]+1:, :]
            # Update score
            if self.pscore_hist is None:
                self.pscore_hist = [curr_pscore]
            elif curr_pscore > np.max(self.pscore_hist[-10:]):
                parts_tried = 1
                self.pscore_hist.extend([curr_pscore])
            elif parts_tried < 10:
                parts_tried += 1
            else:
                self.p_lim = False

    def _search_domain_pmf(self, *a, **kw):
        # Extract kws
        alpha = kw.get("alpha", 0.25)
        beta = kw.get("beta", 1.0)
        n_cuts = kw.get("ncuts")
        # Get index of domain we're searching
        idom = a[0]
        # Get domain we're searching
        domain = a[1]
        # Cut score array 
        _cut_scores = np.zeros((self.ndims))
        # Other half of the cut candidates
        _compliment_domains = np.zeros((self.ndims,
            *self.domains[0].shape))
        # Candidate cuts for domain
        _candidate_domains = np.zeros((self.ndims,
            *self.domains[0].shape))
        # Volume of each candidate/compliment domain
        _potential_volumes = np.ones((self.ndims, 2))
        # Number of points in each candidate/compliment domain
        _potential_counts = np.ones((self.ndims, 2))
        # Set stack of candidate domains to domain
        _candidate_domains[:, :, :] = [domain] * self.ndims
        # Set stack of compliment domains to domain
        _compliment_domains[:, :, :] = [domain] * self.ndims
        # Midpoints of each dimension in domain
        midpoints = np.mean(domain, axis=1)
        # Mask of candidate arrays
        _I = np.ones((self.ndims, self.ndata), dtype=bool)
        # Mask of compliment arrays
        _I_comp = np.ones((self.ndims, self.ndata), dtype=bool)
        # Set masks
        mask_stack = [self.masks[idom]] * self.ndims
        _I[:, :] = np.vstack(mask_stack)
        # Go through each dimension
        for dim in range(self.ndims):
            # Split each candidate domain along one dim
            _candidate_domains[dim, dim, 1] = midpoints[dim]
            # Keep track of other "half" of split candidate
            _compliment_domains[dim, dim, 0] = midpoints[dim]
            # Calculate candidate domain volume
            _potential_volumes[dim, 0] = np.product(np.diff(
                _candidate_domains[dim, :, :]))
            # Calcualte compliment domain volume
            _potential_volumes[dim, 1] = np.product(np.diff(
                _compliment_domains[dim, :, :]))
            # Check if within the new candidate dim bounds
            _I[dim, :] = np.logical_and(_I[dim, :],
                self._data[dim, :] < midpoints[dim])
            # Compliment count is not in candidate domain
            _I_comp[dim, :] = np.logical_not(_I[dim, :])
            # And in domain mask
            _I_comp[dim, :] = np.logical_and(_I_comp[dim, :],
                self.masks[idom])
            # Count nonzero mask in candidate domain
            _potential_counts[dim, 0] = np.count_nonzero(
                _I[dim, :])
            # Count nonzero mask in compliment domain
            _potential_counts[dim, 1] = np.count_nonzero(
                _I_comp[dim, :])
            # Set pmf
            _cut_scores[dim] = abs(self.c_prob(sum(_potential_counts[dim, :]),
                         _potential_counts[dim, 0], _potential_counts[dim, 1]))
            # Handle edge cases of pmf calculation
            if sum(_potential_counts[dim, :]) == 0.0:
                _cut_scores[dim] = 0.0
            if np.isinf(_cut_scores[dim]):
                _cut_scores[dim] = 2e1
        # Build dict of results for best cut in this domain
        best_cut = {"cut_score": _cut_scores,
                    "candidate_domain": _candidate_domains,
                    "compliment_domain": _compliment_domains,
                    "counts": _potential_counts,
                    "volumes": _potential_volumes,
                    "candidate_mask": _I,
                    "compliment_mask": _I_comp}
        # Return best cut from this domain
        return best_cut


    def _cut_domain_pmf(self, idim, *a, **kw):
        # Extract kws
        alpha = kw.get("alpha", 0.25)
        beta = kw.get("beta", 1.0)
        n_cuts = kw.get("ncuts")
        # Get index of domain we're searching
        idom = a[0]
        # Get domain we're searching
        domain = a[1]
        # Cut score array 
        _cut_scores = np.zeros((1))
        # Other half of the cut candidates
        _compliment_domains = np.zeros((1, *self.domains[0].shape))
        # Candidate cuts for domain
        _candidate_domains = np.zeros((1, *self.domains[0].shape))
        # Volume of each candidate/compliment domain
        _potential_volumes = np.ones((1, 2))
        # Number of points in each candidate/compliment domain
        _potential_counts = np.ones((1, 2))
        # Set stack of candidate domains to domain
        _candidate_domains[:, :] = domain
        # Set stack of compliment domains to domain
        _compliment_domains[:, :] = domain
        # Midpoints of each dimension in domain
        midpoints = np.mean(domain, axis=1)
        # Mask of candidate arrays
        _I = np.ones((1, self.ndata), dtype=bool)
        # Mask of compliment arrays
        _I_comp = np.ones((1, self.ndata), dtype=bool)
        # Set masks
        mask_stack = [self.masks[idom]] 
        _I[:, :] = np.vstack(mask_stack)
        # Split each candidate domain along one dim
        _candidate_domains[0, idim, 1] = midpoints[idim]
        # Keep track of other "half" of split candidate
        _compliment_domains[0, idim, 0] = midpoints[idim]
        # Calculate candidate domain volume
        _potential_volumes[0, 0] = np.product(np.diff(
            _candidate_domains[:, :]))
        # Calcualte compliment domain volume
        _potential_volumes[0, 1] = np.product(np.diff(
            _compliment_domains[:, :]))
        # Check if within the new candidate dim bounds
        _I[0, :] = np.logical_and(_I[0, :],
            self._data[idim, :] < midpoints[idim])
        # Compliment count is not in candidate domain
        _I_comp[0, :] = np.logical_not(_I[0, :])
        # And in domain mask
        _I_comp[0, :] = np.logical_and(_I_comp[0, :],
            self.masks[idom])
        # Count nonzero mask in candidate domain
        _potential_counts[0, 0] = np.count_nonzero(
            _I[0, :])
        # Count nonzero mask in compliment domain
        _potential_counts[0, 1] = np.count_nonzero(
            _I_comp[0, :])
        # Set pmf
        _cut_scores[0] = abs(self.c_prob(sum(_potential_counts[0, :]),
                        _potential_counts[0, 0], _potential_counts[0, 1]))
        # Handle edge cases of pmf calculation
        if sum(_potential_counts[0, :]) == 0.0:
            _cut_scores[0] = 0.01
        if np.isinf(_cut_scores[0]):
            _cut_scores[0] = 2e1
        # Build dict of results for best cut in this domain
        best_cut = {"cut_score": _cut_scores,
                    "candidate_domain": _candidate_domains,
                    "compliment_domain": _compliment_domains,
                    "counts": _potential_counts,
                    "volumes": _potential_volumes,
                    "candidate_mask": _I,
                    "compliment_mask": _I_comp}
        # Return best cut from this domain
        return best_cut



    def sample(self, N):
        # Set probability of each domain based on fraction of pts contained
        p = self.counts/sum(self.counts)
        # Draw from domains based on fraction of points contained
        domain_draws = np.random.choice(np.arange(len(self.counts)), size=N,
                                        p=p)

        # For values to be drawn
        u_draws = np.zeros((self.ndims, N))
        # For each domain drawn
        for i, domain_draw in enumerate(domain_draws):
            # Use domain bounds and take uniform dist over the domain
            u_draws[:, i] = np.random.uniform(self.domains[domain_draw][:, 0],
                                              self.domains[domain_draw][:, 1])
        return u_draws, domain_draws

    def sample_gkde(self, N, **kw):
        # Get provided bw
        bw = kw.pop("bw", 0.1)
        # Set probability of each domain based on fraction of pts contained
        p = self.counts/sum(self.counts)
        # Draw from domains based on fraction of points contained
        domain_draws = np.random.choice(np.arange(len(self.counts)), size=N,
                                        p=p)
        # For values to be drawn
        u_draws = np.zeros((self.ndims, N))
        COV = np.cov(self._data)
        # For each domain drawn
        for i, domain_draw in enumerate(domain_draws):
            I = self.masks[domain_draw]
            I = I.astype(bool)
            data = self._data[:,I]
            ref_ind = np.random.choice(np.arange(data.shape[-1]), size=1)[0]
            ref_data = data[:,ref_ind]
            perturb = scist.multivariate_normal(
                            np.zeros(self._data[:, I].shape[0]),
                                COV*bw**2.0).rvs(1)
            u_draws[:, i] = np.ma.clip(ref_data + perturb,
                                        a_min=self.domains[domain_draw][:,0],
                                        a_max=self.domains[domain_draw][:,1])
            
            # Truncate draws to [0,1]
            

        return u_draws, domain_draws

    def sample_studentst(self, N):
        # Set probability of each domain based on fraction of pts contained
        p = self.counts/sum(self.counts)
        # Draw from domains based on fraction of points contained
        domain_draws = np.random.choice(np.arange(len(self.counts)), size=N,
                                        p=p)
        # For values to be drawn
        u_draws = np.zeros((self.ndims, N))
        for i, domain_draw in enumerate(domain_draws):
            I = self.masks[domain_draw]
            I = I.astype(bool)
            if np.count_nonzero(I) == 1:
                u_draws[:, i] = self._data[:, I][:, 0] + \
                    np.random.normal(np.zeros_like(self._data[:, I]),
                                     np.ones((len(self._data[:, I]),
                                              len(self._data[:, I])))
                                     * 1e-10)[:, 0]
            else:
                I_true = np.where(I == True)[0]
                count = np.count_nonzero(I)
                cov = np.cov(self._data[:, I]) + np.eye(101)*1e-10
                draw = np.random.choice(I_true, size=1)[0]
                normdraw = np.random.multivariate_normal(np.zeros(self.ndims),
                            cov * 0.1, size=1).T

                # chidraw = np.random.chisquare(999, size=np.shape(normdraw))

                u_draws[:, i] = normdraw[:, 0]
                u_draws[:, i] += self._data[:, draw]
        return u_draws

    def sample_marginalized(self, N, x, xdim, **kw):
        # Find domains containing x in dimension xdim
        domain_mask = self.marginalize_x(x, xdim)
        # Array of index for all possible domains
        domain_inds = np.arange(len(self.counts))
        # Mask array of all index
        domain_inds = domain_inds[domain_mask]
        # Set probability of each domain based on fraction of pts contained
        p = self.counts/sum(self.counts)
        # Mask p
        p = p[domain_mask] / sum(p[domain_mask])
        # Draw from domains based on fraction of points contained
        domain_draws = np.random.choice(domain_inds, size=N, p=p)
        # For values to be drawn
        u_draws = np.zeros((self.ndims, N))
        # For each domain drawn
        for i, domain_draw in enumerate(domain_draws):
            # Use domain bounds and take uniform dist over the domain
            u_draws[:, i] = np.random.uniform(self.domains[domain_draw][:, 0],
                                              self.domains[domain_draw][:, 1])
        return u_draws

    def sample_gkde_marginalized(self, N, x, xdim, **kw):
        # Get provided bw
        bw = kw.pop("bw", 0.25)
        # Find domains containing x in dimension xdim
        domain_mask = self.marginalize_x(x, xdim)
        # Array of index for all possible domains
        domain_inds = np.arange(len(self.counts))
        # Mask array of all index
        domain_inds = domain_inds[domain_mask]
        # Set probability of each domain based on fraction of pts contained
        p = self.counts/sum(self.counts)
        # Mask p
        p = p[domain_mask] / sum(p[domain_mask])
        # Draw from domains based on fraction of points contained
        domain_draws = np.random.choice(domain_inds, size=N, p=p)
        # For values to be drawn
        u_draws = np.zeros((self.ndims, N))
        # For each domain drawn
        for i, domain_draw in enumerate(domain_draws):
            I = self.masks[domain_draw]
            I = I.astype(bool)
            if np.count_nonzero(I) == 1:
                u_draws[:, i] = self._data[:,I][:,0] + \
                    np.random.normal(np.zeros_like(self._data[:, I]),
                                     np.ones((len(self._data[:, I]),
                                              len(self._data[:, I])))
                                     * 1e-10)[:, 0]
            else:
                pdf = GaussianKde(self._data[:, I], bw_method=bw)
                # Use domain bounds and take uniform dist over the domain
                u_draws[:, i] = pdf.resample(1)[:, 0]
        return u_draws

    def marginalize_x(self, x, xdim):
        # Find domains containing x in dimension xdim
        domain_mask = np.zeros(len(self.domains), dtype=bool)
        # Go through each domain
        for idom, domain in enumerate(self.domains):
            check = False
            # True if within bounds of idim in idom
            check = np.logical_and(x > domain[xdim, 0],
                                   x < domain[xdim, 1])
            # If check survives, give probability density est for domain
            if check:
                domain_mask[idom] = True
        return domain_mask

    def make_midpoints(self):
        # Midpoints of all domains
        midpoints = np.zeros((len(self.domains), self.ndims))
        # 
        for i,domain in enumerate(self.domains):
            midpoints[i, :] = np.diff(domain)[:, 0]
        self.midpoints = midpoints

    # Plot a slice
    def plot_domain_slice(self, dims, filter=False):
        h = plt.figure(figsize=(5.5, 4.4)) 
        # Scatter of all data
        plt.scatter(self._data[dims[0], :], self._data[dims[1], :], s=3.0,
            c="r", alpha=0.15)
        if filter:
            I = np.where(np.array(self.counts) > 0.0)[0]
        else:
            I = np.arange(len(self.counts))
        domains = np.array(self.domains)
        # Plot boxes of all domains
        for domain in domains[I]:
            X1, X2 = np.meshgrid(domain[dims[0], :], domain[dims[1], :])
            plt.plot(X1, X2, c="k")
            plt.plot(X1.T, X2.T, c="k")
        # return plot
        return h

    # Write a mat file of partitions
    def write_mat(self, fname):
        # Read mat into db
        db = rdb.DataKit()
        cols = ["domains", "counts", "masks", "volumes",
                "pscore_hist", "c_probs", "prev_domain_cut"]
        # For each needed column
        for col in cols:
            # Save col to blank db
            db.save_col(col, self.__getattribute__(col))
        # Write mat file of db
        db.write_mat(fname)

    # Read from a mat file of partitions
    def read_mat(self, fname, run_prep=False):
        # Read mat into db
        db = rdb.DataKit(fname)
        cols = ["domains", "counts", "masks", "volumes",
                "pscore_hist", "c_probs", "prev_domain_cut"]
        # For each needed column
        for col in cols:                
            value = db.get(col, [])
            if col == "prev_domain_cut":
                self.__setattr__(col, value)
            else:
                # Read in list versions
                self.__setattr__(col, [*value])
        if run_prep:
            # Set prev_domain_cut to out of range, so recalc everything
            if self.prev_domain_cut == []:
                self.prev_domain_cut = int(len(self.domains) + 2)
            # Load probabilities if not saved
            if self.c_probs == []:
                self.c_probs = np.zeros((len(self.domains), self.ndims))
                # Current number of cuts
                self.search_kws["ncuts"] = len(self.domains)
                for idom, domain in enumerate(self.domains):
                    bpc = self._search_domain_pmf(idom, domain,
                                                **self.search_kws)
                    # Save score of "best cut"
                    self.c_probs[idom, :] = bpc["cut_score"]


def auto_sis_bsp(data, cut_limit: int, n_partitions: int = 200,
                 n_partition_resample: int = 10, bsp = None) -> Bsp:
    
    # Initialize array of bsps
    bsps = np.zeros(n_partitions, dtype=Bsp)
    if bsp is None:
        for i_bsp in range(n_partitions):
            bsps[i_bsp] = Bsp(data)
            ncuts = 0
    else:
        for i_bsp in range(n_partitions):
            # Copy the given partion
            bsps[i_bsp] = copy.deepcopy(bsp)
            # Get number of cuts already made
            ncuts = len(bsp.domains) - 1
    npr = n_partition_resample
    # Remainder amount of cuts after split by npr
    modcut = cut_limit % npr
    # If remainder do this
    if modcut > 0:
        # Break down cut limit into resampling chunks
        cut_segments = np.arange(npr, cut_limit - cut_limit % npr + npr, npr)
        # Add mod remainder to end of segments
        cut_segments = np.append(cut_segments, cut_segments[-1] + modcut)
    else:
        # Break down cut limit into resampling chunks
        cut_segments = np.arange(npr, cut_limit + npr, npr)
    # Only look at the cut_segments after cuts already made
    cut_segments = cut_segments[np.where(cut_segments > ncuts)[0]]
    # For each cutting segment
    for cut_seg in cut_segments:
        # Initialize list of pscores
        pscores = np.zeros_like(bsps, dtype=float)
        # For each bsp sample
        for i_bsp in range(n_partitions):
            # Cut segment amount into this bsp
            bsps[i_bsp].partition_data_pmf(cut_limit=cut_seg)
            # Get score of this bsp
            pscores[i_bsp] = bsps[i_bsp].pscore_hist[-1]
        # Index of the best scoring partition
        i_best_bsp = np.argmax(pscores)
        # If last cut segment, just return all bsps
        if cut_seg == cut_segments[-1] or not bsps[i_best_bsp].p_lim:
            print("Did partitions require all cuts: ", bsps[i_best_bsp].p_lim)
            return i_best_bsp, bsps
        # All partition samples replaced by copy of the best one
        for i_bsp in range(n_partitions):
            bsps[i_bsp] = copy.deepcopy(bsps[i_best_bsp])


def inv_UQcdf_skew(data, UQ, p, x, N, interp):
    # If within [0,1] cdf bounds use this
    if x >= 0.0 and x <= 1.0:
        # Get index of "mean"
        pmin = np.where(p <= x)[0][-1]
        if pmin >= len(data) - 1:
            return(data[-1])
        # Linear estimate of mean
        data_mean = interp(x)
        # data_mean = ((p[pmin+1]-x) / (p[pmin+1] - p[pmin]))*(
        #     data[pmin+1] - data[pmin]) + data[pmin]

        # Assume 3sig truncated Gaussian "Marginal" box cdf
        y = scist.truncnorm.rvs(-1.0, 1.0, loc=data_mean,
                                scale=UQ, size=N)
        # y = data_mean
    # If not, get wild
    elif x < 0.0:
        # Linear extrapolation factor
        # lin_factor = (p[0]-x) / (p[0] - p[1])
        # Linear estimate of mean
        data_mean = interp(x)
        # data_mean = lin_factor*(data[0]) + data[0]
        if data_mean < data[0] - (1*UQ):
            y = data[0] - 3*UQ
        else:
            lower_bnd = (data[0] - data_mean - 1*UQ)/UQ
            # Draw using fixed min value
            y = scist.truncnorm.rvs(lower_bnd,
                                    1.0,
                                    loc=data_mean,
                                    scale=UQ, size=N)
            # y = data_mean

    # If not, get crazy
    elif x > 1.0:
        # Linear extrapolation factor
        # lin_factor = (x - p[-1]) / (p[-1] - p[-2])
        # Linear estimate of mean
        data_mean = interp(x)
        # data_mean = lin_factor*(data[-1]) + data[-1]
        if data_mean > data[-1] + (1*UQ):
            y = data[-1] + (1*UQ)
        else:
            # upper_bnd = data[0] + 3UQ - data_mean = A Uq
            upper_bnd = (data[-1] - data_mean + 1*UQ)/UQ
            # Draw using fixed max value
            y = scist.truncnorm.rvs(-1.0,
                                    upper_bnd, loc=data_mean,
                                    scale=UQ, size=N)
            # y = data_mean

    return(y)


def inv_cdf_rbfs(hist, data, n_domain, x):
    """Estimate inverse CDF of x with local rbfs

    :Input:
        *hist*: ``bsp.Bsp``
            bsp partition of data copula
        *data*: ``np.array``
            Native domain data equivalent of copula data
        *n_domain*: ``int``
            Index of *hist* domain containing *x*
        *x*: ``np.array``
            Data point to transform with inverse cdf
    :Output:
        *rbf(x)*: ``np.array``
            Data point to transformed with inverse cdf
    """
    # Get mask of pts in domain number n
    I_domain = hist.masks[n_domain]
    # Index of points in domain
    I_domain = np.where(I_domain == 1)[0]
    # Check if too few points for rbf
    if len(I_domain) < hist.ndims + 1:
        # See if midpoints already calc
        midpoints = getattr(hist, "midpoints", False)
        # If not, calc them
        if not midpoints:
            hist.make_midpoints()
            # Save midpoints
            midpoints = hist.midpoints
        # Find 1-norm dist to n_domain
        dists = np.zeros(len(hist.domains))
        dists = np.repeat(x, len(hist.domains)).reshape(len(hist.domains), 
                                                        len(x))
        dists = abs(dists - midpoints)
        # Sum 1-norm from all dims
        total_dists = np.sum(dists, axis=1)
        # Index of closest domains to x
        closest_doms = np.argsort(total_dists)
        # Count nubmer of points currently
        count_I = np.count_nonzero(I_domain)
        # Continue to take closest domains until reach min number of pts
        for dom in closest_doms:
            # Get mask of next closest domain
            tmp_I = hist.masks[dom]
            # Save to main mask
            I_domain = np.logical_or(I_domain, tmp_I)
            # Count main mask points
            count_I += np.count_nonzero(I_domain)
            # Break if enough points
            if count_I > hist.ndims + 1:
                break
    # Native domain data of closest points
    nat_d = data[:, I_domain]
    # CDF domain data of closest points
    cdf_d = hist._data[:, I_domain]
    # Make rbf from cdf to native space, thin plate spline w/ lin. poly
    rbf = scint.RBFInterpolator(cdf_d.T, nat_d.T, degree=1)
    # Do dumb dimension adjustment for x
    test = np.zeros((1,len(x)))
    test[0,:] = x
    # Return inv cdf of x
    return rbf(test)


def local_outlier_factor(data, gen_data=None, k=10):
    """Estimate LOF for data

    :Input:
        *data*: ``np.array``
            Data to calculate the LOF of
        *gen_data*: {None} | ``np.array``
            Extra data to use in LOF calculation
        *k*: {10} | ``int``
            Number of nearest points to use in LOF calculation
    :Output:
        *lof*: ``np.array``
            Array of LOFs for each *data* point
        *Nscaled_lof*: ``np.array``
            Array of LOFs statistically normalized
    """
    # See if extra data is included
    if gen_data is None:
        all_data = data
        nadim, nadata = np.shape(all_data)
    else:
        # Concatenate extra data
        all_data = np.hstack([data, gen_data])
        nadim, nadata = np.shape(all_data)
    # Shape of data
    ndim, ndata = np.shape(data)
    # Matrix of kth nearest points to each point
    k_nearest = np.zeros((nadata, k), dtype=int)
    # Distance to kth nearest points
    k_dists = np.zeros((nadata, 1))
    # Density of kth nearest pnts
    lrds = np.zeros((nadata, 1))
    # For each data point
    for i,d in enumerate(all_data.T):
        # Distance to every other pt
        dists = np.sum(np.sqrt((np.repeat(d, nadata).reshape(ndim, nadata)
                                - all_data)**2.0), axis=0)
        # Sort distances
        i_dists_sort = np.argsort(dists)
        # Take kth nearest point indexes
        k_nearest[i,:] = i_dists_sort[1:k+1]
        # Save kth nearest point distances
        k_dists[i] = dists[i_dists_sort[k+1]]
        # Density of kth nearest points
        lrds[i] = np.mean(1/dists[k_nearest[i, :]])
    # LOF
    lof = np.zeros(ndata)
    # For each point in data, calc local outlier factor
    for i in range(ndata):
        lof[i] = np.mean(lrds[k_nearest]) / lrds[i]
    # Init matrix for scaled LOFs
    Nscaled_lof = np.zeros_like(lof)
    # Mean of LOFs
    mu_lof = np.mean(lof)
    # STD of LOFs
    sigma_lof = np.std(lof)
    # For each data point in data, calc gaussian normalized LOF
    for i in range(ndata):
        Nscaled_lof[i] = max(0,
                         mt.erf((lof[i] - mu_lof) / (sigma_lof * np.sqrt(2))))
    # Return LOF and normalized LOF
    return(lof, Nscaled_lof)
