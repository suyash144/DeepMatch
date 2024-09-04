
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist

class _MaskedLoss(torch.nn.Module):
    def forward(self, estimate, output, mask=None):
        feature_mask = mask.expand_as(estimate)
        return self._loss(estimate[feature_mask], output[feature_mask])

class L1Loss(_MaskedLoss):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.L1Loss()

class L2Loss(_MaskedLoss):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.MSELoss()

class AELoss(nn.Module):
    def __init__(self, lambda1, lambda2):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        l1_loss = self.l1_loss(outputs, targets)
        l2_loss = self.l2_loss(outputs, targets)
        combined_loss = self.lambda1 * l1_loss + self.lambda2 * l2_loss
        return combined_loss

class ClipLoss1D(torch.nn.Module):
    """CLIP (See Open AI CLIP) contrastive loss."""

    def __init__(self, linear=None, twin=True, center=False, temp_tau=1.0):
        super().__init__()
        self.linear = None
        self.center = center
        if linear is not None:
            self.linear_est = torch.nn.LazyLinear(linear)
            if twin:
                self.linear_gt = self.linear_est
            else:
                self.linear_gt = torch.nn.LazyLinear(linear)
        self.temp_tau = nn.Parameter(torch.tensor(temp_tau))

    def get_scores(self, estimates: torch.Tensor, candidates: torch.Tensor):
        """Given estimates that is [B, N] and candidates which is [B', N],
        return a [B, B'] matrix of scores of matching."""
        if self.linear:
            estimates = self.linear_est(estimates)
            candidates = self.linear_gt(candidates)
        if self.center:
            estimates = estimates - estimates.mean(dim=1, keepdim=True)
            candidates = candidates - candidates.mean(dim=1, keepdim=True)

        inv_norms = 1 / (1e-8 + candidates.norm(dim=1, p=2))
        inv_norms_2 = 1 / (1e-8 + estimates.norm(dim=1, p=2))
        # scores = torch.einsum("bn,on,o->bo", estimates, candidates, inv_norms)
        # scores = torch.einsum("bn,bn->b", estimates, candidates)
        scores = torch.einsum("bn,on,b,o -> bo", estimates, candidates, inv_norms_2, inv_norms)
        return scores

    def get_probabilities(self, estimates, candidates):
        """Given estimates that is [B, N] and candidates which is [B', N],
        return a [B, B'] matrix of probabilities of matching."""
        scores = self.get_scores(estimates, candidates)
        scores = scores / self.temp_tau
        return F.softmax(scores, dim=1)

    def forward(self, estimate, candidate):
        """Forward method for ClipLoss."""
        assert estimate.size(0) <= candidate.size(0), "need at least as many targets as estimates"
        scores = self.get_probabilities(estimate, candidate)
        target = torch.arange(len(scores), device=estimate.device)
        return F.cross_entropy(scores, target)

class ClipLoss2D(torch.nn.Module):
    """CLIP (See Open AI CLIP) constrastive loss.
    """
    def __init__(self, linear=None, twin=True, pool=False, 
                 center=False, temp_tau= 1.0):
        super().__init__()
        self.linear = None
        self.pool = pool
        self.center = center
        if linear is not None:
            self.linear_est = torch.nn.LazyLinear(linear)
            if twin:
                self.linear_gt = self.linear_est
            else:
                self.linear_gt = torch.nn.LazyLinear(linear)
        self.temp_tau = nn.Parameter(torch.tensor(temp_tau))

    def get_scores(self, estimates: torch.Tensor, candidates: torch.Tensor):
        """Given estimates that is [B, C, T] and candidates
        which is [B', C, T], return a [B, B'] matrix of scores of matching.
        """
        if self.linear:
            estimates = self.linear_est(estimates)
            candidates = self.linear_gt(candidates)
        if self.pool:
            estimates = estimates.mean(dim=2, keepdim=True)
            candidates = candidates.mean(dim=2, keepdim=True)
        if self.center:
            estimates = estimates - estimates.mean(dim=(1, 2), keepdim=True)
            candidates = candidates - candidates.mean(dim=(1, 2), keepdim=True)

        inv_norms = 1 / (1e-8 + candidates.norm(dim=(1, 2), p=2))
        inv_norms_2 = 1 / (1e-8 + estimates.norm(dim=(1, 2), p=2))  
        # scores = torch.einsum("bct,oct,o->bo", estimates, candidates, inv_norms)
        # scores = torch.einsum("bct,bct->bc", estimates, candidates)
        scores = torch.einsum("bct,oct,b,o -> bo", estimates, candidates, inv_norms_2, inv_norms)
        return scores
    
    def get_probabilities(self, estimates, candidates):
        """Given estimates that is [B, C, T] and candidates
        which is [B', C, T], return a [B, B'] matrix of probabilities of matching.
        """
        scores = self.get_scores(estimates, candidates)
        scores = scores / self.temp_tau
        return F.softmax(scores, dim=1)

    def forward(self, estimate, candidate, mask=None):
        """Warning: estimate and candidate are not necessarily symmetrical.
        If estimate of shape [B, C, T] and candidate of size [B', C, T]
        with B'>=B, the first B samples of candidate are targets, while
        the remaining B'-B samples of candidate are only used as negatives.
        """
        # assert mask.all(), "mask is not supported for now"
        assert estimate.size(0) <= candidate.size(0), "need at least as many targets as estimates"
        scores = self.get_probabilities(estimate, candidate)
        target = torch.arange(len(scores), device=estimate.device)
        return F.cross_entropy(scores, target)


class CustomClipLoss(torch.nn.Module):
    """Modified CLIP contrastive loss with weights for positive and negative samples."""
    def __init__(self, linear=None, twin=True, center=False, temp_tau=1.0,negative_weight=10.):
        super().__init__()
        self.linear = None
        self.center = center
        self.negative_weight = negative_weight
        if linear is not None:
            self.linear_est = torch.nn.LazyLinear(linear)
            if twin:
                self.linear_gt = self.linear_est
            else:
                self.linear_gt = torch.nn.LazyLinear(linear)
        self.temp_tau = nn.Parameter(torch.tensor(temp_tau))

    def get_scores(self, estimates: torch.Tensor, candidates: torch.Tensor):
        """Given estimates that is [B, N] and candidates which is [B', N],
        return a [B, B'] matrix of scores of matching."""
        if self.linear:
            estimates = self.linear_est(estimates)
            candidates = self.linear_gt(candidates)
        if self.center:
            estimates = estimates - estimates.mean(dim=1, keepdim=True)
            candidates = candidates - candidates.mean(dim=1, keepdim=True)

        inv_norms = 1 / (1e-8 + candidates.norm(dim=1, p=2))
        inv_norms_2 = 1 / (1e-8 + estimates.norm(dim=1, p=2))
        # scores = torch.einsum("bn,on,o->bo", estimates, candidates, inv_norms)
        # scores = torch.einsum("bn,bn->b", estimates, candidates)
        scores = torch.einsum("bn,on,b,o -> bo", estimates, candidates, inv_norms_2, inv_norms)
        return scores

    def get_probabilities(self, estimates, candidates):
        """Given estimates that is [B, N] and candidates which is [B', N],
        return a [B, B'] matrix of probabilities of matching."""
        scores = self.get_scores(estimates, candidates)
        scores = scores / self.temp_tau
        return F.softmax(scores, dim=1)

    def forward(self, estimate, candidate):
        """Forward method for ClipLoss."""
        assert estimate.size(0) <= candidate.size(0), "need at least as many targets as estimates"
        scores = self.get_probabilities(estimate, candidate)
        # Initialize the weight tensor with ones for all elements
        weight_tensor = torch.ones_like(scores)
        mask = ~torch.eye(scores.size(0), scores.size(1), dtype=torch.bool, device=scores.device)
        # Set the off-diagonal elements (negative pairs) to the desired higher weight
        weight_tensor[mask] = self.negative_weight  # Increase the weight for negative pairs
        weighted_scores = scores * weight_tensor
        target = torch.arange(len(scores), device=estimate.device)
        return F.cross_entropy(weighted_scores, target)


def clip_prob(estimates, candidates, temp_tau=1.0):
    inv_norms = 1 / (1e-8 + candidates.norm(dim=1, p=2))
    inv_norms_2 = 1 / (1e-8 + estimates.norm(dim=1, p=2))
    # scores = torch.einsum("bn,on,o->bo", estimates, candidates, inv_norms)
    # scores = torch.einsum("bn,bn->b", estimates, candidates)
    scores = torch.einsum("bn,on,b,o -> bo", estimates, candidates, inv_norms_2, inv_norms)
    scores = scores / temp_tau
    return F.softmax(scores, dim=1)

def clip_sim(estimates, candidates):
    inv_norms = 1 / (1e-8 + candidates.norm(dim=1, p=2))
    inv_norms_2 = 1 / (1e-8 + estimates.norm(dim=1, p=2))
    # scores = torch.einsum("bn,on,o->bo", estimates, candidates, inv_norms)
    # scores = torch.einsum("bn,bn->b", estimates, candidates)
    scores = torch.einsum("bn,on,b,o -> bo", estimates, candidates, inv_norms_2, inv_norms)
    return scores

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class VICReg(nn.Module):
    def __init__(self, sim_coeff=1.0, std_coeff=1.0, cov_coeff=0.05):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y):
        bsz, num_features = x.shape

        if torch.isnan(x).any() or torch.isnan(y).any():
            print("NaN detected in input tensors")

        repr_loss = F.mse_loss(x, y)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.01)
        std_y = torch.sqrt(y.var(dim=0) + 0.01)
        
        if torch.isnan(std_x).any() or torch.isnan(std_y).any():
            print("NaN detected in standard deviation")
            print(torch.max(x), torch.min(x), torch.max(y), torch.min(y))

        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (bsz - 1)
        cov_y = (y.T @ y) / (bsz - 1)
        if torch.isnan(cov_x).any() or torch.isnan(cov_y).any():
            print("NaN detected in covariance matrix")

        # note cov_loss, to see whether cov make things wrong
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features) + off_diagonal(cov_y).pow_(2).sum().div(num_features)

        if torch.isnan(repr_loss) or torch.isnan(std_loss) or torch.isnan(cov_loss):
            print(f"NaN detected in losses: repr_loss={repr_loss}, std_loss={std_loss}, cov_loss={cov_loss}")

        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss


# Projector
class Projector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden_layers, dropout):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projector(x)