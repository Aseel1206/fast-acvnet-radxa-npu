"""NPU-friendly inference variant of Fast_ACVNet_plus (export only).

Replaces, with mathematically equivalent ops:
  * in-place slice assignment in the correlation volume  -> pad + stack
  * sort/topk + gather (attention top-k, k=4 of D)       -> rank mask + cumsum + masked sums
  * regression_topk (top-2 of 4, softmax, weighted sum)  -> masked softmax
so the graph has no ScatterND/ScatterElements/TopK/GatherElements, which the
Allwinner NPU compiler rejects or runs slowly.
Ties are broken by lower index, like a stable sort.
"""
import torch
import torch.nn.functional as F
import models.Fast_ACV_plus as fap
from models.Fast_ACV_plus import Fast_ACVNet_plus
from models.submodule import norm_correlation, context_upsample


def corr_volume_npu(ref, tgt, maxdisp):
    vols = []
    for i in range(maxdisp):
        if i > 0:
            c = F.pad(norm_correlation(ref[:, :, :, i:], tgt[:, :, :, :-i]), (i, 0))
        else:
            c = norm_correlation(ref, tgt)
        vols.append(c)
    return torch.stack(vols, dim=2)  # (B,1,D,H,W)


def topk_mask(p, dim, k):
    """1.0 where p is among the top-k along `dim` (ties -> lower index wins)."""
    D = p.shape[dim]
    shape = [1] * p.dim(); shape[dim] = D
    idx = torch.arange(D, dtype=p.dtype, device=p.device).view(shape)
    rank = torch.zeros_like(p)
    for j in range(D):
        pj = p.narrow(dim, j, 1)
        rank = rank + (pj > p).to(p.dtype) + (pj == p).to(p.dtype) * (idx > j).to(p.dtype)
    return (rank < (k - 0.5)).to(p.dtype)


class Fast_ACVNet_plus_npu(Fast_ACVNet_plus):
    def forward(self, left, right):
        k = getattr(self, "topk", 4)
        features_left = self.feature(left)
        features_right = self.feature(right)
        features_left, features_right = self.feature_up(features_left, features_right)
        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        corr_volume = corr_volume_npu(match_left, match_right, self.maxdisp // 4)
        corr_volume = self.corr_stem(corr_volume)
        cost_att = self.corr_feature_att_4(corr_volume, features_left[0])
        att_weights = self.hourglass_att(cost_att, features_left)
        prob = F.softmax(att_weights, dim=2)                    # (B,1,D,H,W)

        # top-k along D, kept in ascending index order
        sel = topk_mask(prob, 2, k)                              # (B,1,D,H,W)
        pos = torch.cumsum(sel, dim=2) - 1.0                     # slot of each selected entry
        D = prob.shape[2]
        didx = torch.arange(D, dtype=prob.dtype, device=prob.device).view(1, 1, D, 1, 1)
        att_l, idx_l = [], []
        for m in range(k):
            oh = sel * torch.clamp(1.0 - torch.abs(pos - m), min=0.0)   # exactly 1 at slot m
            att_l.append((prob * oh).sum(2, keepdim=True))
            idx_l.append((didx * oh).sum(2, keepdim=True))
        att_topk = torch.cat(att_l, 2)                           # (B,1,k,H,W)
        disparity_sample_topk = torch.cat(idx_l, 2).squeeze(1)   # (B,k,H,W)

        concat_features_left = self.concat_feature(features_left[0])
        concat_features_right = self.concat_feature(features_right[0])
        concat_volume = self.concat_volume_generator(concat_features_left, concat_features_right,
                                                     disparity_sample_topk)
        volume = att_topk * concat_volume
        volume = self.concat_stem(volume)
        volume = self.concat_feature_att_4(volume, features_left[0])
        cost = self.hourglass(volume, features_left)

        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = F.softmax(self.spx(xspx), 1)

        # regression over the best 2 of k cost entries (masked softmax)
        c = cost.squeeze(1)                                      # (B,k,H,W)
        m2 = topk_mask(c, 1, 2)
        e = torch.exp(c - c.max(1, keepdim=True)[0]) * m2
        pred = (e * disparity_sample_topk).sum(1, keepdim=True) / e.sum(1, keepdim=True)
        return [context_upsample(pred, spx_pred) * 4]
