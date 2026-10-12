import torch
import torch.nn.functional as F

LOSS_NAMES = ("dpo", "sppo", "fpa")


def dpo_obj(logp_w, logp_l, logp_ref_w, logp_ref_l, beta):
    a = beta * (logp_w - logp_ref_w)
    b = beta * (logp_l - logp_ref_l)
    loss = -F.logsigmoid(a - b).mean()
    metrics = {"dpo_loss": loss.detach()}
    return loss, metrics


def sppo_obj(logp_w, logp_l, logp_ref_w, logp_ref_l, beta):
    a = beta * (logp_w - logp_ref_w)
    b = beta * (logp_l - logp_ref_l)
    loss = ((a - 0.5) ** 2 + (b + 0.5) ** 2).mean()
    metrics = {"sppo_loss": loss.detach()}
    return loss, metrics


def ce_obj(logits_w, target_w, residue_mask):
    ce_per_tok = F.cross_entropy(logits_w.transpose(1, 2), target_w, reduction="none")
    loss = (ce_per_tok * residue_mask).sum() / residue_mask.sum()
    metrics = {"ce_loss": loss.detach()}
    return loss, metrics


def fpa_obj(logp_w, logp_l, logp_ref_w, logp_ref_l, beta, lambda_ce, logits_w, target_w, residue_mask):
    sppo_loss, m_sppo = sppo_obj(logp_w, logp_l, logp_ref_w, logp_ref_l, beta) # critical-residue log-probabilities only (from compute_mask)
    ce_loss, m_ce = ce_obj(logits_w, target_w, residue_mask)
    loss = sppo_loss + lambda_ce * ce_loss
    metrics = {"fpa_loss": loss.detach(), **m_ce, **m_sppo}
    return loss, metrics


def preference_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, loss_name, beta, lambda_ce=1.0,
                    logits_w=None, target_w=None, residue_mask=None):
    if loss_name == "dpo":
        loss, metrics = dpo_obj(logp_w, logp_l, logp_ref_w, logp_ref_l, beta)
    elif loss_name == "sppo":
        loss, metrics = sppo_obj(logp_w, logp_l, logp_ref_w, logp_ref_l, beta)
    elif loss_name == "fpa":
        loss, metrics = fpa_obj(logp_w, logp_l, logp_ref_w, logp_ref_l, beta, lambda_ce,
                                logits_w, target_w, residue_mask)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}. Expected one of {LOSS_NAMES}")

    metrics.update({
        "logp_w": logp_w.detach(), 
        "logp_l": logp_l.detach(), 
        "logp_ref_w": logp_ref_w.detach(), 
        "logp_ref_l": logp_ref_l.detach()
    })

    chosen_rewards = beta * (logp_w - logp_ref_w).detach()
    rejected_rewards = beta * (logp_l - logp_ref_l).detach()

    metrics.update({
        "rewards/chosen": chosen_rewards.detach(),
        "rewards/rejected": rejected_rewards.detach(),
        "rewards/margin": (chosen_rewards - rejected_rewards).detach(),
        "rewards/chosen>rejected": (chosen_rewards > rejected_rewards).float().detach()
    })

    return loss, metrics


def compute_mask(pad_mask, target_w, target_l, masked):
    critical_mask = target_w != target_l
    residue_mask = (~pad_mask).long() # non pad mask
    loss_mask = residue_mask * critical_mask.any(dim=1).long().unsqueeze(1) # kill pos_seq == neg_seq samples
    if masked:
        loss_mask = loss_mask * critical_mask.long()
    return loss_mask, residue_mask


def get_batch_logps(logits, labels, average_log_prob=False, mask=None):
    assert logits.shape[:-1] == labels.shape

    if mask is None:
        mask = torch.ones_like(labels)
    mask = mask.to(dtype=logits.dtype, device=logits.device)

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_token_logps = (per_token_logps * mask).sum(-1)

    if average_log_prob:
        per_token_logps = per_token_logps / mask.sum(-1).clamp(min=1)

    return per_token_logps
