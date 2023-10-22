import copy
import numpy as np
import torch


def random_walk(
    device, model, cnn_predictions, batch_size, var_sequence, this_random_sample
):
    _, old_probs = model.forward_sampling(
        this_random_sample,
        cnn_predictions,
        var_sequence,
        device,
        sample_probs=False,
    )
    old_probs = this_random_sample * old_probs + (1 - this_random_sample) * (
        1 - old_probs
    )
    old_pll_scores = torch.sum(torch.log(old_probs), dim=1)

    updated_random_sample = this_random_sample.clone()
    index_to_flip = torch.randint(
        0, this_random_sample.shape[1], (this_random_sample.shape[0], 1)
    )
    updated_random_sample[np.arange(batch_size)[:, None], index_to_flip] = (
        1 - updated_random_sample[np.arange(batch_size)[:, None], index_to_flip]
    )
    _, new_probs = model.forward_sampling(
        updated_random_sample,
        cnn_predictions,
        var_sequence,
        device,
        sample_probs=False,
    )
    new_probs = updated_random_sample * new_probs + (1 - updated_random_sample) * (
        1 - new_probs
    )
    new_pll_scores = torch.sum(torch.log(new_probs), dim=1)
    mask = new_pll_scores > old_pll_scores
    this_random_sample[np.arange(batch_size)[:, None], index_to_flip] = torch.where(
        mask.unsqueeze(1),
        updated_random_sample[np.arange(batch_size)[:, None], index_to_flip],
        this_random_sample[np.arange(batch_size)[:, None], index_to_flip],
    )
    best_pll_scores = torch.where(mask, new_pll_scores, old_pll_scores)
    return this_random_sample, best_pll_scores


def greedy_search(
    device, model, cnn_predictions, batch_size, var_sequence, this_random_sample
):
    _, old_probs = model.forward_sampling(
        this_random_sample,
        cnn_predictions,
        var_sequence,
        device,
        sample_probs=False,
    )
    old_probs = this_random_sample * old_probs + (1 - this_random_sample) * (
        1 - old_probs
    )
    best_pll_scores = torch.sum(torch.log(old_probs), dim=1)
    best_sample = this_random_sample.clone()

    for idx in range(this_random_sample.shape[1]):
        updated_random_sample = copy.deepcopy(this_random_sample)
        index_to_flip = idx
        updated_random_sample[:, index_to_flip] = (
            1 - updated_random_sample[:, index_to_flip]
        )
        _, new_probs = model.forward_sampling(
            updated_random_sample,
            cnn_predictions,
            var_sequence,
            device,
            sample_probs=False,
        )
        new_probs = updated_random_sample * new_probs + (1 - updated_random_sample) * (
            1 - new_probs
        )
        new_pll_scores = torch.sum(torch.log(new_probs), dim=1)
        mask = new_pll_scores > best_pll_scores
        best_pll_scores = torch.where(mask, new_pll_scores, best_pll_scores)
        best_sample[:, index_to_flip] = torch.where(
            mask,
            updated_random_sample[:, index_to_flip],
            best_sample[:, index_to_flip],
        )
    return best_sample, best_pll_scores
