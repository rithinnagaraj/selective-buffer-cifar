import torch
import torch.nn.functional as F

activations_a = torch.load("./activations/split_a.pt")

activations_b = torch.load("./activations/split_b.pt")

def compare_activations():
    # (25000, 512) for both splits
    last_layer_a = activations_a["last_layer_activations"]
    last_layer_b = activations_b["last_layer_activations"]
    ll_arr = []

    # (25000, 2048) for both splits
    second_to_last_layer_a = activations_a["second_to_last_layer_activations"]
    second_to_last_layer_b = activations_b["second_to_last_layer_activations"]
    stll_arr = []

    for idx in range(len(last_layer_a[:, 0])):
        ll_cos_sim = F.cosine_similarity(last_layer_a[idx], last_layer_b[idx], dim=0)
        stll_cos_sim = F.cosine_similarity(second_to_last_layer_a[idx], second_to_last_layer_b[idx], dim=0)

        ll_arr.append(torch.tensor([idx, (1-ll_cos_sim.item())]))
        stll_arr.append(torch.tensor([idx, (1-stll_cos_sim.item())]))

    ll_arr = torch.stack(ll_arr)
    stll_arr = torch.stack(stll_arr)

    combined_scores = 0.5 * ll_arr[:, 1] + 0.5 * stll_arr[:, 1]
    combined_scores = torch.stack([ll_arr[:, 0], combined_scores], dim=1)
    print(combined_scores.shape)
    print(combined_scores[:10])

    # (25000, 2)
    sorted_scores = combined_scores[torch.argsort(combined_scores[:, 1], descending=True)]

    print(sorted_scores.shape)
    print(sorted_scores[:10])

    return sorted_scores


def build_buffer():
    # (25000, 3, 32, 32) for both splits
    datapoints = activations_a["data_points"]
    labels = activations_a["last_layer_labels"]

    idx = compare_activations()

    buffer_contents = datapoints[idx[:6000, 0].int()]
    buffer_labels = labels[idx[:6000, 0].int()]

    torch.save((buffer_contents, buffer_labels), "replay_buffer.pt")

if __name__ == "__main__":
    build_buffer()