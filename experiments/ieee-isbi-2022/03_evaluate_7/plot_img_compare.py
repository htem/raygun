#%%
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# %%
# DIVIDING LINK vs. SPLIT
valid_link = {}  # run #7
valid_link["90nm"] = {
    "link_seed3": {
        "normalized_root_mse": 0.1355599601559981,
        "peak_signal_noise_ratio": 22.58388754178613,
        "structural_similarity": 0.5971274763636932,
    },
    "link_seed13": {
        "normalized_root_mse": 0.1487141861451615,
        "peak_signal_noise_ratio": 21.779468217688887,
        "structural_similarity": 0.581553505905969,
    },
    "link_seed42": {
        "normalized_root_mse": 0.13335912086965168,
        "peak_signal_noise_ratio": 22.726061714679517,
        "structural_similarity": 0.6019693070557244,
    },
}
valid_link["30nm"] = {
    "link_seed3": {
        "normalized_root_mse": 0.2688072997144831,
        "peak_signal_noise_ratio": 16.629633701314493,
        "structural_similarity": 0.3033689184378308,
    },
    "link_seed13": {
        "normalized_root_mse": 0.2642739677635674,
        "peak_signal_noise_ratio": 16.77736716209853,
        "structural_similarity": 0.29481627730500704,
    },
    "link_seed42": {
        "normalized_root_mse": 0.2810432484901749,
        "peak_signal_noise_ratio": 16.242991731921755,
        "structural_similarity": 0.3022412669405743,
    },
}

valid_split = {}
valid_split["90nm"] = {
    "split_seed3": {
        "normalized_root_mse": 0.19521800877831033,
        "peak_signal_noise_ratio": 19.416118618922326,
        "structural_similarity": 0.4750412374763705,
    },
    "split_seed13": {
        "normalized_root_mse": 0.15836141535023626,
        "peak_signal_noise_ratio": 21.233528700303246,
        "structural_similarity": 0.5117169140303613,
    },
    "split_seed42": {
        "normalized_root_mse": 0.17066811350524919,
        "peak_signal_noise_ratio": 20.58346843207555,
        "structural_similarity": 0.5573613904308932,
    },
}
valid_split["30nm"] = {
    "split_seed3": {
        "normalized_root_mse": 0.291638131833061,
        "peak_signal_noise_ratio": 15.92156871400848,
        "structural_similarity": 0.20385557154677644,
    },
    "split_seed13": {
        "normalized_root_mse": 1.0,
        "peak_signal_noise_ratio": 5.218454865637288,
        "structural_similarity": 9.624664095663182e-05,
    },
    "split_seed42": {
        "normalized_root_mse": 0.26463320991215444,
        "peak_signal_noise_ratio": 16.765567972117704,
        "structural_similarity": 0.27868510997603146,
    },
}

same_link = {}  # run #8
same_link["90nm"] = {
    "link_seed3": {
        "normalized_root_mse": 0.1331461963644304,
        "peak_signal_noise_ratio": 22.73994090603148,
        "structural_similarity": 0.6027468997546921,
    },
    "link_seed13": {
        "normalized_root_mse": 0.13528856761763633,
        "peak_signal_noise_ratio": 22.60129421883255,
        "structural_similarity": 0.6087160347944583,
    },
    "link_seed42": {
        "normalized_root_mse": 0.1419233315003493,
        "peak_signal_noise_ratio": 22.185440247888735,
        "structural_similarity": 0.5963789395420624,
    },
}
same_link["30nm"] = {
    "link_seed3": {
        "normalized_root_mse": 0.5019334129360046,
        "peak_signal_noise_ratio": 11.131591834615522,
        "structural_similarity": 0.23699391351667545,
    },
    "link_seed13": {
        "normalized_root_mse": 0.49552985872677335,
        "peak_signal_noise_ratio": 11.243117403272421,
        "structural_similarity": 0.24349817431155177,
    },
    "link_seed42": {
        "normalized_root_mse": 0.4967844224457209,
        "peak_signal_noise_ratio": 11.221154587621376,
        "structural_similarity": 0.23774150447718825,
    },
}


same_split = {}
same_split["90nm"] = {
    "split_seed3": {
        "normalized_root_mse": 0.16904716148908883,
        "peak_signal_noise_ratio": 20.66635853569176,
        "structural_similarity": 0.47266311321617893,
    },
    "split_seed13": {
        "normalized_root_mse": 0.134786170287619,
        "peak_signal_noise_ratio": 22.633609516666855,
        "structural_similarity": 0.5741467458849363,
    },
    "split_seed42": {
        "normalized_root_mse": 0.17750613623737913,
        "peak_signal_noise_ratio": 20.242248775233275,
        "structural_similarity": 0.5309279820716661,
    },
}
same_split["30nm"] = {
    "split_seed3": {
        "normalized_root_mse": 0.49535700353243384,
        "peak_signal_noise_ratio": 11.24614782221088,
        "structural_similarity": 0.1931191268088784,
    },
    "split_seed13": {
        "normalized_root_mse": 0.49195590511019016,
        "peak_signal_noise_ratio": 11.30599041545432,
        "structural_similarity": 0.20146458955024635,
    },
    "split_seed42": {
        "normalized_root_mse": 0.4858738047270455,
        "peak_signal_noise_ratio": 11.414044268722723,
        "structural_similarity": 0.21298072642300428,
    },
}

real = {}
real["90nm"] = {
    "real_30nm": {
        "normalized_root_mse": 0.20083305321954512,
        "peak_signal_noise_ratio": 19.169812376295518,
        "structural_similarity": 0.35483208154093854,
    }
}
real["30nm"] = {
    "real_90nm": {
        "normalized_root_mse": 0.21266902773024562,
        "peak_signal_noise_ratio": 18.664369953833774,
        "structural_similarity": 0.3794669630129207,
    }
}

#%%
tecs = {
    "real": real,
    "valid link": valid_link,
    "valid split": valid_split,
    "same link": same_link,
    "same split": same_split,
}
means = defaultdict(lambda: defaultdict(defaultdict))
maxs = defaultdict(lambda: defaultdict(defaultdict))
mins = defaultdict(lambda: defaultdict(defaultdict))
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for c, metric in enumerate(
    ["normalized_root_mse", "peak_signal_noise_ratio", "structural_similarity"]
):
    for r, res in enumerate(["90nm", "30nm"]):
        axes[r, c].set_title(" ".join(metric.split("_")))
        if c == 0:
            axes[r, c].set_ylabel(res)
        # axes[r,c].set_xticklabels([n for n in tecs.keys()])
        axes[r, c].set_xticks(range(len(tecs)))
        for x, (name, tec) in enumerate(tecs.items()):
            vals = [v[metric] for v in tec[res].values()]
            means[metric][res][name] = np.mean(vals)
            maxs[metric][res][name] = np.max(vals)
            mins[metric][res][name] = np.min(vals)
            axes[r, c].scatter(
                np.ones_like(vals) * x,
                vals,
                label=f"{name} (mean={means[metric][res][name]:10.4f}, min={mins[metric][res][name]:10.4f}, max={maxs[metric][res][name]:10.4f}",
            )

        axes[r, c].set_xticklabels(
            [
                f"{n}\nmean={means[metric][res][n]:3.4f}\nmin={mins[metric][res][n]:3.4f}\nmax={maxs[metric][res][n]:3.4f}"
                for n in tecs.keys()
            ]
        )
fig.tight_layout()

# %%
