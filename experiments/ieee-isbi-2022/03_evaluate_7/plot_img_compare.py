#%%
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# %%
valid = {}  # run #7
valid["90nm"] = {
    "link_seed3": {  # compared to real 90nm
        "normalized_root_mse": 0.4070499767583528,
        "peak_signal_noise_ratio": 13.023355817477524,
        "structural_similarity": 0.50558386994053,
    },
    "link_seed13": {  # compared to real 30nm
        "normalized_root_mse": 0.410932263263303,
        "peak_signal_noise_ratio": 12.940905701371308,
        "structural_similarity": 0.4924305371123494,
    },
    "link_seed42": {
        "normalized_root_mse": 0.406833637845444,
        "peak_signal_noise_ratio": 13.027973421106553,
        "structural_similarity": 0.5094365871233267,
    },
    "split_seed3": {
        "normalized_root_mse": 0.4275923430948426,
        "peak_signal_noise_ratio": 12.595712109389641,
        "structural_similarity": 0.4021388720124944,
    },
    "split_seed13": {
        "normalized_root_mse": 0.4141233917488965,
        "peak_signal_noise_ratio": 12.873715256954716,
        "structural_similarity": 0.4326791927048835,
    },
    "split_seed42": {
        "normalized_root_mse": 0.4177192250451344,
        "peak_signal_noise_ratio": 12.79862123056121,
        "structural_similarity": 0.4730341664046441,
    },
}
valid["30nm"] = {
    "link_seed3": {
        "normalized_root_mse": 0.49826226265105517,
        "peak_signal_noise_ratio": 11.195354056738385,
        "structural_similarity": 0.24491524727718028,
    },
    "link_seed13": {
        "normalized_root_mse": 0.48866432023128675,
        "peak_signal_noise_ratio": 11.364301369338792,
        "structural_similarity": 0.2400500075892979,
    },
    "link_seed42": {
        "normalized_root_mse": 0.5039289071498152,
        "peak_signal_noise_ratio": 11.097128538978954,
        "structural_similarity": 0.2439383819147557,
    },
    "split_seed3": {
        "normalized_root_mse": 0.474231080163536,
        "peak_signal_noise_ratio": 11.624713706367185,
        "structural_similarity": 0.1718275921947333,
    },
    "split_seed13": {
        "normalized_root_mse": 1.0,
        "peak_signal_noise_ratio": 5.144513973828726,
        "structural_similarity": 0.00018324294839675392,
    },
    "split_seed42": {
        "normalized_root_mse": 0.4837886173967707,
        "peak_signal_noise_ratio": 11.4514010529612,
        "structural_similarity": 0.22832709820707656,
    },
}

same = {}  # run #8
same["90nm"] = {
    "link_seed3": {
        "normalized_root_mse": 0.40643792528895917,
        "peak_signal_noise_ratio": 13.036425987312168,
        "structural_similarity": 0.5100103461745463,
    },
    "link_seed13": {
        "normalized_root_mse": 0.40708161353273853,
        "peak_signal_noise_ratio": 13.022680758226086,
        "structural_similarity": 0.5153178314125576,
    },
    "link_seed42": {
        "normalized_root_mse": 0.40902204240737133,
        "peak_signal_noise_ratio": 12.981376243305437,
        "structural_similarity": 0.5047897452097853,
    },
    "split_seed3": {
        "normalized_root_mse": 0.4174255931351246,
        "peak_signal_noise_ratio": 12.804729044234076,
        "structural_similarity": 0.4005852990085749,
    },
    "split_seed13": {
        "normalized_root_mse": 0.406951804523571,
        "peak_signal_noise_ratio": 13.025450931322677,
        "structural_similarity": 0.486221086355541,
    },
    "split_seed42": {
        "normalized_root_mse": 0.4201577335516406,
        "peak_signal_noise_ratio": 12.748063268911684,
        "structural_similarity": 0.4498229390835254,
    },
}
same["30nm"] = {
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
        "normalized_root_mse": 0.20138064420287916,
        "peak_signal_noise_ratio": 19.135955987081637,
        "structural_similarity": 0.35334042226547124,
    }
}
real["30nm"] = {
    "real_90nm": {
        "normalized_root_mse": 0.30896153399713117,
        "peak_signal_noise_ratio": 15.346425719485655,
        "structural_similarity": 0.3556586658635673,
    }
}

# %%
tecs = {"real": real, "valid": valid, "same": same}
means = defaultdict(lambda: defaultdict(defaultdict))
maxs = defaultdict(lambda: defaultdict(defaultdict))
mins = defaultdict(lambda: defaultdict(defaultdict))
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
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
# DIVIDING LINK vs. SPLIT
valid_link = {}  # run #7
valid_link["90nm"] = {
    "link_seed3": {  # compared to real 90nm
        "normalized_root_mse": 0.4070499767583528,
        "peak_signal_noise_ratio": 13.023355817477524,
        "structural_similarity": 0.50558386994053,
    },
    "link_seed13": {  # compared to real 30nm
        "normalized_root_mse": 0.410932263263303,
        "peak_signal_noise_ratio": 12.940905701371308,
        "structural_similarity": 0.4924305371123494,
    },
    "link_seed42": {
        "normalized_root_mse": 0.406833637845444,
        "peak_signal_noise_ratio": 13.027973421106553,
        "structural_similarity": 0.5094365871233267,
    },
}
valid_link["30nm"] = {
    "link_seed3": {
        "normalized_root_mse": 0.49826226265105517,
        "peak_signal_noise_ratio": 11.195354056738385,
        "structural_similarity": 0.24491524727718028,
    },
    "link_seed13": {
        "normalized_root_mse": 0.48866432023128675,
        "peak_signal_noise_ratio": 11.364301369338792,
        "structural_similarity": 0.2400500075892979,
    },
    "link_seed42": {
        "normalized_root_mse": 0.5039289071498152,
        "peak_signal_noise_ratio": 11.097128538978954,
        "structural_similarity": 0.2439383819147557,
    },
}

valid_split = {}
valid_split["90nm"] = {
    "split_seed3": {
        "normalized_root_mse": 0.4275923430948426,
        "peak_signal_noise_ratio": 12.595712109389641,
        "structural_similarity": 0.4021388720124944,
    },
    "split_seed13": {
        "normalized_root_mse": 0.4141233917488965,
        "peak_signal_noise_ratio": 12.873715256954716,
        "structural_similarity": 0.4326791927048835,
    },
    "split_seed42": {
        "normalized_root_mse": 0.4177192250451344,
        "peak_signal_noise_ratio": 12.79862123056121,
        "structural_similarity": 0.4730341664046441,
    },
}
valid_split["30nm"] = {
    "split_seed3": {
        "normalized_root_mse": 0.474231080163536,
        "peak_signal_noise_ratio": 11.624713706367185,
        "structural_similarity": 0.1718275921947333,
    },
    "split_seed13": {
        "normalized_root_mse": 1.0,
        "peak_signal_noise_ratio": 5.144513973828726,
        "structural_similarity": 0.00018324294839675392,
    },
    "split_seed42": {
        "normalized_root_mse": 0.4837886173967707,
        "peak_signal_noise_ratio": 11.4514010529612,
        "structural_similarity": 0.22832709820707656,
    },
}

same_link = {}  # run #8
same_link["90nm"] = {
    "link_seed3": {
        "normalized_root_mse": 0.40643792528895917,
        "peak_signal_noise_ratio": 13.036425987312168,
        "structural_similarity": 0.5100103461745463,
    },
    "link_seed13": {
        "normalized_root_mse": 0.40708161353273853,
        "peak_signal_noise_ratio": 13.022680758226086,
        "structural_similarity": 0.5153178314125576,
    },
    "link_seed42": {
        "normalized_root_mse": 0.40902204240737133,
        "peak_signal_noise_ratio": 12.981376243305437,
        "structural_similarity": 0.5047897452097853,
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
        "normalized_root_mse": 0.4174255931351246,
        "peak_signal_noise_ratio": 12.804729044234076,
        "structural_similarity": 0.4005852990085749,
    },
    "split_seed13": {
        "normalized_root_mse": 0.406951804523571,
        "peak_signal_noise_ratio": 13.025450931322677,
        "structural_similarity": 0.486221086355541,
    },
    "split_seed42": {
        "normalized_root_mse": 0.4201577335516406,
        "peak_signal_noise_ratio": 12.748063268911684,
        "structural_similarity": 0.4498229390835254,
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
        "normalized_root_mse": 0.20138064420287916,
        "peak_signal_noise_ratio": 19.135955987081637,
        "structural_similarity": 0.35334042226547124,
    }
}
real["30nm"] = {
    "real_90nm": {
        "normalized_root_mse": 0.30896153399713117,
        "peak_signal_noise_ratio": 15.346425719485655,
        "structural_similarity": 0.3556586658635673,
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
