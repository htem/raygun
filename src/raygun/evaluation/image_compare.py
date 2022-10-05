import sys
import daisy
from raygun import read_config
from raygun.utils import to_json
from skimage import metrics as skimet

import logging

logger = logging.getLogger(__name__)


def image_compare(
    test,
    target,
    metrics=["normalized_root_mse", "peak_signal_noise_ratio", "structural_similarity"],
):
    roi = test.roi.intersect(target.roi)
    test = test.to_ndarray(roi)
    target = target.to_ndarray(roi)

    results = {}
    for metric in metrics:
        results[metric] = getattr(skimet, metric)(target, test)
        logger.info(f"\t{metric}: {results[metric]}")

    return results


def images_compare(config=None):
    if config is None:
        config = sys.argv[1]

    config = read_config(config)

    target = daisy.open_ds(
        config["target_source"]["path"], config["target_source"]["ds"]
    )
    results = {}
    for name, dataset in config["test_sources"].items():
        try:
            logger.info(f"Comparing {name} to target...")
            test = daisy.open_ds(dataset["path"], dataset["ds"])
            results[name] = image_compare(test, target)
            del test
        except:
            logger.info(f"Failed to compare {name} to target.")

    to_json(results, config["metrics_path"])


if __name__ == "__main__":
    images_compare()
