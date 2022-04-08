import logging

import torch
from torch import as_tensor, tensor, Tensor, float32, float64, ones, zeros, eye
from omegaconf import DictConfig
import hydra
import numpy as np
import pandas as pd
from sbi.analysis import pairplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

# These files live in utils because I otherwise had problems with SLURM and
# multiprocessing. See this error: https://www.pythonanywhere.com/forums/topic/27818/
from multicompartment.utils.evaluation_utils import (
    predictive_traces,
    plot_traces,
    plot_summstats,
    compare_gt_log_probs,
    gt_log_prob,
    coverage,
    plot_coverage,
)
from multicompartment.utils.common_utils import (
    load_prior,
    extract_bounds,
    load_posterior,
)
from multicompartment.models.l5pc.utils import return_gt, return_x_names, return_names
from multicompartment.models.l5pc import L5PC_20D_theta, L5PC_20D_x
from multicompartment.models import summstats_l5pc
from sbi.utils import PosteriorSupport

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="evaluation")
def evaluate(cfg: DictConfig) -> None:
    _ = torch.manual_seed(cfg.seed)
    inference, posterior, used_features = load_posterior(cfg.id, cfg.posterior)
    round_ = posterior.trained_rounds
    log.info(f"Posterior after round: {round_}")

    prior = load_prior()
    prior_bounds = extract_bounds(prior).T.numpy()

    posterior._prior = prior.prior_torch

    posterior_support = PosteriorSupport(
        prior=prior.prior_torch,
        posterior=posterior,
        num_samples_to_estimate_support=10_000,
        allowed_false_negatives=0.001,
        use_constrained_prior=True,
        constrained_prior_quanitle=0.001,
    )

    traces, theta = predictive_traces(
        posterior=posterior, num_samples=cfg.num_predictives, num_cores=cfg.cores
    )
    with open("traces.pkl", "wb") as handle:
        pickle.dump(traces, handle)
    theta.to_pickle("parameters.pkl")

    posterior_stats = summstats_l5pc(traces)
    valid = np.invert(np.any(np.isnan(posterior_stats.to_numpy()), axis=1))
    log.info(f"Fraction of all all valid sims: {np.sum(valid) / cfg.num_predictives}")
    posterior_stats.to_pickle("stats.pkl")

    x_db = L5PC_20D_x()
    theta_db = L5PC_20D_theta()

    data_id = "l20_0" if round_ == 1 else cfg.id
    theta = as_tensor(
        (theta_db & {"round": round_} & {"id": data_id}).fetch(*return_names()),
        dtype=float32,
    ).T
    x = as_tensor(
        (x_db & {"round": round_} & {"id": data_id}).fetch(*return_x_names()),
        dtype=float32,
    ).T

    # x = np.asarray(x_db.fetch(*return_x_names())).T[:10000]
    x = pd.DataFrame(x[:10000].numpy(), columns=return_x_names())
    # theta = np.asarray(theta_db.fetch(*return_names())).T[:10000]
    theta = pd.DataFrame(theta[:10000].numpy(), columns=return_names())

    alpha, cov = coverage(posterior, theta, x, used_features)

    with mpl.rc_context(
        fname="/mnt/qb/macke/mdeistler57/multicompartment/multicompartment/.matplotlibrc"
    ):
        plot_traces(traces)
        plt.savefig("traces.png", dpi=200, bbox_inches="tight")
        plt.show()

        plot_summstats(posterior_stats, x)
        plt.savefig("stats.png", dpi=200, bbox_inches="tight")
        plt.show()
        plot_summstats(posterior_stats, x, used_features=used_features)
        plt.savefig("stats_fitted.png", dpi=200, bbox_inches="tight")
        plt.show()

        post_lp, prior_lp = compare_gt_log_probs(posterior)
        log.info(f"Posterior log-prob: {post_lp}")
        log.info(f"Prior log-prob:     {prior_lp}")
        _, acceptance_rate = posterior_support.sample(
            (100,), return_acceptance_rate=True
        )
        log.info(f"log10(acceptance rate of support): {acceptance_rate}%")

        plot_coverage(alpha, cov)
        plt.savefig("coverage.png", dpi=200, bbox_inches="tight")
        plt.show()

        posterior_samples = posterior.sample((1000,), show_progress_bars=False)
        _ = pairplot(
            posterior_samples,
            limits=prior_bounds,
            upper=["kde"],
            ticks=prior_bounds,
            figsize=(10, 10),
            labels=return_names(),
            points=return_gt(as_pd=False),
            points_colors="r",
        )
        plt.savefig("pairplot.png", dpi=200, bbox_inches="tight")
        plt.show()

        gt_log_prob(posterior)
        plt.savefig("log_probs.png", dpi=200, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    evaluate()
