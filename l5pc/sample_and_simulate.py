import logging
from multiprocessing import Pool
import pickle
import time
import os
import torch
from omegaconf import DictConfig
import hydra

import numpy as np
import pandas as pd

# These files live in utils because I otherwise had problems with SLURM and
# multiprocessing. See this error: https://www.pythonanywhere.com/forums/topic/27818/
from l5pc.utils.simulation_utils import (
    assemble_prior,
    assemble_simulator,
    assemble_db,
    write_to_dj,
)
from l5pc.utils.common_utils import load_posterior
from l5pc.model.utils import return_gt, return_names, return_xo
from sbi.utils import BoxUniform
from sbi.utils.support_posterior import PosteriorSupport

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="sim_model")
def sample_and_simulate(cfg: DictConfig) -> None:
    print(cfg)
    start_time = time.time()
    log.debug(f"Starting run! {time.time() - start_time}")

    assert cfg.id is not None, "Specify an ID. Format: [model][dim]_[run], e.g. j2_3"

    prior = assemble_prior(cfg)
    sim_and_stats = assemble_simulator(cfg.model)
    theta_db, x_db = assemble_db(cfg)

    log.debug(f"Assembled! {time.time() - start_time}")

    seed = int((time.time() % 1) * 1e7) if cfg.seed_prior is None else cfg.seed_prior
    _ = torch.manual_seed(seed)
    np.savetxt(f"seed.txt", [seed], fmt="%d")

    remaining_sims = cfg.sims

    log.debug(f"Starting loop! {time.time() - start_time}")

    if cfg.proposal is None:
        proposal = prior
        round_ = 1
    else:
        inference, posterior, _ = load_posterior(cfg.id, cfg.proposal)
        round_ = inference[0]._round + 2
        log.debug(f"Loaded posterior, round", round_)
        if cfg.thr_proposal:
            _ = torch.manual_seed(0)  # Set seed=0 only for building the proposal.
            proposal = PosteriorSupport(
                prior=prior.prior_torch,
                posterior=posterior,
                num_samples_to_estimate_support=cfg.num_samples_to_estimate_support,
                allowed_false_negatives=cfg.allowed_false_negatives,
                use_constrained_prior=cfg.use_constrained_prior,
                constrained_prior_quanitle=cfg.constrained_prior_quanitle,
            )
            log.debug("Built support")
            _ = torch.manual_seed(seed)
        else:
            proposal = posterior

    while remaining_sims > 0:
        num_to_simulate = min(remaining_sims, cfg.sims_until_save)
        log.debug(f"num_to_simulate", num_to_simulate)
        theta = proposal.sample((num_to_simulate,))

        log.debug(f"Sampled proposal")
        if isinstance(theta, torch.Tensor):
            theta = pd.DataFrame(theta.numpy(), columns=return_names())

        gt = return_gt()
        theta_full = pd.concat([gt] * theta.shape[0], ignore_index=True)
        for specified_parameters in theta.keys():
            theta_full[specified_parameters] = theta[specified_parameters].to_numpy()

        log.debug(f"Time to obtain theta: {time.time() - start_time}")

        # Each worker should process a batch of simulations to reduce the overhead of
        # loading neuron.
        num_splits = max(1, num_to_simulate // cfg.sims_per_worker)
        batches = np.array_split(theta_full, num_splits)

        log.debug(f"Time to obtain batches: {time.time() - start_time}")

        with Pool(cfg.cores) as pool:
            x_list = pool.map(sim_and_stats, batches)

        log.debug(f"Sims done {time.time() - start_time}")
        x = pd.concat(x_list, ignore_index=True)
        log.debug(f"Sims concatenated {time.time() - start_time}")

        # Deal with empty dimensions if not all protocols are run.
        # x_template = return_xo()
        # for col in x_template.columns:
        #     x_template[col] = -9191919.919
        # x_template = pd.concat([x_template] * num_to_simulate, ignore_index=True)
        # for col in x.columns:
        #     x_template[col] = x[col]

        if cfg.save_sims:
            write_to_dj(theta_full, x, theta_db, x_db, round_, cfg.id)

        log.info(f"Written to dj {time.time() - start_time}")

        remaining_sims -= num_to_simulate


if __name__ == "__main__":
    sample_and_simulate()
