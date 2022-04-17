from os.path import join

from copy import deepcopy
import torch
from sbi.inference import SNPE, SNLE
from sbi.utils import BoxUniform, posterior_nn, likelihood_nn, handle_invalid_x
import numpy as np
from omegaconf import DictConfig
import hydra
import pandas as pd
import pickle
import dill
from l5pc.utils.common_utils import load_prior
from l5pc.model import L5PC_20D_theta, L5PC_20D_x
from l5pc.model.utils import return_names, return_x_names, return_xo
from torch import zeros, eye, tensor, float32, as_tensor, Tensor
from l5pc.utils.model_utils import (
    replace_nan,
    add_observation_noise,
)
import logging

log = logging.getLogger("train")


@hydra.main(config_path="config", config_name="train")
def train(cfg: DictConfig) -> None:
    assert cfg.id is not None, "Specify an ID. Format: [model][dim]_[run], e.g. j2_3"

    base_path = "/mnt/qb/macke/mdeistler57/tsnpe_collection/l5pc"
    inference_path = join(base_path, f"results/{cfg.id}/inference")

    if cfg.previous_inference is None:
        previous_feature_list = []
        round_ = 1
    else:
        prev_inference = join(inference_path, cfg.previous_inference)
        with open(join(prev_inference, "used_features.pkl"), "rb") as handle:
            previous_feature_list = pickle.load(handle)
        with open(join(prev_inference, "inference.pkl"), "rb") as handle:
            previous_inferences = dill.load(handle)
        # +1 because in multicompartment we start counting at 1
        # +1 because the counter for inference is only set after the data is passed
        round_ = previous_inferences[0]._round + 1 + 1
    log.info(f"Round: {round_}")

    x_db = L5PC_20D_x()
    theta_db = L5PC_20D_theta()

    data_id = cfg.id if round_ > 1 else "l20_0"
    theta = as_tensor(
        np.asarray(
            (theta_db & {"round": round_} & {"id": data_id}).fetch(*return_names())
        ),
        dtype=float32,
    ).T
    x = as_tensor(
        np.asarray(
            (x_db & {"round": round_} & {"id": data_id}).fetch(*return_x_names())
        ),
        dtype=float32,
    ).T
    theta = theta[: cfg.num_initial]
    x = x[: cfg.num_initial]

    # Extract prior from the config file of the simulations.
    prior = load_prior(data_id)

    log.info("theta dim after loading id: {theta.shape}")
    log.info("x dim after loading id: {x.shape}")

    if cfg.algorithm.name == "snpe":
        method = SNPE
        dens_estim = posterior_nn(
            cfg.density_estimator,
            sigmoid_theta=cfg.algorithm.sigmoid_theta,
            prior=prior.prior_torch,
        )
    elif cfg.algorithm.name == "snle":
        method = SNLE
        dens_estim = likelihood_nn(cfg.density_estimator)
    else:
        raise ValueError

    valid_features = select_features(x, cfg.nan_fraction_threshold_to_exclude)
    if cfg.choose_features == "valid":
        features_to_keep = valid_features
    elif cfg.choose_features == "valid_unused":
        features_to_keep = [f for f in valid_features if f not in previous_feature_list]
    else:
        raise NameError
    with open("used_features.pkl", "wb") as handle:
        pickle.dump(features_to_keep, handle)

    # x_only_good_features = x[:, features_to_keep]
    # is_valid_x, _, _ = handle_invalid_x(x_only_good_features, True)
    if cfg.replace_nan_values or cfg.train_on_all:
        x, replacement_values = replace_nan(x)
        x = add_observation_noise(
            x=x,
            id_=cfg.id,
            noise_multiplier=cfg.observation_noise,
            std_type=cfg.observation_noise_type,
            subset=None,
        )
        x = x[:, features_to_keep]
    else:
        x = x[:, features_to_keep]
        x = add_observation_noise(
            x=x,
            id_=cfg.id,
            noise_multiplier=cfg.observation_noise,
            std_type=cfg.observation_noise_type,
            subset=features_to_keep,
        )
    # if not cfg.train_on_all:
    #     x = x[is_valid_x]
    #     theta = theta[is_valid_x]

    if cfg.num_train is not None:
        theta = theta[: cfg.num_train]
        x = x[: cfg.num_train]

    log.info(f"Selected features: {features_to_keep}")
    log.info(
        f"Names of selected features {np.asarray(return_x_names())[features_to_keep]}"
    )
    log.info(f"theta dim to train: {theta.shape}")
    log.info(f"x dim to train: {x.shape}")

    inferences = []
    for seed in range(cfg.ensemble_size):
        _ = torch.manual_seed(cfg.seed_train + seed)
        if cfg.load_nn_from_prev_inference:
            inference = previous_inferences[seed]
        else:
            inference = method(prior=prior.prior_torch, density_estimator=dens_estim)

        _ = inference.append_simulations(theta, x).train(
            max_num_epochs=cfg.max_num_epochs,
            training_batch_size=cfg.training_batch_size,
        )
        if cfg.previous_inference is not None and not cfg.load_nn_from_prev_inference:
            inference.trained_rounds = round_
        inferences.append(inference)
        log.info(f"_best_val_log_prob {inference._best_val_log_prob}")

    xo_all = return_xo(as_pd=False)[0]
    xo_all = as_tensor(xo_all, dtype=float32)
    if cfg.replace_nan_values or cfg.temper_xo:
        xo = deepcopy(replacement_values)
        xo[features_to_keep] = xo_all[features_to_keep]
    else:
        xo = xo_all[features_to_keep]
    xo = xo.unsqueeze(0)
    log.info(f"xo {xo.shape}")
    with open("xo.pkl", "wb") as handle:
        pickle.dump(xo, handle)

    with open("inference.pkl", "wb") as handle:
        dill.dump(inferences, handle)

    all_val_log_probs = [infer._summary["validation_log_probs"] for infer in inferences]
    all_best_val = [infer._best_val_log_prob for infer in inferences]
    all_epochs = [infer.epoch for infer in inferences]
    with open("val_log_probs.pkl", "wb") as handle:
        pickle.dump(all_val_log_probs, handle)
    np.savetxt("best_val_log_prob.txt", all_best_val, fmt="%10.10f")
    np.savetxt("epochs.txt", all_epochs, fmt="%5.5f")


def select_features(x: Tensor, nan_fraction_threshold_to_exclude: float):
    num_valid_sims_per_feature = np.sum(np.invert(np.isnan(x.numpy())), axis=0)
    condition_features = (
        num_valid_sims_per_feature
        > nan_fraction_threshold_to_exclude * x.numpy().shape[0]
    )
    often_valid_features = np.arange(len(condition_features))[condition_features]
    return often_valid_features


if __name__ == "__main__":
    train()
