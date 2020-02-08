import ray
import json
import numpy as np
from _jsonnet import evaluate_file

from RLPlayground import CONFIG_DIR
from RLPlayground.experiments.dyna_mdp_experiments import Experimenting
from RLPlayground.utils.logger import ProjectLogger
import RLPlayground.utils.plotter as plot

if __name__ == "__main__":
    # logger
    logger = ProjectLogger(level=10)

    # load initial configs for params
    cfg = evaluate_file(f'{CONFIG_DIR}/dyna_mdp_config.jsonnet')
    logger.info(f'Using the following config: \n{cfg}')
    cfg = json.loads(cfg)

    ray.init(
        # local_mode=True,
        ignore_reinit_error=True,
    )

    # specs for the experiment & agent
    experiment_cfg, agent_cfg = cfg['experiment_cfg'], cfg['agent_cfg']
    seeds = np.random.choice(100000, 10, replace=False)
    dp_methods = cfg['dp_methods']
    env_names = cfg['env_names']
    params_vals = [[discount, theta] for theta in cfg['theta_vals'] for discount
                   in cfg['discount_rates']]

    experiment = Experimenting(logger=logger, experiment_cfg=experiment_cfg,
                               agent_cfg=agent_cfg, env_names=env_names,
                               dp_methods=dp_methods, params_vals=params_vals,
                               seeds=seeds)

    # eval hyperparams
    agent_cfg, output = experiment.evaluate_hyperparameters()
    logger.info(f'Tuned hyperparams: \n{agent_cfg}')

    # plot the hyperparams
    plot.generate_plots(episodes=experiment_cfg['episodes'], output=output,
                        plot_hyperparams=True)
    # run dp experiments
    output = experiment.run_dp()
    logger.info(f'Finished running mdp experiments')

    # save plots!
    plot.generate_plots(output=output, episodes=experiment_cfg['episodes'],
                        train_rng=experiment_cfg['train_rng'],
                        plot_hyperparams=False)
