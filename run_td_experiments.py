import ray
import json
from _jsonnet import evaluate_file
import numpy as np

from RLPlayground import CONFIG_DIR, RESULT_DIR, TENSORBOARD_DIR
from RLPlayground.experiments.deep_td_experiments import Experiment
from RLPlayground.utils.logger import ProjectLogger
import RLPlayground.utils.plotter as plot

if __name__ == "__main__":
    # logger
    logger = ProjectLogger(level=10)

    # load initial configs for params
    cfg = evaluate_file(f'{CONFIG_DIR}/n_step_td_config.jsonnet')
    logger.info(f'Using the following config: \n{cfg}')
    cfg = json.loads(cfg)

    ray.init(
        # local_mode=True,
        ignore_reinit_error=True,
    )

    # specs for the experiment & agent
    experiment_cfg, agent_cfg = cfg['experiment_cfg'], cfg['agent_cfg']
    experiment_path = f"{RESULT_DIR}/" \
        f"{cfg['experiment_name']}_experiments.pickle"
    hyperparams_path = f"{RESULT_DIR}/" \
        f"{cfg['experiment_name']}_experiments_hyperparameters.pickle"
    tensorboard_path = f"{TENSORBOARD_DIR}/{cfg['experiment_name']}/trainer"
    experiment_cfg['experiment_path'] = experiment_path
    experiment_cfg['hyperparams_path'] = hyperparams_path
    experiment_cfg['tensorboard_path'] = tensorboard_path
    seeds = np.random.choice(99999, 4, replace=False)
    # seeds = [1337]
    agents = cfg['agents']
    env_names = cfg['env_names']
    params = {'env_names': env_names, 'agents': agents, 'seeds': seeds,
              'experiment_cfg': experiment_cfg, 'agent_cfg': agent_cfg}

    experiment = Experiment.build(type=cfg['experiment_name'], logger=logger,
                                  params=params)

    # # eval hyperparams
    # agent_cfg, output = experiment.tune_hyperparams()
    # logger.info(f'Tuned hyperparams: \n{agent_cfg}')
    #
    # # plot the hyperparams
    # plot.generate_plots(output=output, plot_hyperparams=True,
    # use_tensorboard=True, experiment_cfg=experiment_cfg)

    # run dp experiments
    output = experiment.run()
    logger.info(f'Finished running experiments')

    # save plots!
    # plot.generate_plots(output=output, plot_hyperparams=False,
    # use_tensorboard=True, expeeriment_cfg=experiment_cfg)
