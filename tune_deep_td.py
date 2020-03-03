import ray
from _jsonnet import evaluate_file
import json

from ray import tune

from RLPlayground import CONFIG_DIR, RESULT_DIR, TENSORBOARD_DIR
from RLPlayground.configs.config import AgentConfig, TuneConfig
from RLPlayground.tune.deep_td_tune import DeepTDTrainable

if __name__ == '__main__':
    # load initial configs for params
    cfg = evaluate_file(f'{CONFIG_DIR}/n_step_td_config.jsonnet')
    cfg = json.loads(cfg)

    tune_cfg = evaluate_file(f'{CONFIG_DIR}/n_step_td_tune_config.jsonnet')
    tune_cfg = json.loads(tune_cfg)

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
    agents = cfg['agents']
    env_names = cfg['env_names']

    # tune for each environment, each agent
    for env_name in env_names:
        for agent in agents:
            agent_cfg[agent]['seed'] = 1337

            # select tuning params with grid search
            agent_cfg[agent]['lr'] = tune.choice(experiment_cfg['lrs'])
            agent_cfg[agent]['experience_replay']['params'][
                'capacity'] = tune.choice(
                experiment_cfg['replay_buffer_capacities'])
            agent_cfg[agent]['warm_up_freq'] = tune.choice(
                experiment_cfg['warm_up_freqs'])
            agent_cfg[agent]['update_freq'] = tune.choice(
                experiment_cfg['update_freqs'])
            agent_cfg[agent]['batch_size'] = tune.choice(
                experiment_cfg['batch_sizes'])
            agent_cfg[agent]['experience_replay']['params'][
                'beta'] = tune.choice(experiment_cfg['betas'])
            agent_cfg[agent]['grad_clipping'] = tune.choice(
                experiment_cfg['grad_clippings'])
            # agent_cfg[agent]['eps'] = tune.choice(
            #     experiment_cfg['epsilons'])

            # # gotta find a way to encode all this..
            # tune_cfg['scheduler']['kwargs']['hyperparam_mutations'] = {
            #     'lr': lambda: tune.uniform(0.001, 0.05),
            #     'capacity': lambda: tune.choice(
            #         experiment_cfg['replay_buffer_capacities']),
            #     'warm_up_freqs': lambda: tune.choice(
            #         experiment_cfg['warm_up_freqs']),
            #     'update_freq': lambda: tune.choice(
            #         experiment_cfg['update_freqs']),
            #     'batch_size': lambda: tune.choice(
            #         experiment_cfg['batch_sizes']),
            #     'beta': lambda: tune.choice(experiment_cfg['betas']),
            #     'grad_clipping': lambda: tune.choice(
            #         experiment_cfg['grad_clippings']),
            #     'gamma': lambda: tune.choice(experiment_cfg['gammas']),
            #     # 'eps': lambda: tune.choice(experiment_cfg['epsilon'])
            # }

            # pass config along with optimizing params
            config = {'env_name': env_name, 'agent': agent,
                      'consecutive_steps_to_solve': agent_cfg[
                          'consecutive_steps_to_solve'],
                      'experiment_cfg': experiment_cfg,
                      'agent_cfg': agent_cfg[agent]}
            trials = tune.run(
                run_or_experiment=DeepTDTrainable,
                local_dir=f"{RESULT_DIR}/tune/{cfg['experiment_name']}/"
                f"{env_name}/{agent}",
                config=AgentConfig(config).config,
                **TuneConfig(tune_cfg).config
            )
            # track = tune.track.TrackSession(experiment_dir='')
            # tune.track.log(mean_accuracy=episode_result['cum_reward'])
            # gotta upgrade ray so that tune.run returns an ExperimentAnalysis object
            # analysis = ExperimentAnalysis(experiment_path=trials[0].local_dir)
            # print("Best config: ",
            #       analysis.get_best_config(metric="cum_reward"))
