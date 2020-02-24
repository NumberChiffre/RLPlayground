{
    experiment_name: 'dyna_mdp',

    # dp methods
    algos: ['policy_iteration', 'value_iteration'],

    # environments
    env_names: ['FrozenLake8x8-v0'],

    # specs for the experiment
    experiment_cfg:
    {
        'runs': 1,
        'steps': 2000,
        'episodes': 502,
        'train_rng': 10,
        'test_rng': 5,
    },

    # specs for the agent for tabular environments
    agent_cfg:
    {
        'FrozenLake-v0': {
            'value_iteration': {
                'theta': 1e-6,
                'gamma': 1.0,
            },
            'policy_iteration': {
                'theta': 1e-6,
                'gamma': 1.0,
            },
        },
        'FrozenLake8x8-v0': {
            'value_iteration': {
                'theta': 1e-6,
                'gamma': 1.0,
            },
            'policy_iteration': {
                'theta': 1e-6,
                'gamma': 1.0,
            },
        },
        'Taxi-v3': {
            'value_iteration': {
                'theta': 1e-6,
                'gamma': 1.0,
            },
            'policy_iteration': {
                'theta': 1e-6,
                'gamma': 1.0,
            },
        }
    },

    # random seeds
    seeds: [34243, 3232, 3223, 121, 121, 1212, 32, 111, 221, 5454],
//    seeds: [22],
    # eval hyperparams
    theta_vals: [1e-6, 1e-8, 1e-11],
    gammas: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1],

}