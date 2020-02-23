{
    experiment_name: 'n_step_td',

    # dp methods
    algos: ['q_learning', 'td'],

    # environments
    env_names: ['cartpole-v0'],

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
        'cartpole-v0':
        {
            'q-learning':{
                'eps': 0.1,
                'gamma': 0.9,
                'n_step': 4,
                'alpha': 0.2,
                'algo': 'q_learning',
            },
            'sarsa':{
                'eps': 0.1,
                'gamma': 0.9,
                'n_step': 4,
                'alpha': 0.2,
                'algo': 'sarsa',
            },
            'expected_sarsa':{
                'eps': 0.1,
                'gamma': 0.9,
                'n_step': 4,
                'alpha': 0.2,
                'algo': 'expected_sarsa',
            }
        }
    },

    # random seeds
    seeds: [34243, 3232, 3223, 121, 121, 1212, 32, 111, 221, 5454],
//    seeds: [22],
    # eval hyperparams
    theta_vals: [1e-6, 1e-8, 1e-11],
    gammas: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1],

}