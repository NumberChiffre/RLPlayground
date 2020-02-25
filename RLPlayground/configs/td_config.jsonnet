{
    experiment_name: 'n_step_td',

    # dp methods
    algos: ['expected_sarsa', 'q_learning', 'sarsa'],

    # environments
    env_names: ['CartPole-v0'],

    # specs for the experiment
    experiment_cfg:
    {
        'runs': 1,
        'steps': 5000,
        'episodes': 60,
        'train_rng': 10,
        'test_rng': 5,
        'replay_buffer_capacities': [50, 100, 250, 500],
        'batch_sizes': [32, 64, 128, 256],
    },

    # specs for the agent for tabular environments
    agent_cfg:
    {
        'CartPole-v0':
        {
            'q_learning':{
                'eps': 0.1,
                'gamma': 0.9,
                'n_step': 4,
                'algo': 'q_learning',
                'replay_capacity': 200,
                'nn_hidden_units': [64, 64],
                'lr': 0.001,
                'batch_size': 32,
                'update_freq': 10,
            },
            'sarsa':{
                'eps': 0.1,
                'gamma': 0.9,
                'n_step': 4,
                'algo': 'sarsa',
                'replay_capacity': 200,
                'nn_hidden_units': [64, 64],
                'lr': 0.001,
                'batch_size': 32,
                'update_freq': 10,
            },
            'expected_sarsa':{
                'eps': 0.1,
                'gamma': 0.9,
                'n_step': 4,
                'algo': 'expected_sarsa',
                'replay_capacity': 200,
                'nn_hidden_units': [64, 64],
                'lr': 0.01,
                'batch_size': 32,
                'update_freq': 500,
            }
        }
    },

    # random seeds
//    seeds: [34243, 3232, 3223, 121, 121, 1212, 32, 111, 221, 5454],

//    seeds: [123, 12343],
    # eval hyperparams
    theta_vals: [1e-6, 1e-8, 1e-11],
    gammas: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1],

}