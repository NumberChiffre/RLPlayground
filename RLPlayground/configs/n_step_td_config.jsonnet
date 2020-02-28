{
    experiment_name: 'DeepTDExperiment',
    agents: ['DQNAgent', 'DeepExpectedSarsaAgent', 'DeepSarsaAgent'],
    env_names: ['CartPole-v0'],

    # specs for the experiment
    experiment_cfg:
    {
        runs: 1,
        steps: 300,
        episodes: 2001,
        train_rng: 10,
        test_rng: 5,
        replay_buffer_capacities: [10000, 500, 250, 100, 50],
        batch_sizes: [32, 64, 128, 256],
//        lrs: [0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1],
        lrs: [0.01]
    },

    # specs for the agent for tabular environments
    agent_cfg:
    {
        env_name: 'CartPole-v0',
        average_score_to_solve: 195,
        consecutive_steps_to_solve: 100,
        DQNAgent: {
            eps: 1,
            use_eps_decay: true,
            eps_decay: 0.999,
            eps_min: 0.1,
            gamma: 0.9,
            n_step: 4,
            use_double: true,
            use_grad_clipping: false,
            replay_capacity: 200,
            nn_hidden_units: [64, 64],
            lr: 0.001,
            batch_size: 32,
            update_type: 'hard',
            tau: 0.01,
            update_freq: 12000,
            warm_up_freq: 500
        },
        DeepSarsaAgent: {
            eps: 1,
            use_eps_decay: true,
            eps_decay: 0.999,
            eps_min: 0.1,
            gamma: 0.9,
            n_step: 4,
            use_grad_clipping: false,
            replay_capacity: 200,
            nn_hidden_units: [64, 64],
            lr: 0.001,
            batch_size: 32,
            update_type: 'hard',
            tau: 0.01,
            update_freq: 12000,
            warm_up_freq: 500
        },
        DeepExpectedSarsaAgent: {
            eps: 1,
            use_eps_decay: true,
            eps_decay: 0.999,
            eps_min: 0.1,
            gamma: 0.9,
            n_step: 4,
            use_grad_clipping: false,
            replay_capacity: 200,
            nn_hidden_units: [64, 64],
            lr: 0.001,
            batch_size: 32,
            update_type: 'hard',
            tau: 0.01,
            update_freq: 12000,
            warm_up_freq: 500
        },

//        'DeepSarsaAgent':{
//            'eps': 1,
//            'eps_decay': 0.999,
//            'eps_min': 0.1,
//            'gamma': 0.9,
//            'n_step': 4,
////                'use_double': true,
//            'use_grad_clipping': false,
//            'replay_capacity': 200,
//            'nn_hidden_units': [64, 64],
//            'lr': 0.001,
//            'batch_size': 32,
//            'update_type': 'hard',
//            'tau': 0.01,
//            'update_freq': 12000,
//            'warm_up_freq': 500
//        }
    },

    # random seeds
//    seeds: [34243, 3232, 3223, 121, 121, 1212, 32, 111, 221, 5454],

//    seeds: [123, 12343],
    # eval hyperparams
    theta_vals: [1e-6, 1e-8, 1e-11],
    gammas: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1],

}