local experience_replay_types = ['ExperienceReplay', 'PrioritizedExperienceReplay'];
local agent_types = ['DQNAgent', 'DeepExpectedSarsaAgent', 'DeepSarsaAgent'];
local env_names = ['CartPole-v0'];

{
    experiment_name: 'DeepTDExperiment',
    agents: agent_types,
    env_names: env_names,
    experiment_cfg: {
        runs: 1,
        steps: 300,
        episodes: 200,
        train_rng: 10,
        test_rng: 5,
//        replay_buffer_capacities: [500, 250, 100, 50],
        replay_buffer_capacities: [20000, 10000, 5000, 500, 250, 100, 50],
        update_freqs: [10000, 20000, 50000],
        warm_up_freqs: [500, 1000],
        batch_sizes: [32, 64, 128],
//        lrs: [0.01, 0.025, 0.05],
//        lrs: [0.0001, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.05],
        lrs: [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.02, 0.035, 0.05],
        gammas: [0.9, 0.95, 0.99, 1.0],
        betas: [0.4, 0.5, 0.6],
        epsilons: [0.6, 0.8, 1.0],
        grad_clippings: [0.2, 0.6, 1.0],
    },

    # specs for the agent for tabular environments
    agent_cfg: {
        env_name: 'CartPole-v0',
        average_score_to_solve: 195,
        consecutive_steps_to_solve: 100,
        DQNAgent: {
            use_double: true,
            use_gpu: true,
            experience_replay: {
                type: experience_replay_types[1],
                params: {
                    n_step: 4,
                    gamma: 0.9,
                    capacity: 500,
                    non_zero_variant: 1e-6,
                    beta: 0.4,
                    beta_inc: 10000, //~target network update frequency
                },
            },
            model: {
                type: 'LinearFCBody',
                params: {
                    seed: 1337,
                    state_dim: 4,
                    action_dim: 2,
                    hidden_units: [64, 64],
                },
            },
            eps: 1,
            use_eps_decay: true,
            eps_decay: 0.999,
            eps_min: 0.1,
            gamma: 0.9,
            use_grad_clipping: false,
            grad_clipping: 1.0,
            lr: 0.01,
            batch_size: 32,
            update_type: 'hard',
            tau: 0.01, // soft update param
            update_freq: 10000,
            warm_up_freq: 400
        },
        DeepSarsaAgent: {
            use_gpu: true,
//            use_double: true,
            experience_replay: {
                type: experience_replay_types[1],
                params: {
                    n_step: 4,
                    gamma: 0.9,
                    capacity: 500,
                    non_zero_variant: 1e-6,
                    beta: 0.4,
                    beta_inc: 10000, //~target network update frequency
                },
            },
            model: {
                type: 'LinearFCBody',
                params: {
                    seed: 1337,
                    state_dim: 4,
                    action_dim: 2,
                    hidden_units: [64, 64],
                },
            },
            eps: 1,
            use_eps_decay: true,
            eps_decay: 0.999,
            eps_min: 0.1,
            gamma: 0.9,
            use_grad_clipping: false,
            grad_clipping: 1.0,
            lr: 0.01,
            batch_size: 32,
            update_type: 'hard',
            tau: 0.01, // soft update param
            update_freq: 10000,
            warm_up_freq: 400
        },
        DeepExpectedSarsaAgent: {
            use_gpu: true,
//            use_double: true,
            experience_replay: {
                type: experience_replay_types[1],
                params: {
                    n_step: 4,
                    gamma: 0.9,
                    capacity: 500,
                    non_zero_variant: 1e-6,
                    beta: 0.4,
                    beta_inc: 10000, //~target network update frequency
                },
            },
            model: {
                type: 'LinearFCBody',
                params: {
                    seed: 1337,
                    state_dim: 4,
                    action_dim: 2,
                    hidden_units: [64, 64],
                },
            },
            eps: 1,
            use_eps_decay: true,
            eps_decay: 0.999,
            eps_min: 0.1,
            gamma: 0.9,
            use_grad_clipping: false,
            grad_clipping: 1.0,
            lr: 0.01,
            batch_size: 32,
            update_type: 'hard',
            tau: 0.01, // soft update param
            update_freq: 10000,
            warm_up_freq: 400
        },
    },
}