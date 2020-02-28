local experience_replay_types = ['ExperienceReplay', 'PrioritizedExperienceReplay'];
local agent_types = ['DQNAgent', 'DeepExpectedSarsaAgent', 'DeepSarsaAgent'];
local env_names = ['CartPole-v0'];

{
    experiment_name: 'DeepTDExperiment',
    agents: agent_types,
    env_names: env_names,
    experiment_cfg:
    {
        runs: 1,
        steps: 300,
        episodes: 801,
        train_rng: 10,
        test_rng: 5,
        replay_buffer_capacities: [10000, 500, 250],
        batch_sizes: [32, 64, 128, 256],
        lrs: [0.01, 0.03, 0.08],
//        lrs: [0.01]
    },

    # specs for the agent for tabular environments
    agent_cfg:
    {
        env_name: 'CartPole-v0',
        average_score_to_solve: 195,
        consecutive_steps_to_solve: 100,
        DQNAgent: {
            use_gpu: true,
            use_double: false,
            experience_replay: {
                type: experience_replay_types[0],
                params: {
                    n_step: 4,
                    gamma: 0.9,
                    capacity: 200,
//                    alpha: 0.6,
//                    beta: 0.4,
//                    beta_inc: 12000, //~target network update frequency
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
            lr: 0.001,
            batch_size: 32,
            update_type: 'hard',
            tau: 0.01,
            update_freq: 12000,
            warm_up_freq: 2000
        },
        DeepSarsaAgent: {
            use_gpu: true,
//            use_double: true,
            experience_replay: {
                type: experience_replay_types[0],
                params: {
                    n_step: 4,
                    gamma: 0.9,
                    capacity: 200,
//                    alpha: 0.6,
//                    beta: 0.4,
//                    beta_inc: 12000, //~target network update frequency
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
            lr: 0.001,
            batch_size: 32,
            update_type: 'hard',
            tau: 0.01,
            update_freq: 12000,
            warm_up_freq: 500
        },
        DeepExpectedSarsaAgent: {
            use_gpu: true,
//            use_double: true,
            experience_replay: {
                type: experience_replay_types[0],
                params: {
                    n_step: 4,
                    gamma: 0.9,
                    capacity: 200,
//                    alpha: 0.6,
//                    beta: 0.4,
//                    beta_inc: 12000, //~target network update frequency
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
            lr: 0.001,
            batch_size: 32,
            update_type: 'hard',
            tau: 0.01,
            update_freq: 12000,
            warm_up_freq: 500
        },
    },
}