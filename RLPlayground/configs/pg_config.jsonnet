local agent_types = ['REINFORCEAgent', 'A2CAgent'];
local env_names = ['CartPole-v0'];

{
    experiment_name: 'PGExperiment',
    agents: agent_types,
    env_names: env_names,
    experiment_cfg: {
        runs: 1,
        steps: 200,
        episodes: 2000,
        train_rng: 10,
        test_rng: 5,
        entropy_weights: [0.0001, 0.005, 0.001, 0.005, 0.01],
        lrs: [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.007, 0.01, 0.035, 0.05],
    },

    # specs for the agent for tabular environments
    agent_cfg: {
        'CartPole-v0': {
            env_name: 'CartPole-v0',
            average_score_to_solve: 195,
            consecutive_steps_to_solve: 100,
            REINFORCEAgent: {
                use_gpu: false,
                model: {
                    type: 'A2CLinearFCBody',
                    params: {
                        seed: 1337,
                        state_dim: 4,
                        action_dim: 2,
                        hidden_sizes: [64, 64],
                    },
                },
                gamma: 0.9,
                lr: 0.001,
            },
            A2CAgent: {
                use_gpu: false,
                model: {
                    type: 'A2CLinearFCBody',
                    params: {
                        seed: 1337,
                        state_dim: 4,
                        action_dim: 2,
                        hidden_sizes: [64, 64],
                    },
                },
                entropy_weight: 0.001,
                gamma: 0.9,
                lr: 0.001,
            },
            ACKTRAgent: {
                use_gpu: false,
                model: {
                    type: 'A2CLinearFCBody',
                    params: {
                        seed: 1337,
                        state_dim: 4,
                        action_dim: 2,
                        hidden_sizes: [64, 64],
                    },
                },
                entropy_weight: 0.001,
                value_loss_weight: 0.01,
                gamma: 0.9,
                lr: 0.001,
            },
        },
        'MountainCar-v0': {
            env_name: 'MountainCar-v0',
            average_score_to_solve: 195,
            consecutive_steps_to_solve: 100,
            A2CAgent: {
                use_gpu: false,
                model: {
                    type: 'A2CLinearFCBody',
                    params: {
                        seed: 1337,
                        state_dim: 4,
                        action_dim: 2,
                        hidden_sizes: [64, 64],
                    },
                },
                entropy_weight: 0.001,
                gamma: 0.9,
                lr: 0.001,
            },
            ACKTRAgent: {
                use_gpu: false,
                model: {
                    type: 'A2CLinearFCBody',
                    params: {
                        seed: 1337,
                        state_dim: 4,
                        action_dim: 2,
                        hidden_sizes: [64, 64],
                    },
                },
                entropy_weight: 0.001,
                value_loss_weight: 0.01,
                gamma: 0.9,
                lr: 0.001,
            },
        },
    },


}