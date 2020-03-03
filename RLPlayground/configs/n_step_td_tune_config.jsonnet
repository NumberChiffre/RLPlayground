local episodes = 1000;
local opt_func = 'cum_reward';
local scheduler = [
    'FIFO',
    'AsyncHyperBand',
    'HyperBand',
    'PBT'
];
local search_alg = [
    'AxSearch',
    'BasicVariantGenerator',
    'BayesOptSearch',
    'HyperOptSearch',
    'NevergradSearch',
    'SigOptSearch'
];
{
    stop: {
        track_reward_100: 195, # can import from un-decoupled config file..
        below_reward_thresh: true,
        training_iteration: episodes,
    },
    resources_per_trial: {
        cpu: 1,
        gpu: 0,
    },
    num_samples: 10,
    checkpoint_at_end: true,
    scheduler: {
        type: scheduler[1],
        kwargs: {
            reward_attr: opt_func,
            max_t: episodes,
        },
//        type: scheduler[3],
//        kwargs: {
//            time_attr: 'training_iteration',
//            reward_attr: opt_func,
////            perturbation_interval: 100,
//            hyperparam_mutations: {
//            },
////            resample_probability: 0.3,
//        },
    },
    search_alg: {
        type: search_alg[1],
        kwargs: {
        },
    }
}
