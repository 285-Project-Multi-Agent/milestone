import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import tensorflow_probability as tfp

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    parser.add_argument("--num-agents", type=int, default=4, help="number of agents(total)")
    parser.add_argument("--landmarks", type=int, default=3, help="landmarks total")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="C:\\Users\\TheDonut\\Documents\\Berkeley\\285\\project\\maddpg\\experiments\\tmp\\policy", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="C:\\Users\\TheDonut\\Documents\\Berkeley\\285\\project\\maddpg\\experiments\\tmp\\policy", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")

    parser.add_argument("--multigrid", action="store_true", default=False)
    parser.add_argument("--save_display", action="store_true", default=False)
    parser.add_argument("--custom", '-c', action="store_true", default=False)
    parser.add_argument("--snum", type=int, default=0)

    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_bayesian_models(agents=6, imposters=1):
    #source: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py
    tfd = tfp.distributions
    models = []
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                tf.cast(arglist.max_episode_len, dtype=tf.float32))
    for i in range(agents - imposters):
        model = tf.keras.Sequential([
            tfp.layers.DenseFlipout(
                84, kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu, input_shape=(agents - 1,)),
            tfp.layers.DenseFlipout(
                agents - imposters, kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.softmax)
        ])
        optimizer = tf.keras.optimizers.Adam(lr=arglist.lr)
        model.compile(optimizer, loss='categorical_crossentropy',
                        metrics=['accuracy'], experimental_run_tf_function=False)
        models.append(model)
    return models


def make_env(scenario_name, arglist, benchmark=False):
    if arglist.multigrid:
        # # import gym_multigrid
        # import gym
        # from gym.envs.registration import register
        # register(
        #     id='multigrid-collect-v0',
        #     entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        # )
        # env = gym.make('multigrid-collect-v0')

        # _ = env.reset()
        # return env
        import gym
        import ma_gym
        from ma_gym.wrappers import Monitor

        if not arglist.custom:
            env = gym.make('AmongUs-v{}'.format(arglist.snum))
        else:
            gym.envs.register(
                id='MyAmongUs-v0',
                entry_point='ma_gym.envs.among_us:AmongUs',
                kwargs={'scenario': arglist.snum, 'n_imposter': arglist.num_adversaries, 'max_steps': arglist.max_episode_len}
            )
            env = gym.make('MyAmongUs-v0')

        # gym.envs.register(
        #     id='MyPredatorPrey5x5-v0',
        #     entry_point='ma_gym.envs.among_us:AmongUs',
        #     kwargs={'grid_shape': (10, 10), 'n_agents': 4, 'n_preys': 3}
        # )
        # env = gym.make('MyPredatorPrey5x5-v0')
        if arglist.save_display:
            env = Monitor(env, directory='recordings/{}'.format(arglist.exp_name), video_callable=lambda episode_id: (episode_id + 1)% arglist.save_rate == 0, force=True)
        env.n = env.n_agents
        return env

    else:
        from multiagent.environment import MultiAgentEnv
        import multiagent.scenarios as scenarios

        # load scenario from script
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        if benchmark:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
        else:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

        if arglist.save_display:
            import ma_gym
            from ma_gym.wrappers import Monitor
            env.n_agents = env.n
            env = Monitor(env, directory='recordings/{}'.format(arglist.exp_name), video_callable=lambda episode_id: episode_id%100==0)
        return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        # if arglist.display or arglist.restore or arglist.benchmark:
        #     print('Loading previous state...')
        #     U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        crewmates = arglist.num_agents - arglist.num_adversaries
        bayesian = np.array(make_bayesian_models(arglist.num_agents, arglist.num_adversaries))
        bayesian_frequency = 70
        num_monte_carlo = 50
        bayesian_losses = [0 for i in range(crewmates)]
        bayesian_accuracies = [0 for i in range(crewmates)]
        guesses = []
        storage = []
        imposters = [1 if i < arglist.num_adversaries else 0 for i in range(arglist.num_agents - 1)]
        labels = np.array(imposters).reshape((1, crewmates)).repeat(crewmates, 0)
        batch_x = []
        batch_y = []
        shuffle = []
        threshold = 0.0
        correct = 0
        total = 0

        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            #bayesian training:
            if (episode_step + 1) % bayesian_frequency == 0 and (len(episode_rewards) + 1 % 10):
                inp = np.asarray(batch_x)
                lab = np.asarray(batch_y)
                for i in range(crewmates):
                    bayesian_losses[i], bayesian_accuracies[i] = bayesian[i].train_on_batch(tf.cast(inp[:,i::crewmates,:].reshape((inp.shape[0], inp.shape[2])), dtype=tf.float32), tf.cast(lab[:,i::crewmates,:].reshape((lab.shape[0], lab.shape[2])), dtype=tf.float32))
                storage.append([bayesian_losses, bayesian_accuracies])
                # guesses.append([[[], False, 0, 0] for i in range(crewmates)])
                votes = [0 for i in range(arglist.num_agents)]
                for i in range(crewmates):
                    data = np.array([info_n['moves_obs'][i]])[:,shuffle]
                    probs = np.mean([bayesian[i].predict(data) for _ in range(num_monte_carlo)], axis=0)
                    best = shuffle[np.argmax(probs)]
                    # guesses[-1][i][0].append(probs)
                    # guesses[-1][i][1] = best == 1
                    # guesses[-1][i][2] = probs.max()
                    # guesses[-1][i][3] = episode_step
                    if best > i:
                      best += 1
                    if np.argmax(probs) > threshold:
                      votes[best] += 1
                if max(votes) > len(votes) / 2:
                  env.correct_vote = 1 if np.argmax(votes) == 0 else -1
                batch_x = []
                batch_y = []
                shuffle = np.random.permutation(data.shape[1])
            data = np.asarray(info_n['moves_obs'])
            if len(shuffle) == 0:
              shuffle = np.random.permutation(data.shape[1])
            batch_x.append(data[:,shuffle])
            batch_y.append(labels[:,shuffle])

            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:

                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if (terminal or done) and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir + "/{}/".format(arglist.exp_name), saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
                np.save("./guesses.npy", guesses)
                np.save("./storage.npy", storage)
            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                print("{} correct out of {}".format(correct,total))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
