# Inspired from https://github.com/raillab/dqn
import random
import numpy as np
import gym
from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
import torch
import argparse
import os
from Record import Record
#import ipdb


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DQN Atari')
    parser.add_argument('--load-checkpoint-file', type=str, default=None, 
                        help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')
    parser.add_argument('--random-seed', type=int, default=42, 
                        help='Where random seed should be inputted')
    parser.add_argument('--output-dir', type=str, default="", 
                        help='Where random seed should be inputted')
    parser.add_argument('-j', '--junk',
                        action='store_true')
    parser.add_argument('-r', '--random-choice',
                        action='store_true')
    parser.add_argument('-d', '--difficulty-test',
                        action='store_true')
    
    #Hyper parameters
    parser.add_argument('--learning-rate', type=float, default=1e-4, 
                        help='Input the learning rate')
    parser.add_argument('--num-steps', type=int, default=1e6, 
                        help='Total number of steps for training')
    parser.add_argument('--learning-starts', type=int, default=10000, 
                        help='Episode where learning starts')
    parser.add_argument('--batch-size', type=int, default=10, 
                        help='Batch sizes')

    args = parser.parse_args()
    # If you have a checkpoint file, spend less time exploring
    if(args.load_checkpoint_file):
        eps_start= 0.01
    else:
        eps_start= 1

    hyper_params = {
        "seed": args.random_seed,  # which seed to use
        "env": "SpaceInvadersNoFrameskip-v0",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": args.learning_rate,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "dqn_type":"neurips",
        # total number of steps to run the environment for
        "num-steps": args.num_steps,
        "batch-size": args.batch_size,  # number of transitions to optimize at the same time
        "learning-starts": args.learning_starts,  # number of steps before learning starts
        "learning-freq": 1,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": eps_start,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10,
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    env = gym.make(hyper_params["env"])
    

    difficulties = env.unwrapped.ale.getAvailableDifficulties()
    difficulties = [difficulties[0], difficulties[-1]]
    proportions = [0.5, 0.75] #Only need two, because the last one is until the end
    
    # Temporary difficulty change
    env.env.game_difficulty = difficulties[1]
    # 
    env.seed(hyper_params["seed"])

    outputPath = f'{args.output_dir}/seed{hyper_params["seed"]}' # folder we place data in
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = gym.wrappers.Monitor(
        env, f'{outputPath}/video/', video_callable=lambda episode_id: episode_id % 50 == 0, force=True)

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        use_double_dqn=hyper_params["use-double-dqn"],
        lr=hyper_params['learning-rate'],
        batch_size=hyper_params['batch-size'],
        gamma=hyper_params['discount-factor'],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dqn_type=hyper_params["dqn_type"]
    )
    
    # Set up recording
    Recording_oh=Record(42,args.output_dir)
    Recording_oh.add_activation_hook(agent.policy_network)
    
    if(args.load_checkpoint_file):
        print(f"Loading a policy - { args.load_checkpoint_file } ")
        agent.policy_network.load_state_dict(
            torch.load(args.load_checkpoint_file))

    eps_timesteps = hyper_params["eps-fraction"] * \
        float(hyper_params["num-steps"])

    state = env.reset()
    learning = True
    episode_rewards = [[0.0, 0, 0, env.env.game_difficulty, learning]] # The second number represents number of exploits, third number is total steps
    episode_counter=0
    for t in range(hyper_params["num-steps"]):
        episode_rewards[-1][2] += 1
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * \
            (hyper_params["eps-end"] - hyper_params["eps-start"])
        sample = random.random()

        if args.random_choice:
            action = env.action_space.sample()
        elif(sample > eps_threshold):
            # Exploit

            action = agent.act(state)
            episode_rewards[-1][1] += 1
        else:
            # Explore
            action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)
        # Recording_oh.recordObservation(next_state, episode_counter)

        # If this is junk, we should use a random number instead of the reward
        adjustedReward = reward
        if args.junk:
            adjustedReward = (random.random()*40)-20
        if learning:
            agent.memory.add(state, action, adjustedReward, next_state, float(done))
        state = next_state

        episode_rewards[-1][0] += reward
        if done:
   #         Recording_oh.grab_w_n_b(agent,episode_counter)
            episode_counter+=1
            
            """ For testing purposes only, please delete lines below """
      #      if episode_counter==50:
     #           Recording_oh.concat_w_n_b()
    #            ipdb.set_trace()
                
            if args.difficulty_test:
                # If the step is greater than 50% of total, change difficulty to hard
                # Do not learn
                if t > hyper_params["num-steps"] * proportions[0]:
                    env.env.game_difficulty = difficulties[1]
                    env.seed(hyper_params["seed"])
                    learning = False
                # If we are past the 75% point, start learning again
                if t > hyper_params["num-steps"] * proportions[1]:
                    learning = True
            state = env.reset()
            episode_rewards.append([0.0, 0, 0, env.env.game_difficulty, learning])

        if learning:
            if t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0:
                agent.optimise_td_loss()

            if t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0:
                agent.update_target_network()

        num_episodes = len(episode_rewards)

        if done and hyper_params["print-freq"] is not None and len(episode_rewards) % hyper_params[
                "print-freq"] == 0:
            mean_100ep_reward = round(np.mean([episode[0] for episode in episode_rewards[-101:-1]]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
            torch.save(agent.policy_network.state_dict(), f'{outputPath}/checkpoint_seed_{hyper_params["seed"]}.pth')
            np.savetxt(f'{outputPath}/rewards_per_episode_seed_{hyper_params["seed"]}_{hyper_params["learning-rate"]}_{hyper_params["batch-size"]}.csv', episode_rewards,
                       delimiter=',', fmt='[%1.3f, %s, %s, %s, %s]')
