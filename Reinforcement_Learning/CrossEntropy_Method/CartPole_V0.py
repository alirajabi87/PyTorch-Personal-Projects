import gym
import random
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)

        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)

        if done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            if len(batch) == batch_size:
                # random.shuffle(batch)
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    # print(list(map(lambda s: s.steps, batch)))
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))


    train_obs = []
    train_act = []

    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, directory="Monitor", force=True)
    obs_size = env.observation_space.shape[0]
    n_action = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_action)
    Criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env=env, net=net, batch_size=BATCH_SIZE)):

        obs_v, acts_v, reward_b, reward_m = filter_batch(batch=batch, percentile=PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)

        loss_v = Criterion(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        print(f"{iter_no}, loss= {loss_v.item():6.3f}, reward_mean= {reward_m:3.1f}, "
              f"reward_boundary= {reward_b:3.1f}")

        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        if reward_m > 195:
            print("solved")
            break
        writer.close()
    sm = nn.Softmax(dim=1)
    with torch.no_grad():
        iter_no = 0
        obs = env.reset()
        while True:
            env.render()
            obs = torch.FloatTensor([obs])
            action = sm(net(obs)).argmax(1).item()
            obs, reward, done, _ = env.step(action)
            iter_no += 1
            if iter_no == 1000:
                break
