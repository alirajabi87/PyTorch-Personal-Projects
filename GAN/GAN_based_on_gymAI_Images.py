import random
import argparse
import cv2 as cv

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import torchvision.utils as vutils

import gym, gym.spaces

import numpy as np

log = gym.logger
log.set_level(gym.logger.INFO)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENR_FILTERS = 64
BATCH_SIZE = 16

IMAGE_SIZE = 64
LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000


class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of the input numpy array
    convert input image tensor from (H, W, Chan) --> (Chan, H, W)
    """

    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32,
        )

    def observation(self, observation):
        new_obs = cv.resize(
            observation, (IMAGE_SIZE, IMAGE_SIZE))
        new_obs = np.transpose(new_obs, (2, 0, 1))
        # new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.conv_pip = nn.Sequential(
            nn.Conv2d(input_shape[0], DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            self.convBlock(DISCR_FILTERS, DISCR_FILTERS * 2, kernel_size=4, stride=2, padding=1),
            self.convBlock(DISCR_FILTERS * 2, DISCR_FILTERS * 4, kernel_size=4, stride=2, padding=1),
            self.convBlock(DISCR_FILTERS * 4, DISCR_FILTERS * 8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels= DISCR_FILTERS * 8, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pip(x)
        return conv_out.view(-1, 1).squeeze(dim=1)

    def convBlock(self, in_c, out_c, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, *args, **kwargs),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        Gen_size = [GENR_FILTERS * 8, GENR_FILTERS * 4, GENR_FILTERS * 2, GENR_FILTERS * 1]

        body = [self.convBlock(in_c, out_c, kernel_size=4, stride=2, padding=1)
                for in_c, out_c in zip(Gen_size, Gen_size[1:])]

        self.pip = nn.Sequential(
            self.convBlock(LATENT_VECTOR_SIZE, GENR_FILTERS * 8, kernel_size=4, stride=1, padding=0),
            *body,
            nn.ConvTranspose2d(in_channels=GENR_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pip(x)

    def convBlock(self, in_c, out_c, *args, **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, *args, **kwargs),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = [e.reset() for e in envs]
    for i in range(len(batch)):
        print(np.array(batch[i]).shape)
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        obs, reward, done, _ = e.step(e.action_space.sample())

        if np.mean(obs) > 0.01:
            batch.append(obs)

        if len(batch) == batch_size:
            # Normalize input between -1, 1

            # batch_up = batch * 2.0 / 255.0 - 1.0
            # yield torch.FloatTensor(batch_up)
            # batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            batch_np = [batch[i]* 2.0 / 255.0 - 1.0 for i in range(len(batch))]
            batch_np = np.stack(batch_np, axis=0)
            # batch_np = np.array(batch, dtype=np.float32)*2.0/255.0-1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if done:
            e.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action='store_true',
        help="Enable cuda computation"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    envs = [
        InputWrapper(gym.make(name))
        for name in ('Breakout-v0', 'Atlantis-v0', 'Pong-v0') #
    ]
    input_shape = envs[0].observation_space.shape
    # print(input_shape)

    model_disc = Discriminator(input_shape).to(device)
    model_Gen = Generator(input_shape).to(device)

    print(model_disc)
    print(model_Gen)

    Criterion = nn.BCELoss()
    Gen_optimizer = optim.Adam(model_Gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    Discr_optmizer = optim.Adam(model_disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    writer = SummaryWriter()

    gen_losses = []
    discr_losses = []
    iter_no = 0

    true_label_v = torch.ones(BATCH_SIZE, device=device)
    fake_label_v = torch.zeros(BATCH_SIZE, device=device)

    for batch_v in iterate_batches(envs):
        gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1)

        batch_v = batch_v.to(device)
        gen_input_v = gen_input_v.to(device)

        gen_output = model_Gen(gen_input_v)

        # Train Discriminator

        Discr_optmizer.zero_grad()
        Discr_output_true_v = model_disc(batch_v)
        Discr_output_fake_v = model_disc(gen_output.detach())

        discr_loss = Criterion(Discr_output_true_v, true_label_v) + \
                     Criterion(Discr_output_fake_v, fake_label_v)

        discr_loss.backward()
        Discr_optmizer.step()

        discr_losses.append(discr_loss.item())

        # train generator

        Gen_optimizer.zero_grad()
        dis_out_v = model_disc(gen_output)
        gen_loss = Criterion(dis_out_v, true_label_v)
        gen_loss.backward()
        Gen_optimizer.step()

        gen_losses.append(gen_loss.item())

        iter_no += 1

        if not iter_no % REPORT_EVERY_ITER:
            log.info(f"Iter: {iter_no}, gen_loss: {np.mean(gen_losses): 6.3f}, dis_loss: {np.mean(discr_losses):6.3f}")
            writer.add_scalar(
                "gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar(
                "dis_loss", np.mean(discr_losses), iter_no)
            gen_losses = []
            discr_losses = []

        if not iter_no % SAVE_IMAGE_EVERY_ITER:
           writer.add_image("fake", vutils.make_grid(
               gen_output.data[:64], normalize=True), iter_no)
           writer.add_image("true", vutils.make_grid(
               batch_v.data[:64], normalize=True), iter_no)