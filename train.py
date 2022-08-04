import torch
from dataset import LowNormalDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(disc_L, disc_N, gen_N, gen_L, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    L_reals = 0
    L_fakes = 0
    N_reals = 0
    N_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (normal, low) in enumerate(loop):
        normal = normal.to(config.DEVICE)
        low = low.to(config.DEVICE)

        # Train Discriminators L and N
        with torch.cuda.amp.autocast():
            fake_low = gen_L(normal)
            D_L_real = disc_L(low)
            D_L_fake = disc_L(fake_low.detach())
            L_reals += D_L_real.mean().item()
            L_fakes += D_L_fake.mean().item()
            D_L_real_loss = mse(D_L_real, torch.ones_like(D_L_real))
            D_L_fake_loss = mse(D_L_fake, torch.zeros_like(D_L_fake))
            D_L_loss = D_L_real_loss + D_L_fake_loss

            fake_normal = gen_N(low)
            D_N_real = disc_N(normal)
            D_N_fake = disc_N(fake_normal.detach())
            N_reals += D_N_real.mean().item()
            N_fakes += D_N_fake.mean().item()
            D_N_real_loss = mse(D_N_real, torch.ones_like(D_N_real))
            D_N_fake_loss = mse(D_N_fake, torch.zeros_like(D_N_fake))
            D_N_loss = D_N_real_loss + D_N_fake_loss

            # put it together
            D_loss = (D_L_loss + D_N_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators L and N
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_L_fake = disc_L(fake_low)
            D_N_fake = disc_N(fake_normal)
            loss_G_L = mse(D_L_fake, torch.ones_like(D_L_fake))
            loss_G_N = mse(D_N_fake, torch.ones_like(D_N_fake))

            # cycle loss
            cycle_normal = gen_N(fake_low)
            cycle_low = gen_L(fake_normal)
            cycle_normal_loss = l1(normal, cycle_normal)
            cycle_low_loss = l1(low, cycle_low)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_normal = gen_N(normal)
            identity_low = gen_L(low)
            identity_normal_loss = l1(normal, identity_normal)
            identity_low_loss = l1(low, identity_low)

            # add all together
            G_loss = (
                    loss_G_N
                    + loss_G_L
                    + cycle_normal_loss * config.LAMBDA_CYCLE
                    + cycle_low_loss * config.LAMBDA_CYCLE
                    + identity_low_loss * config.LAMBDA_IDENTITY
                    + identity_normal_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx == 200:
            # save_image(fake_low * 0.5 + 0.5, f"gen_images/low_fake_{idx}.png")
            save_image(fake_normal * 0.5 + 0.5, f"gen_images/normal_fake_{idx}.png")

        loop.set_postfix(L_real=L_reals / (idx + 1), L_fake=L_fakes / (idx + 1),
                         N_real=N_reals / (idx + 1), N_fake=N_fakes / (idx + 1))


def main():
    disc_L = Discriminator(in_channels=3).to(config.DEVICE)
    disc_N = Discriminator(in_channels=3).to(config.DEVICE)
    gen_N = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_L = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_L.parameters()) + list(disc_N.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_N.parameters()) + list(gen_L.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_L, gen_L, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_N, gen_N, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_L, disc_L, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_N, disc_N, opt_disc, config.LEARNING_RATE,
        )

    dataset = LowNormalDataset(
        root_low=config.TRAIN_DIR + "/low",
        root_normal=config.TRAIN_DIR + "/normal",
        transform=config.transforms
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_L, disc_N, gen_N, gen_L, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_L, opt_gen, filename=config.CHECKPOINT_GEN_L)
            save_checkpoint(gen_N, opt_gen, filename=config.CHECKPOINT_GEN_N)
            save_checkpoint(disc_L, opt_disc, filename=config.CHECKPOINT_CRITIC_L)
            save_checkpoint(disc_N, opt_disc, filename=config.CHECKPOINT_CRITIC_N)


if __name__ == "__main__":
    main()
