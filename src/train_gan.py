from models import Generator, Discriminator, weights_init
from dataloader import get_images, get_dataloader, get_images_2

import ast
import os
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam


def train(train_path, img_size, batch_size, lr, epochs, nz, pickled_folder, train_img_names=None):
    if train_img_names:
        images = get_images_2(train_path, train_img_names, img_size)
    else:
        images = get_images(train_path, img_size)
    train_loader = get_dataloader(images, batch_size)

    disc = Discriminator(ngpu=0)
    gen = Generator(ngpu=0)
    disc.apply(weights_init)
    gen.apply(weights_init)

    loss_fn = nn.BCELoss()
    optimizer_disc = Adam(disc.parameters(), lr=lr)
    optimizer_gen = Adam(gen.parameters(), lr=lr)

    epoch_disc_loss = []
    epoch_gen_loss = []
    epoch_disc_acc = []
    epoch_gen_acc = []
    generated_imgs = []
    best_gen_sd = {}
    lowest_gen_loss = 999
    best_epoch = 0

    for epoch in range(epochs):
        print(f">>> EPOCH {epoch+1} / {epochs}")

        batch_disc_loss = []
        batch_gen_loss = []
        batch_disc_acc = []
        batch_gen_acc = []

        for batch, input in enumerate(train_loader):
            batch += 1
            if batch % 5 == 0:
                print(f"Training Batch {batch} / {len(train_loader)}")

            input = input[0]
            bs = input.size(0)

            # train discriminator
            disc.zero_grad()
            output_disc = disc(input).flatten()
            labels = torch.ones(bs)
            loss_disc = loss_fn(output_disc, labels)
            loss_disc.backward()

            batch_disc_loss.append(loss_disc.item())
            batch_disc_acc.append(acc_fn(output_disc, labels))

            noise = torch.randn(bs, nz, 1, 1)
            fake_input = gen(noise)
            output_disc_2 = disc(fake_input.detach()).flatten()
            labels_2 = torch.zeros(bs)
            loss_disc_2 = loss_fn(output_disc_2, labels_2)
            loss_disc_2.backward()
            
            batch_disc_loss.append(loss_disc_2.item())
            batch_disc_acc.append(acc_fn(output_disc_2, labels_2))

            optimizer_disc.step()

            # train generator
            gen.zero_grad()
            output_disc_3 = disc(fake_input).flatten()
            loss_gen = loss_fn(output_disc_3, labels)  # use real label, 1
            loss_gen.backward()
    
            batch_gen_loss.append(loss_gen.item())
            batch_gen_acc.append(acc_fn(output_disc_3, labels))

            optimizer_gen.step()
    
        epoch_disc_loss.append(sum(batch_disc_loss) / len(batch_disc_loss))
        epoch_disc_acc.append(sum(batch_disc_acc) / len(batch_disc_acc))
        epoch_gen_loss.append(sum(batch_gen_loss) / len(batch_gen_loss))
        epoch_gen_acc.append(sum(batch_gen_acc) / len(batch_gen_acc))
        if epoch % 10 == 0:
            generated_imgs.append(gen(torch.randn(64, nz, 1, 1)).detach())
        if epoch_gen_loss[-1] < lowest_gen_loss and epoch >= 5:
            lowest_gen_loss = epoch_gen_loss[-1]
            best_gen_sd = gen.state_dict()
            best_epoch = epoch

        print("STATS:")
        print(f"discriminator loss = {epoch_disc_loss[-1]}; acc = {epoch_disc_acc[-1]}")
        print(f"generator loss = {epoch_gen_loss[-1]}; acc = {epoch_gen_acc[-1]}")
        print()
    
    file = open(os.path.join(pickled_folder, "generated_imgs"), "ab")
    pickle.dump(generated_imgs, file)
    file.close()
    file = open(os.path.join(pickled_folder, "epoch_disc_loss"), "ab")
    pickle.dump(epoch_disc_loss, file)
    file.close()
    file = open(os.path.join(pickled_folder, "epoch_gen_loss"), "ab")
    pickle.dump(epoch_gen_loss, file)
    file.close()
    torch.save(best_gen_sd, os.path.join(pickled_folder, "best_gen.pth"))
    print(f"Saved model from epoch {best_epoch} with loss {lowest_gen_loss}")
    torch.save(gen.state_dict(), os.path.join(pickled_folder, "last_gen.pth"))


def acc_fn(preds, labels):
    preds = preds.round()
    corr = (preds == labels).float()
    acc = corr.sum() / len(corr)
    return acc.item()


if __name__ == "__main__":
    train_path = "data/roadsigns/images"
    img_size = (64, 64)
    batch_size = 2
    lr = 1e-4
    epochs = 1000
    nz = 100
    with open("data/roadsigns/cw_train_imgs.txt", "r") as f:
        txt = f.read()
        train_img_names = ast.literal_eval(txt)
    pickled_folder = "src/pickled_road"

    train(train_path, img_size, batch_size, lr, epochs, nz, pickled_folder, train_img_names)
