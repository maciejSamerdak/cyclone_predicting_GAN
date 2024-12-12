import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cols = 2
parameter = ""
indexes = [32, 64, 128]
rows = len(indexes)
total = 4
xlim = [0, 50]

title_extension = " wg rozmiaru obrazu"
meteo = False


def subtitle(x):
    return f"{x} x {x}"


acc_y_bonds = [0, 100]
x_bonds = [0, 50]

acc_fig, acc_axes = plt.subplots(nrows=rows, ncols=1)
loss_fig, loss_axes = plt.subplots(nrows=rows, ncols=cols)

for i in range(len(indexes)):
    gen_losses = []
    disc_losses = []
    image_losses = []
    val_accuracy = []
    output_data = []
    for j in range(total):
        with open(f"training_summary/{'meteo_' if meteo else ''}gan_train_loss_{parameter}_{indexes[i]}_{j+1}_of_{total}.json") as gan_train_loss:
            train_loss: list = json.load(gan_train_loss)

        with open(f"training_summary/{'meteo_' if meteo else ''}gan_val_acc_{parameter}_{indexes[i]}_{j+1}_of_{total}.json") as gan_val_acc:
            val_accuracy.append(json.load(gan_val_acc))

        gen_losses.append([float(gen_loss) for [gen_loss, _, _] in train_loss])
        disc_losses.append([float(disc_loss) for [_, disc_loss, _] in train_loss])
        image_losses.append([float(img_loss) for [_, _, img_loss] in train_loss])

    ds_gen, ds_disc, ds_img = pd.DataFrame(gen_losses), pd.DataFrame(disc_losses), pd.DataFrame(image_losses)

    gen_mins, disc_mins, img_mins = ds_gen.min(0), ds_disc.min(0), ds_img.min(0)
    gen_maxes, disc_maxes, img_maxes = ds_gen.max(0), ds_disc.max(0), ds_img.max(0)
    gen_means, disc_means, img_means = ds_gen.mean(0), ds_disc.mean(0), ds_img.mean(0)

    gen_means.plot(
        xlabel="epoka",
        yerr=np.array([(gen_means - gen_mins).to_numpy(), (gen_maxes - gen_means).to_numpy()]),
        ylabel="strata autoenkodera",
        title=subtitle(indexes[i]),
        ax=loss_axes[i, 0],
        legend=False,
        capsize=4,
        rot=0,
        ylim=[0, 1500],
        xlim=xlim
    )
    disc_means.plot(
        xlabel="epoka",
        yerr=np.array([(disc_means - disc_mins).to_numpy(), (disc_maxes - disc_means).to_numpy()]),
        ylabel="strata dyskryminatora",
        title=subtitle(indexes[i]),
        ax=loss_axes[i, 1],
        legend=False,
        capsize=4,
        rot=0,
        ylim=[0, 50],
        xlim=xlim,
        color="orange"
    )
    img_means.plot(
        xlabel="epoka",
        yerr=np.array([(img_means - img_mins).to_numpy(), (img_maxes - img_means).to_numpy()]),
        ylabel="strata obrazu",
        title=subtitle(indexes[i]),
        ax=loss_axes[i, 2],
        legend=False,
        capsize=4,
        rot=0,
        ylim=[0, 200],
        xlim=xlim,
        color="green"
    )

    ds_acc = pd.DataFrame(val_accuracy, dtype=float) * 100
    acc_mins = ds_acc.min(0)
    acc_maxes = ds_acc.max(0)
    acc_means = ds_acc.mean(0)
    acc_means.plot(
        xlabel="epoka",
        yerr=[(acc_means - acc_mins).to_numpy(), (acc_maxes - acc_means).to_numpy()],
        ylabel="dokładność [%]",
        title=subtitle(indexes[i]),
        legend=False,
        ax=acc_axes[i],
        ylim=acc_y_bonds,
        capsize=4,
        rot=0,
        xlim=xlim,
    )

    print(subtitle(i))
    max_acc_index = [*acc_maxes.to_numpy()].index(max(acc_maxes.to_numpy()))
    print(max_acc_index)
    output_data = [
        ["", "Minimum", "Maximum", "Średnia", "Mediana"],
        ["Strata autoenkodera", min(gen_mins.to_numpy()), max(gen_maxes.to_numpy()), np.mean(gen_means.to_numpy()), np.median(ds_gen.median().to_numpy())],
        ["Strata dyskryminatora", min(disc_mins.to_numpy()), max(disc_maxes.to_numpy()), np.mean(disc_means.to_numpy()), np.median(ds_disc.median().to_numpy())],
        ["Strata obrazu", min(img_mins.to_numpy()), max(img_maxes.to_numpy()), np.mean(img_means.to_numpy()), np.median(ds_img.median().to_numpy())],
        ["Dokładność", min(acc_mins.to_numpy()), max(acc_maxes.to_numpy()), np.mean(acc_means.to_numpy()), np.median(ds_acc.median().to_numpy())]
    ]
    with open(f"training_summary/numbers/{'meteo_' if meteo else ''}gan_{parameter}_{indexes[i]}.csv", "w") as data_file:
        data_file.writelines([";".join([str(value) for value in data]) + "\n" for data in output_data])

loss_fig.suptitle(f"Średni przebieg funkcji strat w procesie uczenia na obrazach {'meteorologicznych' if meteo else 'rzeczywistych'}" + title_extension, fontsize="x-large", fontweight="bold")
acc_fig.suptitle(f"Średnia dokładność wyników uzyskanych na zbiorze walidacyjnym obrazów {'meteorologicznych' if meteo else 'rzeczywistych'}" + title_extension, fontsize="x-large", fontweight="bold")

height = 10.5 / 3 * rows

for fig in [acc_fig, loss_fig]:
    fig.set_size_inches(14.5, height)
    fig.dpi = 60
    fig.subplots_adjust(
        top=0.92,
        bottom=0.033,
        left=0.056,
        right=0.981,
        hspace=0.341,
        wspace=0.182
    )


plt.show()
