import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cols = 3
parameter = "refinement_deepest"
total = 4
xlim = [0, 50]

title_extension = " po pogłębieniu autoenkodera"
meteo = True


def subtitle(x):
    return ["strata autoenkodera", "strata dyskryminatora", "strata obrazu"][x]


acc_y_bonds = [0, 100]
x_bonds = [0, 50]

acc_fig, acc_axes = plt.subplots(nrows=1, ncols=1)
loss_fig, loss_axes = plt.subplots(nrows=1, ncols=cols)
# log_loss_fig, log_loss_axes = plt.subplots(nrows=rows, ncols=cols)

gen_losses = []
disc_losses = []
image_losses = []
val_accuracy = []
for j in range(total):
    with open(f"training_summary/{'meteo_' if meteo else ''}gan_train_loss_{parameter}_{j+1}_of_{total}.json") as gan_train_loss:
        train_loss: list = json.load(gan_train_loss)

    with open(f"training_summary/{'meteo_' if meteo else ''}gan_val_acc_{parameter}_{j+1}_of_{total}.json") as gan_val_acc:
        val_accuracy.append(json.load(gan_val_acc))

    gen_losses.append([float(gen_loss) for [gen_loss, _, _] in train_loss])
    disc_losses.append([float(disc_loss) for [_, disc_loss, _] in train_loss])
    image_losses.append([float(img_loss) for [_, _, img_loss] in train_loss])

if total > 4:
    for _ in range(total - 4):
        gen_losses.pop(), disc_losses.pop(), image_losses.pop()

ds_gen, ds_disc, ds_img = pd.DataFrame(gen_losses), pd.DataFrame(disc_losses), pd.DataFrame(image_losses)
# print(ds)
# ds_var.plot(logy=True, xlabel="epoka", ylabel="war. straty", title=subtitle(indexes[i]), ax=loss_axes[i, 1], legend=False, ylim=[10**-6, 10**6])
# ds.plot(logy=True, title=subtitle(indexes[i]), ax=log_loss_axes[i // cols, i % cols], legend=False, ylim=loss_y_bonds)

gen_mins, disc_mins, img_mins = ds_gen.min(0), ds_disc.min(0), ds_img.min(0)
gen_maxes, disc_maxes, img_maxes = ds_gen.max(0), ds_disc.max(0), ds_img.max(0)
gen_means, disc_means, img_means = ds_gen.mean(0), ds_disc.mean(0), ds_img.mean(0)

gen_means.plot(
    xlabel="epoka",
    yerr=np.array([(gen_means - gen_mins).to_numpy(), (gen_maxes - gen_means).to_numpy()]),
    ylabel="strata",
    title=subtitle(0),
    ax=loss_axes[0],
    legend=False,
    capsize=4,
    rot=0,
    ylim=[0, 1500],
    xlim=xlim
)
disc_means.plot(
    xlabel="epoka",
    yerr=np.array([(disc_means - disc_mins).to_numpy(), (disc_maxes - disc_means).to_numpy()]),
    ylabel="strata",
    title=subtitle(1),
    ax=loss_axes[1],
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
    ylabel="strata",
    title=subtitle(2),
    ax=loss_axes[2],
    legend=False,
    capsize=4,
    rot=0,
    ylim=[0, 200],
    xlim=xlim,
    color="green"
)

# gen_means.plot(xlabel="epoka", ylabel="strata autoenkodera", title=subtitle(indexes[i]), ax=loss_axes[i, 0], legend=False)
# disc_means.plot(xlabel="epoka", ylabel="strata dyskryminatora", title=subtitle(indexes[i]), ax=loss_axes[i, 1], legend=False)
# img_means.plot(xlabel="epoka", ylabel="strata obrazu", title=subtitle(indexes[i]), ax=loss_axes[i, 2], legend=False)

ds_acc = pd.DataFrame(val_accuracy, dtype=float) * 100
acc_mins = ds_acc.min(0)
acc_maxes = ds_acc.max(0)
acc_means = ds_acc.mean(0)
# ds_var = pd.DataFrame({"Dokładność": pd.DataFrame(val_accuracy, dtype=float).var()})
acc_means.plot(
    xlabel="epoka",
    yerr=[(acc_means - acc_mins).to_numpy(), (acc_maxes - acc_means).to_numpy()],
    ylabel="dokładność [%]",
    legend=False,
    ax=acc_axes,
    ylim=acc_y_bonds,
    capsize=4,
    rot=0,
    xlim=xlim,
)

# ds_var.plot(xlabel="epoka", ylabel="war. dokładności", title=subtitle(indexes[i]), legend=False, ax=acc_axes[i, 1], ylim=acc_y_bonds)

max_acc_index = [*acc_maxes.to_numpy()].index(max(acc_maxes.to_numpy()))
print(max_acc_index)
output_data = [
    ["", "Minimum", "Maximum", "Średnia", "Mediana"],
    ["Strata autoenkodera", min(gen_mins.to_numpy()), max(gen_maxes.to_numpy()), np.mean(gen_means.to_numpy()),
     np.median(ds_gen.median().to_numpy())],
    ["Strata dyskryminatora", min(disc_mins.to_numpy()), max(disc_maxes.to_numpy()), np.mean(disc_means.to_numpy()),
     np.median(ds_disc.median().to_numpy())],
    ["Strata obrazu", min(img_mins.to_numpy()), max(img_maxes.to_numpy()), np.mean(img_means.to_numpy()),
     np.median(ds_img.median().to_numpy())],
    ["Dokładność", min(acc_mins.to_numpy()), max(acc_maxes.to_numpy()), np.mean(acc_means.to_numpy()),
     np.median(ds_acc.median().to_numpy())]
]
with open(f"training_summary/numbers/{'meteo_' if meteo else ''}gan_{parameter}.csv", "w") as data_file:
    data_file.writelines([";".join([str(value) for value in data]) + "\n" for data in output_data])

    # plt.plot(gen_losses, disc_losses, image_losses)
loss_fig.suptitle(f"Średni przebieg funkcji strat w procesie uczenia na obrazach {'meteorologicznych' if meteo else 'rzeczywistych'}" + title_extension, fontsize="x-large", fontweight="bold")
acc_fig.suptitle(f"Średnia dokładność wyników uzyskanych na zbiorze walidacyjnym \n obrazów {'meteorologicznych' if meteo else 'rzeczywistych'}" + title_extension, fontweight="bold")
# log_loss_fig.suptitle("Przebieg funkcji strat w procesie uczenia" + title_extension, fontsize="x-large", fontweight="bold")

# log_loss_handles, log_loss_labels = log_loss_axes[0, 0].get_legend_handles_labels()
# log_loss_fig.legend(log_loss_handles, log_loss_labels, loc='lower center')

# if len(indexes) % 2 != 0:
#     acc_fig.delaxes(acc_axes[rows - 1][cols - 1])
#     loss_fig.delaxes(loss_axes[rows - 1][cols - 1])
#     # log_loss_fig.delaxes(log_loss_axes[rows - 1][cols - 1])

height = 10.5 / 3

for fig in [loss_fig]:
    fig.set_size_inches(12.5, height)
    fig.dpi = 100
    fig.subplots_adjust(
        top=0.795,
        bottom=0.168,
        left=0.061,
        right=0.976,
        hspace=0.341,
        wspace=0.182
    )

for fig in [acc_fig]:
    fig.set_size_inches(8, height*1.1)
    fig.dpi = 100
    fig.subplots_adjust(
        top=0.855,
        bottom=0.145,
        left=0.125,
        right=0.9,
        hspace=0.2,
        wspace=0.2
    )


plt.show()
