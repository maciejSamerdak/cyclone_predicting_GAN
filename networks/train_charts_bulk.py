import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cols = 2
rows = 2
parameter = "size"
indexes = [32, 64, 128]

title_extension = " wg rozdzielczości obrazów"

meteo = False

def subtitle(x):
    return f"{x}px x {x}px"


acc_y_bonds = [0, 100]
loss_y_bonds = [0, 2000]

acc_fig, acc_axes = plt.subplots(nrows=rows, ncols=cols)
loss_fig, loss_axes = plt.subplots(nrows=rows, ncols=cols)
log_loss_fig, log_loss_axes = plt.subplots(nrows=rows, ncols=cols)

for i in range(len(indexes)):

    with open(f"training_summary/{'meteo_' if meteo else ''}gan_train_loss_{parameter}_{indexes[i]}.json") as gan_train_loss:
        train_loss: list = json.load(gan_train_loss)

    with open(f"training_summary/{'meteo_' if meteo else ''}gan_val_acc_{parameter}_{indexes[i]}.json") as gan_val_acc:
        val_accuracy = json.load(gan_val_acc)

    generator_loss = []
    discriminator_loss = []
    gen_losses = [gen_loss for [gen_loss, _, _] in train_loss]
    disc_losses = [disc_loss for [_, disc_loss, _] in train_loss]
    image_losses = [img_loss for [_, _, img_loss] in train_loss]

    ds = pd.DataFrame({"Strata autoenkodera": gen_losses, "Strata dyskryminatora": disc_losses, "Strata obrazu": image_losses}, dtype=float)

    # print(ds)

    ds.plot(xlabel="epoka", ylabel="strata", title=subtitle(indexes[i]), ax=loss_axes[i // cols, i % cols], legend=False, ylim=loss_y_bonds)

    ds.plot(logy=True, title=subtitle(indexes[i]), ax=log_loss_axes[i // cols, i % cols], legend=False, ylim=loss_y_bonds)

    ds = pd.DataFrame({"Dokładność": val_accuracy}, dtype=float) * 100
    ds.plot(xlabel="epoka", ylabel="dokładność [%]", title=subtitle(indexes[i]), legend=False, ax=acc_axes[i // cols, i % cols], ylim=acc_y_bonds)

    max_acc_index = val_accuracy.index(max(val_accuracy))
    print(max_acc_index)
    output_data = [["", f"Najlepsza predykcja [{max_acc_index}]", "Minimum", "Maximum", "Średnia", "Mediana"]]
    sets = [gen_losses, disc_losses, image_losses, val_accuracy]
    labels = ["Strata autoenkodera", "Strata dyskryminatora", "Strata obrazu", "Dokładność"]
    for j in range(len(sets)):
        d_set = sets[j]
        d_set = np.array(d_set, dtype=float)
        data_row = [labels[j], d_set[max_acc_index], min(d_set), max(d_set), np.mean(d_set), np.median(d_set)]
        print(*data_row)
        output_data.append(data_row)

    with open(f"training_summary/numbers/{'meteo_' if meteo else ''}gan_{parameter}_{indexes[i]}.csv",
              "w") as data_file:
        data_file.writelines([";".join([str(value) for value in data]) + "\n" for data in output_data])

    # plt.plot(gen_losses, disc_losses, image_losses)
loss_fig.suptitle("Przebieg funkcji strat w procesie uczenia" + title_extension, fontsize="x-large", fontweight="bold")
log_loss_fig.suptitle("Przebieg funkcji strat w procesie uczenia" + title_extension, fontsize="x-large", fontweight="bold")
acc_fig.suptitle("Dokładność wyników uzyskanych na zbiorze walidacyjnym" + title_extension, fontsize="x-large", fontweight="bold")

loss_handles, loss_labels = loss_axes[0, 0].get_legend_handles_labels()
loss_fig.legend(loss_handles, loss_labels, loc='lower center')
log_loss_handles, log_loss_labels = log_loss_axes[0, 0].get_legend_handles_labels()
log_loss_fig.legend(log_loss_handles, log_loss_labels, loc='lower center')

if len(indexes) % 2 != 0:
    acc_fig.delaxes(acc_axes[rows - 1][cols - 1])
    loss_fig.delaxes(loss_axes[rows - 1][cols - 1])
    log_loss_fig.delaxes(log_loss_axes[rows - 1][cols - 1])

height = 10.5 / 3 * rows

for fig in [acc_fig, loss_fig, log_loss_fig]:
    fig.set_size_inches(10.5, height)
    fig.subplots_adjust(
        top=0.88,
        bottom=0.11,
        left=0.085,
        right=0.94,
        hspace=0.4,
        wspace=0.22
    )

plt.tight_layout()
plt.show()
