import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


title_extension = ""


acc_y_bonds = [0, 100]
loss_y_bonds = [0, 200]

output_data_file = "gan_Wed_Jan_19_23_18_41_2022"

with open(f"training_summary/gan_train_loss_Wed_Jan_19_23_18_41_2022.json") as gan_train_loss:
    train_loss: list = json.load(gan_train_loss)

with open(f"training_summary/gan_val_acc_Wed_Jan_19_23_18_41_2022.json") as gan_val_acc:
    val_accuracy = json.load(gan_val_acc)

generator_loss = []
discriminator_loss = []
gen_losses = [gen_loss for [gen_loss, _, _] in train_loss]
disc_losses = [disc_loss for [_, disc_loss, _] in train_loss]
image_losses = [img_loss for [_, _, img_loss] in train_loss]

ds = pd.DataFrame({"Strata autoenkodera": gen_losses, "Strata dyskryminatora": disc_losses, "Strata obrazu": image_losses}, dtype=float)

# print(ds)

ds.plot(xlabel="epoka", ylabel="strata", title="Przebieg funkcji strat w procesie uczenia" +title_extension, ylim=loss_y_bonds)

ds.plot(logy=True, title="Przebieg funkcji strat w procesie uczenia" + title_extension, ylim=loss_y_bonds)

ds = pd.DataFrame({"Dokładność": val_accuracy}, dtype=float) * 100
ds.plot(xlabel="epoka", ylabel="dokładność [%]", title="Dokładność wyników uzyskanych na zbiorze walidacyjnym" + title_extension, legend=False, ylim=acc_y_bonds)

max_acc_index = val_accuracy.index(max(val_accuracy))
print(max_acc_index)
data = [["", f"Najlepsza predykcja [{max_acc_index}]", "Minimum", "Maximum", "Średnia", "Mediana"]]
sets = [gen_losses, disc_losses, image_losses, val_accuracy]
labels = ["Strata autoenkodera", "Strata dyskryminatora", "Strata obrazu", "Dokładność"]
for i in range(len(sets)):
    d_set = sets[i]
    d_set = np.array(d_set, dtype=float)
    data_row = [labels[i], d_set[max_acc_index], min(d_set), max(d_set), np.mean(d_set), np.median(d_set)]
    print(*data_row)
    data.append(data_row)

with open(f"training_summary/numbers/{output_data_file}.csv", "w") as data_file:
    data_file.writelines([";".join([str(value) for value in row]) + "\n" for row in data])

plt.tight_layout()
plt.show()
