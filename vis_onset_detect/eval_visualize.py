import matplotlib.pyplot as plt
import numpy as np

sync_score = np.load("syncability.json")["per_frame"]
onset_label = np.load("labels_framewise/2015-02-16-16-49-06_labels.npy")

plt.plot(sync_score, label="Synchronizability")
plt.plot(onset_label*max(sync_score), label="Onset Label", alpha=0.6)
plt.legend(); plt.title("Synchronizability vs Onset")
plt.show()
