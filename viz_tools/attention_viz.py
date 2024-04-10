import os
import matplotlib.pyplot as plt

def plot_attention_flow(flow_matrix, token_labels, topk_prefix=15, savepdf=None, 
                        cbar_text=None,
                        title=None,
                        figsize=(3,2)):
    flow_matrix = flow_matrix[:topk_prefix]
    token_labels = token_labels[:topk_prefix]
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    h = ax.pcolor(
        flow_matrix,
        cmap="Blues",
        vmin=0,
    )
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(flow_matrix))])
    ax.set_xticks([0.5 + i for i in range(0, flow_matrix.shape[1] - 6, 5)])
    ax.set_xticklabels(list(range(0, flow_matrix.shape[1] - 6, 5)))
    ax.set_yticklabels(token_labels, fontsize=8)
    cb = plt.colorbar(h)
    ax.set_xlabel(f"Layers")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Attention contribution to generation")
    if cbar_text:
        cb.ax.set_title(cbar_text, y=-0.16, fontsize=8)

    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    return fig