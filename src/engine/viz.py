from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def draw_circuit_1q(gate_names: list[str], label: str = "q0"):
    n = len(gate_names)
    fig_w = max(6, 1.2 + 0.9 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 2.2))

    y = 0.5
    ax.plot([0, n + 1], [y, y], linewidth=2)
    ax.text(-0.2, y, label, va="center", ha="right", fontsize=12)

    for i, name in enumerate(gate_names, start=1):
        x = i
        w, h = 0.6, 0.35
        ax.add_patch(plt.Rectangle((x - w/2, y - h/2), w, h, fill=True,facecolor="white",edgecolor="black" ,linewidth=2 , zorder=3))
        ax.text(x, y, name, va="center", ha="center", fontsize=12, zorder=4)

    ax.set_xlim(-0.6, n + 1.2)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Circuit")
    plt.tight_layout()
    return fig

def plot_probs(p0: float, p1: float):
    fig = plt.figure()
    plt.bar(["P(0)", "P(1)"], [p0, p1])
    plt.ylim(0, 1)
    plt.title("Z-basis measurement probabilities")
    plt.tight_layout()
    return fig

def plot_bloch_point(x: float, y: float, z: float):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, linewidth=0.3)


    ax.plot([-1,1],[0,0],[0,0] , linewidth=1 ,color="Gray")
    ax.plot([0,0],[-1,1],[0,0] , linewidth=1 ,color="Gray")
    ax.plot([0,0],[0,0],[-1,1] , linewidth=1 ,color="Gray")

    ax.text(1.1 , 0 , 0 ,"x" , fontsize=12)
    ax.text(0 , 1.1 , 0 ,"y" , fontsize=12)
    ax.text(0 , 0 , 1.1 ,"z" , fontsize=12)




    ax.scatter([x], [y], [z], s=70)
    ax.plot([0,x],[0,y],[0,z],linewidth=2)
    
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Bloch sphere")
    plt.tight_layout()
    return fig
