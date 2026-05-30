# Genetic Weights Visualizer

import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class WeightVisualizer:

    def __init__(self, root):

        self.root = root

        self.root.title("Genetic Network Visualizer")

        self.population = None

        self.current_agent = 0

        self.create_ui()

    # =====================================================
    # UI
    # =====================================================

    def create_ui(self):

        top_frame = tk.Frame(self.root)
        top_frame.pack(fill="x")

        load_button = tk.Button(
            top_frame,
            text="Load Population",
            command=self.load_population
        )

        load_button.pack(side="left", padx=5, pady=5)

        self.agent_label = tk.Label(
            top_frame,
            text="Agent: 0"
        )

        self.agent_label.pack(side="left", padx=10)

        prev_button = tk.Button(
            top_frame,
            text="< Prev",
            command=self.prev_agent
        )

        prev_button.pack(side="left", padx=5)

        next_button = tk.Button(
            top_frame,
            text="Next >",
            command=self.next_agent
        )

        next_button.pack(side="left", padx=5)

        self.info_label = tk.Label(
            self.root,
            text="No population loaded"
        )

        self.info_label.pack(pady=5)

        self.fig, self.axes = plt.subplots(
            2,
            2,
            figsize=(12, 8)
        )

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=self.root
        )

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    # =====================================================
    # LOAD
    # =====================================================

    def load_population(self):

        file_path = filedialog.askopenfilename(
            title="Select genetic_population.pt",
            filetypes=[("PyTorch", "*.pt")]
        )

        if not file_path:
            return

        data = torch.load(
            file_path,
            map_location="cpu"
        )

        self.population = data["population"]

        generation = data.get("generation", 0)

        self.info_label.config(
            text=f"Generation: {generation} | Population: {len(self.population)}"
        )

        self.current_agent = 0

        self.render_agent()

    # =====================================================
    # NAVIGATION
    # =====================================================

    def next_agent(self):

        if self.population is None:
            return

        self.current_agent += 1

        self.current_agent %= len(self.population)

        self.render_agent()

    def prev_agent(self):

        if self.population is None:
            return

        self.current_agent -= 1

        self.current_agent %= len(self.population)

        self.render_agent()

    # =====================================================
    # RENDER
    # =====================================================

    def render_agent(self):

        self.agent_label.config(
            text=f"Agent: {self.current_agent}"
        )

        agent = self.population[self.current_agent]

        keys = list(agent.keys())

        # полностью очищаем figure
        self.fig.clf()

        self.fig.suptitle(
            f"Agent {self.current_agent}",
            fontsize=16
        )

        weight_keys = [
            k for k in keys
            if "weight" in k
        ]

        plot_count = len(weight_keys)

        for i, key in enumerate(weight_keys):

            ax = self.fig.add_subplot(
                2,
                2,
                i + 1
            )

            weights = agent[key].numpy()

            im = ax.imshow(
                weights,
                aspect="auto"
            )

            ax.set_title(key)

            self.fig.colorbar(
                im,
                ax=ax
            )

        self.fig.tight_layout()

        self.canvas.draw()


# =========================================================
# START
# =========================================================

root = tk.Tk()

app = WeightVisualizer(root)

root.mainloop()
