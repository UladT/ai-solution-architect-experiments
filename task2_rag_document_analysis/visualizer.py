"""
Visualizer - Task 2 Resume Intelligence Platform

Generates four publication-quality charts from extracted resume data:
  1. Category Distribution          — pie chart
  2. Top Skills Frequency           — horizontal bar chart
  3. Experience Level Distribution  — vertical bar chart
  4. Skills-by-Category Heatmap     — seaborn heatmap

All charts are saved as PNG files to results/charts/.

AC-1: Data extraction + visualization is a core functional requirement.
"""

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for headless runs)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from config import config


class Visualizer:
    """Renders and saves data visualizations from resume analysis results."""

    def __init__(self) -> None:
        os.makedirs(config.charts_dir, exist_ok=True)
        # Consistent style across all charts
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "#f8f9fa",
                "axes.grid": True,
                "grid.alpha": 0.4,
                "font.family": "DejaVu Sans",
                "font.size": 11,
            }
        )

    # ── 1. Category Distribution Pie Chart ───────────────────────────────────

    def plot_category_distribution(
        self, category_counts: Dict[str, int], title: str = "Resume Category Distribution"
    ) -> str:
        """
        Pie chart showing proportion of resumes per job category.

        Args:
            category_counts: {category_name: count} dict.
            title:           Chart title.

        Returns:
            Path to saved PNG file.
        """
        labels = list(category_counts.keys())
        sizes = list(category_counts.values())
        total = sum(sizes)

        colors = cm.Set3(np.linspace(0, 1, len(labels)))

        fig, ax = plt.subplots(figsize=(10, 7))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
            pctdistance=0.82,
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        )
        for atext in autotexts:
            atext.set_fontsize(9)
            atext.set_fontweight("bold")

        # Legend with counts
        legend_labels = [f"{lbl} ({cnt})" for lbl, cnt in zip(labels, sizes)]
        ax.legend(
            wedges,
            legend_labels,
            title="Category (count)",
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            fontsize=10,
        )
        ax.set_title(f"{title}\n(Total: {total} resumes)", fontsize=14, fontweight="bold", pad=20)
        fig.patch.set_facecolor("white")
        plt.tight_layout()

        path = os.path.join(config.charts_dir, "01_category_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── 2. Top Skills Frequency Bar Chart ────────────────────────────────────

    def plot_top_skills(
        self,
        skill_freq: Dict[str, int],
        top_n: int = 20,
        title: str = "Top Technical Skills Across All Resumes",
    ) -> str:
        """
        Horizontal bar chart of the most frequent skills.

        Args:
            skill_freq: {skill_name: frequency} dict.
            top_n:      Number of top skills to display.
            title:      Chart title.

        Returns:
            Path to saved PNG file.
        """
        if not skill_freq:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(
                0.5,
                0.5,
                "No skill data available",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )
            ax.axis("off")
            path = os.path.join(config.charts_dir, "02_top_skills.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return path

        # Take top_n and sort ascending (so highest bar is at top)
        top = dict(list(skill_freq.items())[:top_n])
        sorted_skills = sorted(top.items(), key=lambda x: x[1])
        skills = [s[0] for s in sorted_skills]
        counts = [s[1] for s in sorted_skills]

        cmap = cm.viridis(np.linspace(0.3, 0.9, len(skills)))

        fig, ax = plt.subplots(figsize=(12, max(6, len(skills) * 0.4)))
        bars = ax.barh(skills, counts, color=cmap, height=0.7, edgecolor="white")

        # Value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xlabel("Number of Resumes", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlim(0, max(counts) * 1.2)
        ax.tick_params(axis="y", labelsize=10)
        plt.tight_layout()

        path = os.path.join(config.charts_dir, "02_top_skills.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── 3. Experience Level Distribution Bar Chart ────────────────────────────

    def plot_experience_distribution(
        self,
        exp_dist: Dict[str, int],
        title: str = "Experience Level Distribution",
    ) -> str:
        """
        Vertical bar chart of candidates grouped by years of experience.

        Args:
            exp_dist: {"0–2 yrs": n, "3–5 yrs": n, ...} dict.
            title:    Chart title.

        Returns:
            Path to saved PNG file.
        """
        buckets = list(exp_dist.keys())
        counts = list(exp_dist.values())
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(buckets, counts, color=colors[: len(buckets)], width=0.55, edgecolor="white")

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(count),
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_ylabel("Number of Candidates", fontsize=12)
        ax.set_xlabel("Experience Range", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_ylim(0, max(counts) * 1.3 if counts else 5)
        plt.tight_layout()

        path = os.path.join(config.charts_dir, "03_experience_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── 4. Skills-by-Category Heatmap ─────────────────────────────────────────

    def plot_skills_heatmap(
        self,
        skills_by_category: Dict[str, List[str]],
        top_skills: int = 15,
        title: str = "Skills vs Category Heatmap",
    ) -> str:
        """
        Heatmap showing which categories use which top skills.

        Rows = categories, Columns = top global skills.
        Cell value = count of that skill's occurrences in that category's resumes.

        Args:
            skills_by_category: {category: [skill, skill, ...]} dict.
            top_skills:         Number of most-frequent skills to display.
            title:              Chart title.

        Returns:
            Path to saved PNG file.
        """
        # Build global top skills list
        all_freq: Dict[str, int] = {}
        for skill_list in skills_by_category.values():
            for skill in skill_list:
                all_freq[skill] = all_freq.get(skill, 0) + 1

        top_skill_names = [
            s for s, _ in sorted(all_freq.items(), key=lambda x: x[1], reverse=True)
        ][:top_skills]

        categories = list(skills_by_category.keys())
        if not categories or not top_skill_names:
            return ""

        # Build count matrix
        matrix = np.zeros((len(categories), len(top_skill_names)), dtype=int)
        for r, cat in enumerate(categories):
            cat_skills = skills_by_category[cat]
            for c, skill in enumerate(top_skill_names):
                matrix[r, c] = cat_skills.count(skill)

        fig, ax = plt.subplots(
            figsize=(max(10, len(top_skill_names) * 0.7), max(5, len(categories) * 0.6))
        )
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

        # Axes labels
        ax.set_xticks(np.arange(len(top_skill_names)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels(top_skill_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(categories, fontsize=10)

        # Cell annotations
        for r in range(len(categories)):
            for c in range(len(top_skill_names)):
                val = matrix[r, c]
                if val > 0:
                    ax.text(
                        c, r, str(val), ha="center", va="center",
                        fontsize=8, color="black" if val < matrix.max() * 0.7 else "white",
                    )

        plt.colorbar(im, ax=ax, shrink=0.8, label="Occurrences")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()

        path = os.path.join(config.charts_dir, "04_skills_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── Summary ────────────────────────────────────────────────────────────────

    def generate_all(
        self,
        category_counts: Dict[str, int],
        skill_freq: Dict[str, int],
        exp_dist: Dict[str, int],
        skills_by_category: Dict[str, List[str]],
    ) -> Dict[str, str]:
        """Generate all four charts and return a dict of {name: path}."""
        paths = {}
        paths["category_distribution"] = self.plot_category_distribution(category_counts)
        paths["top_skills"] = self.plot_top_skills(skill_freq)
        paths["experience_distribution"] = self.plot_experience_distribution(exp_dist)
        heatmap_path = self.plot_skills_heatmap(skills_by_category)
        if heatmap_path:
            paths["skills_heatmap"] = heatmap_path
        return paths
