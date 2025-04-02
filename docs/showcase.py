from pathlib import Path
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from matplotlib.colors import to_rgb
import peppr

# Colors are based on Lab color space at L = 50
RED = to_rgb("#db3a35")
BLUE = to_rgb("#1772f0")
GRAY = to_rgb("#767676")
LIGHT_PURPLE = to_rgb("#d95ca1")
DARK_PURPLE = to_rgb("#9b5aa4")
LIGHT_GREEN = to_rgb("#5abf95")
DARK_GREEN = to_rgb("#3dae82")

METRICS = [
    peppr.MonomerTMScore(),
    peppr.MonomerLDDTScore(),
    peppr.LDDTPLIScore(),
    peppr.ContactFraction(),
    peppr.InterfaceRMSD(),
    peppr.LigandRMSD(),
    peppr.PocketAlignedLigandRMSD(),
    peppr.MonomerRMSD(threshold=2.0),
    peppr.BiSyRMSD(threshold=2.0),
]


def load_structure(cif_path):
    pdbx_file = pdbx.CIFFile.read(cif_path)
    system = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    system = system[
        ~struc.filter_solvent(system) & ~struc.filter_monoatomic_ions(system)
    ]
    return system


def setup_matplotlib(theme):
    if theme == "light":
        plt.style.use(["default"])
    if theme == "dark":
        plt.style.use(["dark_background"])
    matplotlib.rcParams["figure.dpi"] = 300
    matplotlib.rcParams["font.sans-serif"] = "Geologica"


def create_metric_plot(metrics, reference, pose, output_path, theme):
    setup_matplotlib(theme)

    score_results = {}
    distance_results = {}
    for metric in metrics:
        result = metric.evaluate(reference, pose)
        # For the showcase metrics, the ones which have ideally low values
        # are RMSD-based metrics
        if metric.smaller_is_better():
            distance_results[metric.name] = result
        else:
            score_results[metric.name] = result

    fig, (score_ax, distance_ax) = plt.subplots(
        ncols=2,
        figsize=(6, 3),
        constrained_layout=True,
        dpi=300,
        # Use 'axisartist.Axes' to allow setting tick label alignment
        # (https://matplotlib.org/stable/gallery/axisartist/demo_ticklabel_alignment.html)
        subplot_kw=dict(axes_class=axisartist.Axes),
    )

    score_ax.bar(
        list(score_results.keys()), list(score_results.values()), color=LIGHT_PURPLE
    )
    distance_ax.bar(
        list(distance_results.keys()),
        list(distance_results.values()),
        color=LIGHT_PURPLE,
    )

    for ax in [score_ax, distance_ax]:
        ax.axis["bottom"].major_ticklabels.set_ha("left")
        ax.axis["bottom"].major_ticklabels.set_rotation(-30)
        ax.axis["top"].set_visible(False)
    score_ax.set_ylim(0, 1)
    distance_ax.set_ylim(0, 10)
    score_ax.set_ylabel("Score")
    distance_ax.set_ylabel("Distance (Å)")
    # Adjust axis positions
    score_ax.axis["right"].set_visible(False)
    distance_ax.axis["left"].set_visible(False)
    distance_ax.axis["right"].major_ticklabels.set_visible(True)
    distance_ax.axis["right"].major_ticks.set_visible(True)
    distance_ax.axis["right"].label.set_visible(True)

    fig.savefig(output_path, transparent=True)


if __name__ == "__main__":
    showcase_dir = Path(__file__).parent / "showcase"

    reference = load_structure(showcase_dir / "reference.cif")
    pose = load_structure(showcase_dir / "pose.cif")
    reference_order, pose_order = peppr.find_matching_atoms(reference, pose)
    reference = reference[reference_order]
    pose = pose[pose_order]

    for theme in ["light", "dark"]:
        suffix = "_dark" if theme == "dark" else ""
        create_metric_plot(
            METRICS, reference, pose, showcase_dir / f"metrics{suffix}.png", theme
        )
