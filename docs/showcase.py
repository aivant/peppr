from pathlib import Path
import biotite.interface.pymol as pymol_interface
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


def setup_pymol():
    # Set style of PyMOL images
    pymol_interface.cmd.bg_color("white")
    pymol_interface.cmd.set("depth_cue", 0)
    pymol_interface.cmd.set("ray_shadows", 0)
    pymol_interface.cmd.set("spec_reflect", 0)
    pymol_interface.cmd.set("ray_trace_mode", 1)
    pymol_interface.cmd.set("ray_trace_disco_factor", 1)
    pymol_interface.cmd.set("cartoon_side_chain_helper", 1)
    pymol_interface.cmd.set("valence", 0)
    pymol_interface.cmd.set("cartoon_oval_length", 1.0)
    pymol_interface.cmd.set("label_color", "black")
    pymol_interface.cmd.set("label_size", 30)
    pymol_interface.cmd.set("dash_gap", 0.3)
    pymol_interface.cmd.set("dash_width", 2.0)
    pymol_interface.cmd.set_color("lightpurple", LIGHT_PURPLE)
    pymol_interface.cmd.set_color("darkpurple", DARK_PURPLE)
    pymol_interface.cmd.set_color("lightgreen", LIGHT_GREEN)
    pymol_interface.cmd.set_color("darkgreen", DARK_GREEN)
    pymol_interface.cmd.set_color("carbon", GRAY)
    pymol_interface.cmd.set_color("oxygen", RED)
    pymol_interface.cmd.set_color("nitrogen", BLUE)


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


def visualize_systems(reference, pose, output_path):
    setup_pymol()

    pymol_reference = pymol_interface.PyMOLObject.from_structure(
        reference, delocalize_bonds=True
    )
    pymol_pose = pymol_interface.PyMOLObject.from_structure(pose, delocalize_bonds=True)

    peptide_mask = struc.filter_amino_acids(pose)
    pymol_reference.show_as("cartoon")
    pymol_reference.show("sticks", ~peptide_mask)
    pymol_reference.color("gray")
    pymol_reference.set("stick_transparency", 0.5)
    pymol_reference.set("cartoon_transparency", 0.5)

    peptide_mask = struc.filter_amino_acids(pose)
    pymol_pose.show_as("cartoon", peptide_mask)
    pymol_pose.color("darkpurple", peptide_mask & (pose.chain_id == "0"))
    pymol_pose.color("lightpurple", peptide_mask & (pose.chain_id == "1"))
    pymol_pose.show_as("sticks", ~peptide_mask)
    pymol_pose.color("darkgreen", ~peptide_mask)
    pymol_pose.orient()

    # Tweak the camera
    pymol_interface.cmd.turn("x", 180)
    pymol_interface.cmd.move("x", -8)
    pymol_interface.cmd.move("z", 90)

    pymol_interface.cmd.png(output_path.as_posix(), width=1000, height=700, ray=True)


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
        visualize_systems(reference, pose, showcase_dir / f"system{suffix}.png")
