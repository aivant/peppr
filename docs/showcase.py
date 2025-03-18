from pathlib import Path
import biotite.interface.pymol as pymol_interface
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import peppr

# Colors are based on Lab color space at L = 50
RED = to_rgb("#db3a35")
GREEN = to_rgb("#088b05")
BLUE = to_rgb("#1772f0")
VIOLET = to_rgb("#cb38aa")
GRAY = to_rgb("#767676")

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


def setup_matplotlib():
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
    pymol_interface.cmd.set_color("red", RED)
    pymol_interface.cmd.set_color("green", GREEN)
    pymol_interface.cmd.set_color("blue", BLUE)
    pymol_interface.cmd.set_color("violet", VIOLET)
    pymol_interface.cmd.set_color("gray", GRAY)
    pymol_interface.cmd.set_color("carbon", GRAY)
    pymol_interface.cmd.set_color("oxygen", RED)
    pymol_interface.cmd.set_color("nitrogen", BLUE)


def create_metric_plot(metrics, reference, model, output_path):
    setup_matplotlib()

    score_results = {}
    distance_results = {}
    for metric in metrics:
        result = metric.evaluate(reference, model)
        # For the showcase metrics, the ones which have ideally low values
        # are RMSD-based metrics
        if metric.smaller_is_better():
            distance_results[metric.name] = result
        else:
            score_results[metric.name] = result

    fig, (score_ax, distance_ax) = plt.subplots(
        ncols=2, figsize=(6, 3), constrained_layout=True, dpi=300
    )

    score_ax.bar(list(score_results.keys()), list(score_results.values()), color=RED)
    distance_ax.bar(
        list(distance_results.keys()), list(distance_results.values()), color=RED
    )

    score_ax.tick_params(axis="x", labelrotation=-30)
    distance_ax.tick_params(axis="x", labelrotation=-30)
    score_ax.set_ylim(0, 1)
    distance_ax.set_ylim(0, 10)
    score_ax.set_ylabel("Score")
    distance_ax.set_ylabel("Distance (Å)")
    # Adjust axis positions
    score_ax.spines.top.set_visible(False)
    score_ax.spines.right.set_visible(False)
    distance_ax.spines.top.set_visible(False)
    distance_ax.spines.left.set_visible(False)
    distance_ax.tick_params(
        axis="y", left=False, labelleft=False, right=True, labelright=True
    )
    distance_ax.yaxis.set_label_position("right")

    fig.savefig(output_path)


def visualize_systems(reference, model, output_path):
    setup_pymol()

    pymol_reference = pymol_interface.PyMOLObject.from_structure(reference)
    pymol_model = pymol_interface.PyMOLObject.from_structure(model)

    peptide_mask = struc.filter_amino_acids(model)
    pymol_reference.show_as("cartoon")
    pymol_reference.show("sticks", ~peptide_mask)
    pymol_reference.color("gray")
    pymol_reference.set("stick_transparency", 0.5)
    pymol_reference.set("cartoon_transparency", 0.5)

    peptide_mask = struc.filter_amino_acids(model)
    pymol_model.show_as("cartoon", peptide_mask)
    pymol_model.color("red", peptide_mask)
    pymol_model.show_as("sticks", ~peptide_mask)
    pymol_model.color("green", ~peptide_mask)
    pymol_model.orient()
    pymol_model.zoom()
    # Use an angle where the ligand is better visible
    pymol_interface.cmd.rotate("x", 180)

    pymol_interface.cmd.png(output_path.as_posix(), width=1000, height=1000, ray=True)


if __name__ == "__main__":
    showcase_dir = Path(__file__).parent / "showcase"

    reference = load_structure(showcase_dir / "reference.cif")
    model = load_structure(showcase_dir / "models.cif")

    create_metric_plot(METRICS, reference, model, showcase_dir / "metrics.png")
    visualize_systems(reference, model, showcase_dir / "system.png")
