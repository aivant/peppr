from pathlib import Path
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest
import peppr


def _filter_relevant_atoms(atoms):
    return ~struc.filter_solvent(atoms) & (atoms.element != "H")


def _get_protein_receptor_and_ligand(pdb_path, receptor_chain_id, ligand_chain_id):
    """
    Get a protein receptor-ligand pair from a PDB file with the given `pdb_path`.
    """
    pdb_file = pdb.PDBFile.read(pdb_path)
    atoms = pdb.get_structure(pdb_file, model=1)
    atoms = atoms[struc.filter_amino_acids(atoms) & _filter_relevant_atoms(atoms)]
    receptor = atoms[atoms.chain_id == receptor_chain_id]
    ligand = atoms[atoms.chain_id == ligand_chain_id]
    return receptor, ligand


def _get_receptor_and_small_molecule(pdb_path, chain_id):
    """
    Get a protein receptor-small molecule pair from a PDB file with the given
    `pdb_path`.
    The chains are renamed into `R` and `L`, respectively, to ensure the correct
    chains are used by the reference DockQ.
    These new chains are also written into the returned PDB file.
    """
    pdb_file = pdb.PDBFile.read(pdb_path)
    atoms = pdb.get_structure(pdb_file, model=1)
    atoms = atoms[(atoms.chain_id == chain_id) & _filter_relevant_atoms(atoms)]
    atoms.bonds = struc.connect_via_residue_names(atoms)
    receptor = atoms[struc.filter_amino_acids(atoms)]
    receptor.chain_id[:] = "R"
    ligand = atoms[~struc.filter_amino_acids(atoms)]
    ligand.chain_id[:] = "L"
    ligand.hetero[:] = True
    return receptor, ligand


def _write_dockq_compatible_cif(path, receptor, ligand):
    pdbx_file = pdbx.CIFFile()
    pdbx.set_structure(pdbx_file, receptor + ligand)
    pdbx_file.block["atom_site"]["B_iso_or_equiv"] = np.zeros(
        pdbx_file.block["atom_site"].row_count
    )
    pdbx_file.block["atom_site"]["occupancy"] = np.ones(
        pdbx_file.block["atom_site"].row_count
    )
    pdbx_file.write(path)


@pytest.fixture
def data_dir():
    return Path(__file__).parent / "data" / "dockq"


def test_perfect_docking(data_dir):
    """
    Using the same complex as pose and reference structure should result in a
    perfect DockQ score.
    """
    pdb_file = pdb.PDBFile.read(data_dir / "1A2K" / "model.pdb")
    reference = pdb.get_structure(pdb_file, model=1)
    reference = reference[struc.filter_amino_acids(reference)]
    # The model is simply the same complex as reference,
    # but rotated to make the test less trivial
    pose = struc.rotate(reference, [0, 0, np.pi / 2])

    reference_receptor = reference[reference.chain_id == "B"]
    reference_ligand = reference[reference.chain_id == "C"]
    pose_receptor = pose[pose.chain_id == "B"]
    pose_ligand = pose[pose.chain_id == "C"]

    dockq_result = peppr.dockq(
        reference_receptor,
        reference_ligand,
        pose_receptor,
        pose_ligand,
    )

    assert dockq_result.fnat == pytest.approx(1.0, abs=1e-5)
    assert dockq_result.fnonnat == pytest.approx(0.0, abs=1e-5)
    assert dockq_result.irmsd == pytest.approx(0.0, abs=1e-5)
    assert dockq_result.lrmsd == pytest.approx(0.0, abs=1e-5)
    assert dockq_result.score == pytest.approx(1.0, abs=1e-5)


def test_dockq_with_no_contacts(data_dir):
    """
    SUT: peppr.dockq
    Collaborators:
    - peppr.find_matching_atoms

    DockQ should return NaN for fnat, fnonnat, irmsd and score when the
    two chains are not in contact.

    We first move the reference ligand away to make sure they reference receptor
    and reference ligand are not in contact. Then we run dockq and check if the
    attributes are NaN.

    :param data_dir: pytest fixture, path to the directory containing the test data
    """
    pose_receptor, pose_ligand = _get_protein_receptor_and_ligand(
        data_dir / "1A2K" / "model.pdb", "C", "B"
    )
    reference_receptor, reference_ligand = _get_protein_receptor_and_ligand(
        data_dir / "1A2K" / "native.pdb", "C", "B"
    )
    # move reference ligand to far away
    reference_ligand = struc.translate(reference_ligand, [200, 200, 200])

    # align pose and reference and then run dockq
    reference_indices, pose_indices = peppr.find_matching_atoms(
        reference_receptor, pose_receptor
    )
    reference_receptor = reference_receptor[reference_indices]
    pose_receptor = pose_receptor[pose_indices]
    reference_indices, pose_indices = peppr.find_matching_atoms(
        reference_ligand, pose_ligand
    )
    reference_ligand = reference_ligand[reference_indices]
    pose_ligand = pose_ligand[pose_indices]
    dockq_result = peppr.dockq(
        reference_receptor, reference_ligand, pose_receptor, pose_ligand
    )

    # fnat, fnonnat, irmsd and score are nan when the
    # two chains are not in contact

    nan_attrs = ["fnat", "fnonnat", "irmsd", "score"]
    for nan_attr in nan_attrs:
        assert np.isnan(getattr(dockq_result, nan_attr))

    # expect lrmsd is still calculated when chains are not in contact
    assert not np.isnan(dockq_result.lrmsd)


def test_contact_parity(data_dir):
    """
    Exchanging which chain is the receptor and which is the ligand should not
    change the contacts and in consequence not fnat/fnonnat, too.
    """
    pose_chain_1, pose_chain_2 = _get_protein_receptor_and_ligand(
        data_dir / "1A2K" / "model.pdb", "B", "C"
    )
    reference_chain_1, reference_chain_2 = _get_protein_receptor_and_ligand(
        data_dir / "1A2K" / "native.pdb", "B", "C"
    )
    reference_indices, pose_indices = peppr.find_matching_atoms(
        reference_chain_1, pose_chain_1
    )
    reference_chain_1 = reference_chain_1[reference_indices]
    pose_chain_1 = pose_chain_1[pose_indices]
    reference_indices, pose_indices = peppr.find_matching_atoms(
        reference_chain_2, pose_chain_2
    )
    reference_chain_2 = reference_chain_2[reference_indices]
    pose_chain_2 = pose_chain_2[pose_indices]

    dockq_result_1 = peppr.dockq(
        reference_chain_1,
        reference_chain_2,
        pose_chain_1,
        pose_chain_2,
    )
    dockq_result_2 = peppr.dockq(
        reference_chain_2,
        reference_chain_1,
        pose_chain_2,
        pose_chain_1,
    )

    assert dockq_result_1.fnat == pytest.approx(dockq_result_2.fnat, abs=1e-5)
    assert dockq_result_1.fnonnat == pytest.approx(dockq_result_2.fnonnat, abs=1e-5)
    # The interface RMSD does also not distinguish between receptor and ligand
    assert dockq_result_1.irmsd == pytest.approx(dockq_result_2.irmsd, abs=1e-5)


@pytest.mark.parametrize("as_peptide", [False, True])
@pytest.mark.parametrize(
    ["entry", "receptor_chain", "ligand_chain"],
    [
        ("1A2K", "C", "A"),
        ("1A2K", "C", "B"),
        ("5O2Z", "R", "L"),
        ("6S0A", "A", "B"),
        ("6J6J", "A", "B"),
        ("6J6J", "A", "C"),
        # ("2E31", "R", "L"),  # Disabled due to https://github.com/bjornwallner/DockQ/pull/40
    ],
)
def test_reference_consistency(
    tmp_path, data_dir, entry, receptor_chain, ligand_chain, as_peptide
):
    """
    Expect the same DockQ results as output from the DockQ reference implementation.
    """
    ref_impl = pytest.importorskip("DockQ.DockQ")

    pose_receptor, pose_ligand = _get_protein_receptor_and_ligand(
        data_dir / entry / "model.pdb", receptor_chain, ligand_chain
    )
    reference_receptor, reference_ligand = _get_protein_receptor_and_ligand(
        data_dir / entry / "native.pdb", receptor_chain, ligand_chain
    )
    reference_indices, pose_indices = peppr.find_matching_atoms(
        reference_receptor, pose_receptor, min_sequence_identity=0.9
    )
    reference_receptor = reference_receptor[reference_indices]
    pose_receptor = pose_receptor[pose_indices]
    reference_indices, pose_indices = peppr.find_matching_atoms(
        reference_ligand, pose_ligand, min_sequence_identity=0.9
    )
    reference_ligand = reference_ligand[reference_indices]
    pose_ligand = pose_ligand[pose_indices]

    # Ensure the reference DockQ implementation gets the same atom matching
    pose_tmp_file = tmp_path / "pose.cif"
    _write_dockq_compatible_cif(pose_tmp_file, pose_receptor, pose_ligand)
    reference_tmp_file = tmp_path / "reference.cif"
    _write_dockq_compatible_cif(
        reference_tmp_file, reference_receptor, reference_ligand
    )
    pose = ref_impl.load_PDB(str(pose_tmp_file))
    reference = ref_impl.load_PDB(str(reference_tmp_file))
    chain_map = {receptor_chain: receptor_chain, ligand_chain: ligand_chain}
    ref_dockq_all_combinations, _ = ref_impl.run_on_all_native_interfaces(
        pose, reference, chain_map=chain_map, capri_peptide=as_peptide
    )
    # Key is a tuple, and both tuple orders are possible
    # There is only one result due to mapping constraints given as input
    ref_dockq = next(iter(ref_dockq_all_combinations.values()))

    if reference_receptor.array_length() == reference_ligand.array_length():
        # There is ambiguity which chain is selected as receptor
        # in the reference implementation
        # Hence choose the closer one
        test_dockq_variants = [
            peppr.dockq(nat_receptor, nat_ligand, mod_receptor, mod_ligand, as_peptide)
            for nat_receptor, nat_ligand, mod_receptor, mod_ligand in [
                (reference_receptor, reference_ligand, pose_receptor, pose_ligand),
                (reference_ligand, reference_receptor, pose_ligand, pose_receptor),
            ]
        ]
        test_dockq = test_dockq_variants[
            np.argmin(
                [
                    np.abs(result.score - ref_dockq["DockQ"])
                    for result in test_dockq_variants
                ]
            )
        ]
    else:
        test_dockq = peppr.dockq(
            reference_receptor, reference_ligand, pose_receptor, pose_ligand, as_peptide
        )

    assert test_dockq.fnat == pytest.approx(ref_dockq["fnat"], abs=1e-3)
    assert test_dockq.fnonnat == pytest.approx(ref_dockq["fnonnat"], abs=1e-3)
    assert test_dockq.irmsd == pytest.approx(ref_dockq["iRMSD"], abs=1e-3)
    assert test_dockq.lrmsd == pytest.approx(ref_dockq["LRMSD"], abs=1e-3)
    assert test_dockq.score == pytest.approx(ref_dockq["DockQ"], abs=1e-3)


def test_reference_consistency_small_molecule(tmp_path, data_dir):
    """
    Expect the same DockQ results as output from the DockQ reference implementation
    for a complex between a protein and a small molecule.
    """
    ref_impl = pytest.importorskip("DockQ.DockQ")

    (
        pose_receptor,
        pose_ligand,
    ) = _get_receptor_and_small_molecule(data_dir / "6J6J" / "model.pdb", "A")
    (
        reference_receptor,
        reference_ligand,
    ) = _get_receptor_and_small_molecule(data_dir / "6J6J" / "native.pdb", "A")
    reference_indices, pose_indices = peppr.find_matching_atoms(
        reference_receptor, pose_receptor
    )
    reference_receptor = reference_receptor[reference_indices]
    pose_receptor = pose_receptor[pose_indices]
    reference_indices, pose_indices = peppr.find_matching_atoms(
        reference_ligand, pose_ligand
    )
    reference_ligand = reference_ligand[reference_indices]
    pose_ligand = pose_ligand[pose_indices]
    test_dockq = peppr.dockq(
        reference_receptor, reference_ligand, pose_receptor, pose_ligand
    )

    # Ensure the reference DockQ implementation gets the same atom matching
    pose_tmp_file = tmp_path / "model.cif"
    _write_dockq_compatible_cif(pose_tmp_file, pose_receptor, pose_ligand)
    reference_tmp_file = tmp_path / "reference.cif"
    _write_dockq_compatible_cif(
        reference_tmp_file, reference_receptor, reference_ligand
    )
    pose = ref_impl.load_PDB(str(pose_tmp_file), small_molecule=True)
    reference = ref_impl.load_PDB(str(reference_tmp_file), small_molecule=True)
    chain_map = {"R": "R", "L": "L"}
    ref_dockq_all_combinations, _ = ref_impl.run_on_all_native_interfaces(
        pose, reference, chain_map=chain_map
    )
    # Key is a tuple, and both tuple orders are possible
    # There is only one result due to mapping constraints given as input
    ref_dockq = next(iter(ref_dockq_all_combinations.values()))

    assert test_dockq.lrmsd == pytest.approx(ref_dockq["LRMSD"], abs=1e-3)
    assert test_dockq.score == pytest.approx(ref_dockq["DockQ"], abs=1e-3)


def test_multi_model(data_dir):
    """
    Check if providing an `AtomArrayStack` with multiple identical models, gives the
    same result for each model.
    """
    N_DUPLICATES = 5

    pose_receptor, pose_ligand = _get_protein_receptor_and_ligand(
        data_dir / "1A2K" / "model.pdb", "C", "B"
    )
    reference_receptor, reference_ligand = _get_protein_receptor_and_ligand(
        data_dir / "1A2K" / "native.pdb", "C", "B"
    )
    reference_indices, pose_indices = peppr.find_matching_atoms(
        reference_receptor, pose_receptor
    )
    reference_receptor = reference_receptor[reference_indices]
    pose_receptor = pose_receptor[pose_indices]
    reference_indices, pose_indices = peppr.find_matching_atoms(
        reference_ligand, pose_ligand
    )
    reference_ligand = reference_ligand[reference_indices]
    pose_ligand = pose_ligand[pose_indices]
    ref_dockq = peppr.dockq(
        reference_receptor, reference_ligand, pose_receptor, pose_ligand
    )

    # Put duplicates of the same model into a stack
    pose_receptor = struc.stack([pose_receptor] * N_DUPLICATES)
    pose_ligand = struc.stack([pose_ligand] * N_DUPLICATES)
    test_dockq = peppr.dockq(
        reference_receptor, reference_ligand, pose_receptor, pose_ligand
    )

    for attr in ["fnat", "fnonnat", "irmsd", "lrmsd", "score"]:
        if not np.isnan(getattr(ref_dockq, attr)):
            assert getattr(test_dockq, attr) == pytest.approx(
                [getattr(ref_dockq, attr)] * N_DUPLICATES, abs=1e-3
            )
        else:
            assert np.isnan(getattr(test_dockq, attr)).all()
