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
    Using the same complex as model and native structure should result in a
    perfect DockQ score.
    """
    pdb_file = pdb.PDBFile.read(data_dir / "1A2K" / "model.pdb")
    native = pdb.get_structure(pdb_file, model=1)
    native = native[struc.filter_amino_acids(native)]
    # The model complex is simply the same complex as native,
    # but rotated to make the test less trivial
    model = struc.rotate(native, [0, 0, np.pi / 2])

    native_receptor = native[native.chain_id == "B"]
    native_ligand = native[native.chain_id == "C"]
    model_receptor = model[model.chain_id == "B"]
    model_ligand = model[model.chain_id == "C"]

    dockq_result = peppr.dockq(
        native_receptor,
        native_ligand,
        model_receptor,
        model_ligand,
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

    We first move the native ligand away to make sure they native receptor
    and native ligand are not in contact. Then we run dockq and check if the
    attributes are NaN.

    :param data_dir: pytest fixture, path to the directory containing the test data
    """
    model_receptor, model_ligand = _get_protein_receptor_and_ligand(
        data_dir / "1A2K" / "model.pdb", "C", "B"
    )
    native_receptor, native_ligand = _get_protein_receptor_and_ligand(
        data_dir / "1A2K" / "native.pdb", "C", "B"
    )
    # move native ligand to far away
    native_ligand = struc.translate(native_ligand, [200, 200, 200])

    # align model and native and then run dockq
    native_indices, model_indices = peppr.find_matching_atoms(
        native_receptor, model_receptor
    )
    native_receptor = native_receptor[native_indices]
    model_receptor = model_receptor[model_indices]
    native_indices, model_indices = peppr.find_matching_atoms(
        native_ligand, model_ligand
    )
    native_ligand = native_ligand[native_indices]
    model_ligand = model_ligand[model_indices]
    dockq_result = peppr.dockq(
        native_receptor, native_ligand, model_receptor, model_ligand
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
    model_chain_1, model_chain_2 = _get_protein_receptor_and_ligand(
        data_dir / "1A2K" / "model.pdb", "B", "C"
    )
    native_chain_1, native_chain_2 = _get_protein_receptor_and_ligand(
        data_dir / "1A2K" / "native.pdb", "B", "C"
    )
    native_indices, model_indices = peppr.find_matching_atoms(
        native_chain_1, model_chain_1
    )
    native_chain_1 = native_chain_1[native_indices]
    model_chain_1 = model_chain_1[model_indices]
    native_indices, model_indices = peppr.find_matching_atoms(
        native_chain_2, model_chain_2
    )
    native_chain_2 = native_chain_2[native_indices]
    model_chain_2 = model_chain_2[model_indices]

    dockq_result_1 = peppr.dockq(
        native_chain_1,
        native_chain_2,
        model_chain_1,
        model_chain_2,
    )
    dockq_result_2 = peppr.dockq(
        native_chain_2,
        native_chain_1,
        model_chain_2,
        model_chain_1,
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

    model_receptor, model_ligand = _get_protein_receptor_and_ligand(
        data_dir / entry / "model.pdb", receptor_chain, ligand_chain
    )
    native_receptor, native_ligand = _get_protein_receptor_and_ligand(
        data_dir / entry / "native.pdb", receptor_chain, ligand_chain
    )
    native_indices, model_indices = peppr.find_matching_atoms(
        native_receptor, model_receptor, min_sequence_identity=0.9
    )
    native_receptor = native_receptor[native_indices]
    model_receptor = model_receptor[model_indices]
    native_indices, model_indices = peppr.find_matching_atoms(
        native_ligand, model_ligand, min_sequence_identity=0.9
    )
    native_ligand = native_ligand[native_indices]
    model_ligand = model_ligand[model_indices]

    # Ensure the reference DockQ implementation gets the same atom matching
    model_tmp_file = tmp_path / "model.cif"
    _write_dockq_compatible_cif(model_tmp_file, model_receptor, model_ligand)
    native_tmp_file = tmp_path / "native.cif"
    _write_dockq_compatible_cif(native_tmp_file, native_receptor, native_ligand)
    model = ref_impl.load_PDB(str(model_tmp_file))
    native = ref_impl.load_PDB(str(native_tmp_file))
    chain_map = {receptor_chain: receptor_chain, ligand_chain: ligand_chain}
    ref_dockq_all_combinations, _ = ref_impl.run_on_all_native_interfaces(
        model, native, chain_map=chain_map, capri_peptide=as_peptide
    )
    # Key is a tuple, and both tuple orders are possible
    # There is only one result due to mapping constraints given as input
    ref_dockq = next(iter(ref_dockq_all_combinations.values()))

    if native_receptor.array_length() == native_ligand.array_length():
        # There is ambiguity which chain is selected as receptor
        # in the reference implementation
        # Hence choose the closer one
        test_dockq_variants = [
            peppr.dockq(nat_receptor, nat_ligand, mod_receptor, mod_ligand, as_peptide)
            for nat_receptor, nat_ligand, mod_receptor, mod_ligand in [
                (native_receptor, native_ligand, model_receptor, model_ligand),
                (native_ligand, native_receptor, model_ligand, model_receptor),
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
            native_receptor, native_ligand, model_receptor, model_ligand, as_peptide
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
        model_receptor,
        model_ligand,
    ) = _get_receptor_and_small_molecule(data_dir / "6J6J" / "model.pdb", "A")
    (
        native_receptor,
        native_ligand,
    ) = _get_receptor_and_small_molecule(data_dir / "6J6J" / "native.pdb", "A")
    native_indices, model_indices = peppr.find_matching_atoms(
        native_receptor, model_receptor
    )
    native_receptor = native_receptor[native_indices]
    model_receptor = model_receptor[model_indices]
    native_indices, model_indices = peppr.find_matching_atoms(
        native_ligand, model_ligand
    )
    native_ligand = native_ligand[native_indices]
    model_ligand = model_ligand[model_indices]
    test_dockq = peppr.dockq(
        native_receptor, native_ligand, model_receptor, model_ligand
    )

    # Ensure the reference DockQ implementation gets the same atom matching
    model_tmp_file = tmp_path / "model.cif"
    _write_dockq_compatible_cif(model_tmp_file, model_receptor, model_ligand)
    native_tmp_file = tmp_path / "native.cif"
    _write_dockq_compatible_cif(native_tmp_file, native_receptor, native_ligand)
    model = ref_impl.load_PDB(str(model_tmp_file), small_molecule=True)
    native = ref_impl.load_PDB(str(native_tmp_file), small_molecule=True)
    chain_map = {"R": "R", "L": "L"}
    ref_dockq_all_combinations, _ = ref_impl.run_on_all_native_interfaces(
        model, native, chain_map=chain_map
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

    model_receptor, model_ligand = _get_protein_receptor_and_ligand(
        data_dir / "1A2K" / "model.pdb", "C", "B"
    )
    native_receptor, native_ligand = _get_protein_receptor_and_ligand(
        data_dir / "1A2K" / "native.pdb", "C", "B"
    )
    native_indices, model_indices = peppr.find_matching_atoms(
        native_receptor, model_receptor
    )
    native_receptor = native_receptor[native_indices]
    model_receptor = model_receptor[model_indices]
    native_indices, model_indices = peppr.find_matching_atoms(
        native_ligand, model_ligand
    )
    native_ligand = native_ligand[native_indices]
    model_ligand = model_ligand[model_indices]
    ref_dockq = peppr.dockq(
        native_receptor, native_ligand, model_receptor, model_ligand
    )

    # Put duplicates of the same model into a stack
    model_receptor = struc.stack([model_receptor] * N_DUPLICATES)
    model_ligand = struc.stack([model_ligand] * N_DUPLICATES)
    test_dockq = peppr.dockq(
        native_receptor, native_ligand, model_receptor, model_ligand
    )

    for attr in ["fnat", "fnonnat", "irmsd", "lrmsd", "score"]:
        if not np.isnan(getattr(ref_dockq, attr)):
            assert getattr(test_dockq, attr) == pytest.approx(
                [getattr(ref_dockq, attr)] * N_DUPLICATES, abs=1e-3
            )
        else:
            assert np.isnan(getattr(test_dockq, attr)).all()
