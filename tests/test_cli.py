import json
import biotite.structure.info as info
import biotite.structure.io as strucio
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
import peppr
from peppr.cli import (
    _METRICS,
    create,
    evaluate,
    evaluate_batch,
    run,
    summarize,
    tabulate,
)
from tests.common import assemble_predictions

N_SYSTEMS = 5
N_POSES = 3
N_METRICS = 2


@pytest.fixture(scope="module")
def test_system():
    return assemble_predictions("7znt__2__1.F_1.G__1.J")


@pytest.fixture()
def system_dir(test_system, tmp_path):
    """
    A directory containing systems with multiple poses each.
    All systems have the same structures.
    All poses are also equal.
    """
    reference, pose = test_system
    system_dir = tmp_path / "systems"
    for i in range(N_SYSTEMS):
        system_id = f"system_{i}"
        reference_dir = system_dir / "references"
        pose_dir = system_dir / "poses" / system_id
        reference_dir.mkdir(parents=True, exist_ok=True)
        pose_dir.mkdir(parents=True, exist_ok=True)
        strucio.save_structure(reference_dir / f"{system_id}.bcif", reference)
        for j in range(N_POSES):
            pose_id = f"pose_{j}"
            strucio.save_structure(pose_dir / f"{pose_id}.bcif", pose)
    return system_dir


@pytest.fixture()
def metrics():
    """
    Some metric names available in the CLI.
    """
    return list(_METRICS.keys())[:N_METRICS]


def test_all_metrics_available():
    """
    Check if all metrics implemented in `peppr` are available in the CLI.
    """
    cli_metrics = set([type(metric) for metric in _METRICS.values()])

    for attr_name, attr in peppr.__dict__.items():
        if attr is peppr.Metric:
            # The base class itself should not be part of the CLI
            continue
        if isinstance(attr, type) and issubclass(attr, peppr.Metric):
            assert attr in cli_metrics, f"{attr_name} is not available in the CLI"


@pytest.mark.parametrize("use_multi_pose", [False, True])
@pytest.mark.parametrize("use_batch", [False, True])
def test_tabulate(
    metrics,
    system_dir,
    tmp_path,
    use_multi_pose,
    use_batch,
):
    """
    Check if the ``tabulate`` command works for different combinations of input.
    If successful, the shape of the created table should match the number of input
    systems and metrics.
    """
    evaluator_path = tmp_path / "peppr.pkl"
    table_path = tmp_path / "table.csv"

    runner = CliRunner()

    result = runner.invoke(create, ["--strict", evaluator_path.as_posix()] + metrics)
    if result.exception:
        raise result.exception

    if use_batch:
        reference_pattern = f"{system_dir.as_posix()}/references/*.bcif"
        if use_multi_pose:
            pose_pattern = f"{system_dir.as_posix()}/poses/*"
        else:
            pose_pattern = f"{system_dir.as_posix()}/poses/*/pose_0.bcif"
        result = runner.invoke(
            evaluate_batch, [evaluator_path.as_posix(), reference_pattern, pose_pattern]
        )
        if result.exception:
            raise result.exception
    else:
        for i in range(N_SYSTEMS):
            reference_path = (system_dir / "references" / f"system_{i}.bcif").as_posix()
            if use_multi_pose:
                pose_paths = [
                    path.as_posix()
                    for path in (system_dir / "poses" / f"system_{i}").iterdir()
                ]
            else:
                pose_paths = [
                    (system_dir / "poses" / f"system_{i}" / "pose_0.bcif").as_posix()
                ]
            result = runner.invoke(
                evaluate, [evaluator_path.as_posix(), reference_path] + pose_paths
            )
            if result.exception:
                raise result.exception

    tabulate_args = [evaluator_path.as_posix(), table_path.as_posix()]
    if use_multi_pose:
        tabulate_args.append("oracle")
    result = runner.invoke(tabulate, tabulate_args)

    if result.exception:
        raise result.exception
    table = pd.read_csv(table_path)
    # The first column is the system ID, the rest are the metrics
    assert table.shape == (N_SYSTEMS, N_METRICS + 1)
    assert table["System ID"].to_list() == [f"system_{i}" for i in range(N_SYSTEMS)]


@pytest.mark.parametrize(
    ["selector", "selector_name"],
    [
        (peppr.MeanSelector(), "mean"),
        (peppr.MedianSelector(), "median"),
        (peppr.OracleSelector(), "oracle"),
        (peppr.TopSelector(2), "top2"),
        (peppr.RandomSelector(2), "random2"),
    ],
)
def test_summarize(metrics, system_dir, tmp_path, selector, selector_name):
    """
    Check if the ``summarize`` command created a valid JSON file.
    Furthermore, check if all selectors work, by looking for their names in the JSON
    file.
    """
    evaluator_path = tmp_path / "peppr.pkl"
    summary_path = tmp_path / "summary.json"

    runner = CliRunner()

    result = runner.invoke(create, ["--strict", evaluator_path.as_posix()] + metrics)
    if result.exception:
        raise result.exception

    reference_pattern = f"{system_dir.as_posix()}/references/*.bcif"
    pose_pattern = f"{system_dir.as_posix()}/poses/*"
    result = runner.invoke(
        evaluate_batch, [evaluator_path.as_posix(), reference_pattern, pose_pattern]
    )
    if result.exception:
        raise result.exception

    result = runner.invoke(
        summarize, [evaluator_path.as_posix(), summary_path.as_posix(), selector_name]
    )
    if result.exception:
        raise result.exception
    with open(summary_path) as f:
        summary = json.load(f)
    assert f"CA-RMSD <2.0 ({selector.name})" in summary.keys()


def test_single_multi_core_equivalence(tmp_path, system_dir, metrics):
    """
    Check if the ``tabulate`` command gives the same results in a single- and
    multiprocessed setting.
    """
    evaluator_path = tmp_path / "peppr.pkl"
    table_path = tmp_path / "table.csv"
    reference_pattern = f"{system_dir.as_posix()}/references/*.bcif"
    pose_pattern = f"{system_dir.as_posix()}/poses/*/pose_0.bcif"

    runner = CliRunner()

    tables = []
    for cores in [1, 2]:
        result = runner.invoke(
            create, ["--strict", evaluator_path.as_posix()] + metrics
        )
        if result.exception:
            raise result.exception
        result = runner.invoke(
            evaluate_batch,
            [
                evaluator_path.as_posix(),
                reference_pattern,
                pose_pattern,
                "--cores",
                str(cores),
            ],
        )
        if result.exception:
            raise result.exception
        result = runner.invoke(
            tabulate, [evaluator_path.as_posix(), table_path.as_posix()]
        )
        if result.exception:
            raise result.exception
        tables.append(pd.read_csv(table_path))

    assert tables[0].equals(tables[1])


@pytest.mark.parametrize("metric_name", list(_METRICS.keys()))
def test_run_metrics(system_dir, metric_name):
    """
    Check if the ``run`` command works for all available metrics.
    This also checks if each metrics can be successfully created from name.
    """
    reference_path = (system_dir / "references" / "system_0.bcif").as_posix()
    pose_path = (system_dir / "poses" / "system_0" / "pose_0.bcif").as_posix()

    result = CliRunner().invoke(run, [metric_name, reference_path, pose_path])
    if result.exception:
        raise result.exception
    # Result must be convertible to float
    # and not NaN as the selected system supports all metrics
    result = float(result.output)
    assert not np.isnan(result)


@pytest.mark.parametrize("match_method", [m.value for m in peppr.Evaluator.MatchMethod])
def test_run_match_method(system_dir, match_method):
    """
    Check if the ``run`` command works for all available match methods.
    This also checks if each metrics can be successfully created from name.
    """
    reference_path = (system_dir / "references" / "system_0.bcif").as_posix()
    pose_path = (system_dir / "poses" / "system_0" / "pose_0.bcif").as_posix()

    result = CliRunner().invoke(
        run, ["irmsd", reference_path, pose_path, "-m", match_method]
    )
    if result.exception:
        raise result.exception
    # Result must be convertible to float and not NaN
    result = float(result.output)
    assert not np.isnan(result)


@pytest.mark.parametrize("format", ["cif", "bcif", "pdb", "mol", "sdf"])
def test_structure_formats(tmp_path, format):
    """
    Use the ``run`` command to check if systems can be read from different file formats.
    Use the same structure for reference and pose to check if the output is a perfect
    fit.
    As the formats also comprise small molecule formats, use a small molecule as
    system.
    """
    system = info.residue("PNN")
    system.hetero[:] = True
    path = tmp_path / f"system.{format}"
    strucio.save_structure(path, system)

    # Use a metric that supports small molecules
    result = CliRunner().invoke(run, ["ligand-lddt", path.as_posix(), path.as_posix()])
    if result.exception:
        raise result.exception
    assert float(result.output) == pytest.approx(1.0)
