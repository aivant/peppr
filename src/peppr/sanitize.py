# mypy: disable-error-code="attr-defined"

__all__ = ["sanitize"]

import rdkit
import rdkit.Chem.AllChem as Chem


def sanitize(mol: Chem.Mol, max_fix_iterations: int = 1000) -> None:
    """
    Fix small issues with RDKit SanitizeMol and sanitize molecule.

    This is an alternative to using :func:`SanitizeMol()` directly, in cases it fails
    due to issues with the molecule which are fixed by this function.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        The molecule to sanitize.
    max_fix_iterations : int, optional
        The maximum number of iterations to fix problems.

    Notes
    -----
    Deals with cases:

    - ``AtomValenceException``: add charge if total valence is exceeding default (for N, O atoms only)
    - ``KekulizeException``: for ring N atoms with unspecified protonation
    """
    mol.UpdatePropertyCache(strict=False)
    prev_problems: list[Exception] = []

    # RDKit detects probels iteratively,
    # i.e. new problems may show up as soon as previous problems are fixed
    for _ in range(max_fix_iterations):
        with rdkit.rdBase.BlockLogs():
            # Temporarily disable RDKit verbosity while in scope
            problems = Chem.DetectChemistryProblems(mol)
        if not problems:
            # All problematic places have been fixed
            break
        for problem in problems:
            for prev_problem in prev_problems:
                if _is_same_problem(problem, prev_problem):
                    # Fixing this problem previously failed,
                    # so there is no need to try again
                    break
        for problem in problems:
            _fix_problem(mol, problem)
        prev_problems = list(problems)

    with rdkit.rdBase.BlockLogs():
        # if any issue remains - it will be caught as an Error
        Chem.SanitizeMol(mol)


def _fix_problem(mol: Chem.Mol, problem: Exception) -> None:
    """
    Apply fixes for common problems.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        The molecule to fix.
    problem : Exception
        The problem to fix.
    """
    pt = Chem.GetPeriodicTable()
    if problem.GetType() == "AtomValenceException":
        at = mol.GetAtomWithIdx(problem.GetAtomIdx())
        elem = at.GetAtomicNum()
        if elem == 6:
            # we do not like charged carbons, thus pass without modification
            return
        elif elem in [7, 8]:
            # only process positively charged N, O atoms
            opt_charge = at.GetExplicitValence() - pt.GetDefaultValence(elem)
            formal_charge = at.GetFormalCharge()
            if opt_charge > formal_charge:
                if abs(opt_charge) > 1:
                    raise AssertionError(f"absolute expected charge > 1: {opt_charge}")
                else:
                    # fix explicit valence issue
                    at.SetFormalCharge(opt_charge)
        else:
            # for higher elements like S - may need to use pt.GetValenceList()
            # currently not implemented
            return
    if problem.GetType() == "KekulizeException":
        # hack: only works for nitrogens with missing explicit Hs
        for atidx in problem.GetAtomIndices():
            at = mol.GetAtomWithIdx(atidx)
            # set one of the nitrogens with two bonds in a ring system as "[nH]"
            if at.GetAtomicNum() == 7 and at.GetDegree() == 2:
                at.SetNumExplicitHs(1)
                break


def _is_same_problem(problem1: Exception, problem2: Exception) -> bool:
    """
    Check if two exceptions related to chemistry problems are the same.

    Parameters
    ----------
    problem1, problem2 : Exception
        The problems to compare.

    Returns
    -------
    same : bool
        True if the two problems are the same.
    """
    if problem1.GetType() != problem2.GetType():
        return False
    if problem1.GetAtomIndices() != problem2.GetAtomIndices():
        return False
    return True
