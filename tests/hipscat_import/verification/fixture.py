"""Run pass/fail tests and generate verification report of existing hipscat table."""

from pathlib import Path

import attrs
import yaml

from hipscat_import.verification.arguments import VerificationArguments
from hipscat_import.verification.run_verification import Verifier


@attrs.define
class VerifierFixture:
    """"""

    test_targets: dict[str, list | dict] = attrs.field(validator=attrs.validators.instance_of(dict))
    verifier: Verifier = attrs.field(validator=attrs.validators.instance_of(Verifier))
    assert_passed: bool | dict = attrs.field(validator=attrs.validators.instance_of((bool, dict)))

    @classmethod
    def from_param(
        cls, fixture_param: str, malformed_catalog_dirs: dict[str, Path], tmp_path: Path
    ) -> "VerifierFixture":
        with open(Path(__file__).parent / "fixture_defs.yaml", "r") as fin:
            fixture_defs = yaml.safe_load(fin)
        fixture_def = fixture_defs[fixture_param]

        truth_schema = fixture_def.get("truth_schema")
        if truth_schema is not None:
            truth_schema = malformed_catalog_dirs[truth_schema.split("/")[0]] / truth_schema.split("/")[1]
        args = VerificationArguments(
            input_catalog_path=malformed_catalog_dirs[fixture_def["input_dir"]],
            output_path=tmp_path,
            truth_schema=truth_schema,
            truth_total_rows=fixture_def.get("truth_total_rows"),
        )

        fixture = cls(
            test_targets=fixture_defs["test_targets"],
            verifier=Verifier.from_args(args),
            assert_passed=fixture_def["assert_passed"],
        )
        return fixture

    @staticmethod
    def unpack_assert_passed(
        assert_passed: bool | dict, *, targets: list | None = None
    ) -> tuple[bool, list] | dict:
        if isinstance(assert_passed, bool):
            if targets is None:
                return assert_passed, []
            return {target: assert_passed for target in targets}

        # assert_passed is a dict

        if targets is None:
            # Expecting a single item with key=False, value=list of file suffixes that should have failed.
            msg = "Unexpected key. There is probably a bug in the fixture definition."
            assert set(assert_passed) == {False}, msg
            return False, assert_passed[False]

        # Expecting one key per target
        msg = "Unexpected set of targets. There is probably a bug in the fixture definition."
        assert set(assert_passed) == set(targets), msg
        return assert_passed
