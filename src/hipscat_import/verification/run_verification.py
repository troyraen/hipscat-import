"""Run pass/fail tests and generate verification report of existing hipscat table."""

import collections
import datetime
from pathlib import Path

import attrs
import hipscat.io.validation
import pandas as pd
import pyarrow.dataset

from hipscat_import.verification.arguments import VerificationArguments


def run(args: VerificationArguments, write_mode: str = "a"):
    """Run verification pipeline."""
    if not args:
        raise TypeError("args is required and should be type VerificationArguments")
    if not isinstance(args, VerificationArguments):
        raise TypeError("args must be type VerificationArguments")

    verifier = Verifier.from_args(args)
    verifier.run(write_mode=write_mode)

    return verifier


Result = collections.namedtuple(
    "Result", ["passed", "test", "target", "description", "affected_files", "datetime"]
)
"""Verification test result."""


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S %Z")


@attrs.define
class Verifier:
    """Class for verification tests. Instantiate using the 'from_args' method."""

    args: VerificationArguments = attrs.field()
    """Arguments to use during verification."""
    files_ds: pyarrow.dataset.Dataset = attrs.field()
    """Pyarrow dataset, loaded from the actual files on disk."""
    metadata_ds: pyarrow.dataset.Dataset = attrs.field()
    """Pyarrow dataset, loaded from the _metadata file."""
    common_ds: pyarrow.dataset.Dataset = attrs.field()
    """Pyarrow dataset, loaded from the _common_metadata file."""
    truth_schema: pyarrow.Schema = attrs.field()
    """Pyarrow schema to be used as truth. This will be loaded from args.truth_schema
    if provided, and then hipscat columns and metadata will be added if not already present.
    If args.truth_schema not provided, the catalog's _common_metadata file will be used."""
    truth_src: str = attrs.field()
    """'truth_schema' if args.truth_schema was provided, else '_common_metadata'."""
    results: list[Result] = attrs.field(factory=list)
    """List of results, one for each test that has been done."""
    _distributions_df: pd.DataFrame | None = attrs.field(default=None)

    @classmethod
    def from_args(cls, args: VerificationArguments) -> "Verifier":
        # make sure the output directory exists
        args.output_path.mkdir(exist_ok=True, parents=True)

        # load a dataset from the actual files on disk
        files_ds = pyarrow.dataset.dataset(
            args.input_catalog_path,
            ignore_prefixes=[
                ".",
                "_",
                "catalog_info.json",
                "partition_info.csv",
                "point_map.fits",
                "provenance_info.json",
            ],
        )

        # load a dataset from the _metadata file
        metadata_ds = pyarrow.dataset.parquet_dataset(f"{args.input_catalog_path}/_metadata")

        # load a dataset from the _common_metadata file
        common_ds = pyarrow.dataset.parquet_dataset(f"{args.input_catalog_path}/_common_metadata")

        # load the input schema if provided, else use the _common_metadata schema
        if args.truth_schema is not None:
            truth_schema = pyarrow.dataset.parquet_dataset(args.truth_schema).schema
            truth_src = "truth_schema"
        else:
            truth_schema = common_ds.schema
            truth_src = "_common_metadata"

        return cls(
            args=args,
            files_ds=files_ds,
            metadata_ds=metadata_ds,
            common_ds=common_ds,
            truth_schema=truth_schema,
            truth_src=truth_src,
        )

    def run(self, write_mode: str = "a"):
        self.test_file_sets()
        self.test_is_valid_catalog()
        self.test_num_rows()
        self.test_rowgroup_stats(write_mode=write_mode)
        self.test_schemas()

        self.record_results(mode=write_mode)

    @property
    def results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def truth_schema_plus_common_metadata(self) -> pyarrow.Schema:
        """Copy of truth_schema with hipscat fields and metadata added from common_ds.schema."""
        hipscat_cols = ["Norder", "Dir", "Npix", "_hipscat_index"]
        new_fields = [
            self.common_ds.schema.field(fld) for fld in hipscat_cols if fld not in self.truth_schema.names
        ]

        # use pandas metadata from common_ds but keep all other metadata from truth_schema
        metadata = self.truth_schema.metadata or {}
        metadata[b"pandas"] = self.common_ds.schema.metadata[b"pandas"]

        return pyarrow.schema(list(self.truth_schema) + new_fields).with_metadata(metadata)

    def test_file_sets(self) -> bool:
        test = "file sets"
        description = "Test that files in _metadata match files on disk."
        test_info = dict(test=test, description=description)
        print(f"\nStarting: {description}")

        targets = "_metadata vs files on disk"
        base_dir = str(self.args.input_catalog_path)
        files_ds_files = [f.removeprefix(base_dir).strip("/") for f in self.files_ds.files]
        metadata_ds_files = [f.removeprefix(base_dir).strip("/") for f in self.metadata_ds.files]
        failed_files = list(set(files_ds_files).symmetric_difference(metadata_ds_files))
        passed = len(failed_files) == 0
        self._append_result(passed=passed, target=targets, affected_files=failed_files, **test_info)

        print(f"Result: {'PASSED' if passed else 'FAILED'}")
        return passed

    def test_is_valid_catalog(self) -> bool:
        test = "is valid catalog"
        target = self.args.input_catalog_path
        # [FIXME] How to get the hipscat version?
        description = "Test that this is a valid HiPSCat catalog using hipscat version <VERSION>."
        print(f"\nStarting: {description}")

        passed = hipscat.io.validation.is_valid_catalog(target, strict=True)
        self._append_result(test=test, description=description, passed=passed, target=target.name)
        print(f"Result: {'PASSED' if passed else 'FAILED'}")
        return passed

    def test_num_rows(self) -> bool:
        test = "num rows"
        description = "Test that number of rows are equal."
        test_info = dict(test=test, description=description)
        print(f"\nStarting: {description}")

        # get the number of rows in each file, indexed by file path. we treat this as truth.
        files_df = self._load_nrows(self.files_ds, explicit_count=True)

        # check _metadata
        targets = "_metadata vs file footers"
        print(f"\t{targets}")
        metadata_df = self._load_nrows(self.metadata_ds)
        row_diff = files_df - metadata_df
        failed_frags = row_diff.loc[row_diff.num_rows != 0].index.to_list()
        passed = len(failed_frags) == 0
        self._append_result(passed=passed, target=targets, affected_files=failed_frags, **test_info)

        # check user-supplied total
        if self.args.truth_total_rows is not None:
            targets = "user total vs file footers"
            print(f"\t{targets}")
            _passed = self.args.truth_total_rows == files_df.num_rows.sum()
            self._append_result(passed=_passed, target=targets, **test_info)
        else:
            _passed = True  # this test did not fail. this is only needed for the return value.

        all_passed = all([passed, _passed])
        print(f"Result: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

    def _load_nrows(self, dataset: pyarrow.dataset.Dataset, explicit_count: bool = False) -> pd.DataFrame:
        nrows_df = pd.DataFrame(
            columns=["num_rows", "frag_path"],
            data=[
                (
                    # [TODO] check cpu/ram usage to try to determine if there is a difference here
                    frag.count_rows() if explicit_count else frag.metadata.num_rows,
                    frag.path.removeprefix(str(self.args.input_catalog_path)).strip("/"),
                )
                for frag in dataset.get_fragments()
            ],
        )
        nrows_df = nrows_df.set_index("frag_path").sort_index()
        return nrows_df

    def test_rowgroup_stats(self, *, write_mode: str | None = "a") -> bool:
        test = "rowgroup stats"
        description = "Test that statstistics were recorded for all row groups."
        target = "_metadata"
        test_info = dict(test=test, description=description, target=target)
        print(f"\nStarting: {description}")

        common_truth_schema = self.truth_schema_plus_common_metadata()
        self._distributions_df = None  # start fresh
        try:
            assert set(self.distributions_df.index) == set(common_truth_schema.names)
        except AssertionError:
            passed = False
        else:
            passed = True
        self._append_result(passed=passed, **test_info)
        print(f"Result: {'PASSED' if passed else 'FAILED'}")

        if passed and (write_mode is not None):
            fout = self.args.output_path / self.args.output_distributions_filename
            fout.parent.mkdir(exist_ok=True, parents=True)
            header = False if (write_mode == "a" and fout.is_file()) else True
            self.distributions_df.to_csv(fout, mode=write_mode, header=header, index=True)
            print(f"Distributions written to {fout}")

        return passed

    @property
    def distributions_df(self) -> None:
        if self._distributions_df is not None:
            return self._distributions_df

        print("Gathering distributions (min/max) for fields.")
        common_truth_schema = self.truth_schema_plus_common_metadata()

        try:
            rowgrp_stats = [
                rg.statistics for frag in self.metadata_ds.get_fragments() for rg in frag.row_groups
            ]
        except pyarrow.ArrowTypeError as exc:
            msg = "Distributions failed due to mismatched schemas. Run 'test_schemas' to find problematic files."
            raise pyarrow.ArrowTypeError(msg) from exc

        dist = pd.json_normalize(rowgrp_stats)

        # if dist doesn't contain all expected columns, fail now
        msg = "Statistics not found"
        assert set([c.split(".")[0] for c in dist.columns]) == set(common_truth_schema.names), msg

        min_ = dist[[f"{c}.min" for c in common_truth_schema.names]].min()
        min_ = min_.rename(index={name: name.removesuffix(".min") for name in min_.index})

        max_ = dist[[f"{c}.max" for c in common_truth_schema.names]].max()
        max_ = max_.rename(index={name: name.removesuffix(".max") for name in max_.index})

        self._distributions_df = pd.DataFrame({"minimum": min_, "maximum": max_}).rename_axis(index="field")
        return self._distributions_df

    def test_schemas(self) -> bool:
        test, testmd = "schema", "schema metadata"
        test_info = dict(test=test, description="Test that schemas are equal.")
        testmd_info = dict(test=testmd, description="Test that schema metadata is equal.")
        print(f"\nStarting: {test_info['description']}")

        passed_cm = self._test_schema__common_metadata(test_info, testmd_info)
        passed_md = self._test_schema__metadata(test_info, testmd_info)
        passed_ff = self._test_schema_file_footers(test_info, testmd_info)

        all_passed = all([passed_cm, passed_md, passed_ff])
        print(f"Result: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

    def _test_schema__common_metadata(self, test_info: dict, testmd_info: dict) -> bool:
        pandas_passed = self._test_schema__common_metadata_pandas()

        if self.truth_src == "_common_metadata":
            # no input schema provided => _common_metadata is being used as truth, so skip the rest
            return pandas_passed

        # an input schema was provided as truth, so we need to test _common_metadata against it
        targets = f"_common_metadata vs {self.truth_src}"
        print(f"\t{targets}")
        common_truth_schema = self.truth_schema_plus_common_metadata()

        # check schema and metadata separately because we want to report the results separately
        passed = self.common_ds.schema.equals(common_truth_schema, check_metadata=False)
        self._append_result(passed=passed, target=targets, **test_info)
        passedmd = self.common_ds.schema.metadata == common_truth_schema.metadata
        self._append_result(passed=passedmd, target=targets, **testmd_info)

        return all([pandas_passed, passed, passedmd])

    def _test_schema__common_metadata_pandas(self) -> bool:
        test = "schema metadata"
        description = "Test that pandas metadata contains correct field names and types."
        target = "b'pandas' in _common_metadata"
        test_info = dict(test=test, description=description, target=target)
        print(f"\t{target}")

        common_truth_schema = self.truth_schema_plus_common_metadata()
        base_schema = pyarrow.schema([pyarrow.field(fld.name, fld.type) for fld in common_truth_schema])
        pandas_md = common_truth_schema.pandas_metadata
        pfields = [
            pyarrow.field(pcol["name"], pyarrow.from_numpy_dtype(pcol["pandas_type"]))
            for pcol in pandas_md["columns"]
        ]
        pandas_schema = pyarrow.schema(pfields)

        passed = base_schema.equals(pandas_schema) and (pandas_md["index_columns"] == ["_hipscat_index"])
        self._append_result(passed=passed, **test_info)
        return passed

    def _test_schema__metadata(self, test_info: dict, testmd_info: dict) -> bool:
        targets = f"_metadata vs {self.truth_src}"
        print(f"\t{targets}")
        common_truth_schema = self.truth_schema_plus_common_metadata()

        # check schema and metadata separately because we want to report the results separately
        passed = self.metadata_ds.schema.equals(common_truth_schema, check_metadata=False)
        self._append_result(passed=passed, target=targets, **test_info)
        passedmd = self.metadata_ds.schema.metadata == common_truth_schema.metadata
        self._append_result(passed=passedmd, target=targets, **testmd_info)

        return all([passed, passedmd])

    def _test_schema_file_footers(self, test_info: dict, testmd_info: dict) -> bool:
        targets = f"file footers vs {self.truth_src}"
        print(f"\t{targets}")
        common_truth_schema = self.truth_schema_plus_common_metadata()

        affected_files, affectedmd_files = [], []
        for frag in self.files_ds.get_fragments():
            frag_path = str(Path(frag.path).relative_to(self.args.input_catalog_path))
            # check schema and metadata separately because we want to report the results separately
            if not frag.physical_schema.equals(common_truth_schema, check_metadata=False):
                affected_files.append(frag_path)
            if not frag.physical_schema.metadata == common_truth_schema.metadata:
                affectedmd_files.append(frag_path)

        passed = len(affected_files) == 0
        self._append_result(passed=passed, target=targets, affected_files=affected_files, **test_info)
        passedmd = len(affectedmd_files) == 0
        self._append_result(passed=passedmd, target=targets, affected_files=affectedmd_files, **testmd_info)

        return all([passed, passedmd])

    def _append_result(
        self,
        *,
        test: str,
        target: str,
        description: str,
        passed: bool,
        affected_files: list[str] | None = None,
    ):
        self.results.append(
            Result(
                datetime=now(),
                passed=passed,
                test=test,
                target=target,
                description=description,
                affected_files=affected_files or [],
            )
        )

    def record_results(self, *, mode: str = "a") -> None:
        fout = self.args.output_path / self.args.output_report_filename
        fout.parent.mkdir(exist_ok=True, parents=True)
        header = False if (mode == "a" and fout.is_file()) else True
        self.results_df.to_csv(fout, mode=mode, header=header, index=False)
        print(f"\nVerifier results written to {fout}")
