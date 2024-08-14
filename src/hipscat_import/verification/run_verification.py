"""Run pass/fail checks and generate verification report of existing hipscat table."""

import pandas as pd
import pyarrow.dataset
import re
from hipscat_import.verification.arguments import VerificationArguments


def run(args):
    """Run verification pipeline."""
    if not args:
        raise TypeError("args is required and should be type VerificationArguments")
    if not isinstance(args, VerificationArguments):
        raise TypeError("args must be type VerificationArguments")

    # implement everything else.
    raise NotImplementedError("Verification not yet implemented.")


def _verify_parquet_files(args):
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
    schema = pyarrow.dataset.parquet_dataset(f"{args.input_catalog_path}/_common_metadata").schema

    schemas_passed = _check_schemas(files_ds, schema)
    file_set_passed = _check_file_set(args, files_ds)
    statistics_passed = _check_statistics(files_ds, schema.names)
    num_rows_passed = _check_num_rows(args, files_ds)

    return all([schemas_passed, file_set_passed, statistics_passed, num_rows_passed])


def _check_schemas(files_ds, schema):
    # Check schema against _common_metadata
    # [TODO] Are there cases where this will fail but the schema is actually valid? Maybe if a column has all nulls?
    schemas_passed = all(
        [frag.physical_schema.equals(schema, check_metadata=True) for frag in files_ds.get_fragments()]
    )
    return schemas_passed


def _check_file_set(args, files_ds):
    # Check that parquet files on disk == files in _metadata
    metadata_ds = pyarrow.dataset.parquet_dataset(f"{args.input_catalog_path}/_metadata")
    # Paths in hipscat _metadata have a double slash ("//") after the dataset name. need to get rid of it.
    file_set_passed = set(files_ds.files) == set(f.replace("//", "/") for f in metadata_ds.files)
    return file_set_passed


def _check_statistics(files_ds, column_names):
    # Check that row group stats were written
    statistics_passed = all(
        [
            set(rg.statistics.keys()) == set(column_names)
            for frag in files_ds.get_fragments()
            for rg in frag.row_groups
        ]
    )
    return statistics_passed


def _check_num_rows(args, files_ds):
    # Check that num rows in each file matches partition_info.csv
    partition_cols = ["Norder", "Dir", "Npix"]
    part_df = pd.read_csv(f"{args.input_catalog_path}/partition_info.csv").set_index(partition_cols)
    files_df = pd.DataFrame(
        [
            (
                int(re.search(r"Norder=(\d+)", frag.path).group(1)),
                int(re.search(r"Dir=(\d+)", frag.path).group(1)),
                int(re.search(r"Npix=(\d+)", frag.path).group(1)),
                frag.metadata.num_rows,
            )
            for frag in files_ds.get_fragments()
        ],
        columns=["Norder", "Dir", "Npix", "num_rows"],
    ).set_index(partition_cols)
    num_rows_passed = part_df.equals(files_df)
    return num_rows_passed
