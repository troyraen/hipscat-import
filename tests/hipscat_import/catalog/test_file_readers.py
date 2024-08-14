"""Test dataframe-generating file readers"""

import hipscat.io.write_metadata as io
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hipscat.catalog.catalog import CatalogInfo

from hipscat_import.catalog.file_readers import (
    CsvReader,
    FitsReader,
    IndexedCsvReader,
    IndexedParquetReader,
    ParquetReader,
    get_file_reader,
)


# pylint: disable=redefined-outer-name
@pytest.fixture
def basic_catalog_info():
    info = {
        "catalog_name": "test_catalog",
        "catalog_type": "object",
        "total_rows": 100,
        "ra_column": "ra",
        "dec_column": "dec",
    }
    return CatalogInfo(**info)


def test_unknown_file_type():
    """File reader factory method should fail for unknown file types"""
    with pytest.raises(NotImplementedError):
        get_file_reader("")
    with pytest.raises(NotImplementedError):
        get_file_reader("unknown")


def test_file_exists(small_sky_dir):
    """File reader factory method should fail for missing files or directories"""
    with pytest.raises(FileNotFoundError, match="File not found"):
        next(CsvReader().read("foo_not_really_a_path"))
    with pytest.raises(FileNotFoundError, match="Directory found at path"):
        next(CsvReader().read(small_sky_dir))


def test_csv_reader(small_sky_single_file):
    """Verify we can read the csv file into a single data frame."""
    total_chunks = 0
    for frame in CsvReader().read(small_sky_single_file):
        total_chunks += 1
        assert len(frame) == 131

    assert total_chunks == 1


def test_csv_gzip_reader(formats_dir):
    """Verify we can read the csv file into a single data frame."""
    total_chunks = 0
    for frame in CsvReader(compression="zip").read(formats_dir / "catalog.zip"):
        total_chunks += 1
        assert len(frame) == 131

    assert total_chunks == 1

    total_chunks = 0
    for frame in CsvReader(compression="gzip").read(formats_dir / "catalog.csv.gz"):
        total_chunks += 1
        assert len(frame) == 131

    assert total_chunks == 1


def test_csv_reader_chunks(small_sky_single_file):
    """Verify we can read the csv file into multiple data frames."""
    total_chunks = 0
    for frame in CsvReader(chunksize=15).read(small_sky_single_file):
        total_chunks += 1
        assert len(frame) <= 15

    assert total_chunks == 9

    total_chunks = 0
    for frame in CsvReader(chunksize=1).read(small_sky_single_file):
        total_chunks += 1
        assert len(frame) == 1

    assert total_chunks == 131


def test_csv_reader_no_headers(small_sky_single_file):
    """Verify we can read the csv file without a header row."""
    total_chunks = 0
    for frame in CsvReader(header=None).read(small_sky_single_file):
        total_chunks += 1
        # The "header" row is interpreted as just another data row
        assert len(frame) == 132

    assert total_chunks == 1


def test_csv_reader_parquet_metadata(small_sky_single_file, tmp_path):
    """Verify we can read the csv file without a header row."""
    small_sky_schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("ra", pa.float64()),
            pa.field("dec", pa.float64()),
            pa.field("ra_error", pa.float64()),
            pa.field("dec_error", pa.float64()),
        ]
    )
    schema_file = tmp_path / "metadata.parquet"
    pq.write_metadata(
        small_sky_schema,
        schema_file,
    )

    frame = next(
        CsvReader(schema_file=schema_file, parquet_kwargs={"dtype_backend": "numpy_nullable"}).read(
            small_sky_single_file
        )
    )
    assert len(frame) == 131

    column_types = frame.dtypes.to_dict()
    expected_column_types = {
        "id": pd.Int64Dtype(),
        "ra": pd.Float64Dtype(),
        "dec": pd.Float64Dtype(),
        "ra_error": pd.Float64Dtype(),
        "dec_error": pd.Float64Dtype(),
    }
    assert np.all(column_types == expected_column_types)


def test_csv_reader_kwargs(small_sky_single_file):
    """Verify we can read the csv file using kwargs passed to read_csv call."""

    ## Input file has 5 columns: ["id", "ra", "dec", "ra_error", "dec_error"]
    frame = next(CsvReader(usecols=["id", "ra", "dec"]).read(small_sky_single_file))
    assert len(frame) == 131

    assert len(frame.columns) == 3
    assert np.all(frame.columns == ["id", "ra", "dec"])


def test_csv_reader_pipe_delimited(formats_pipe_csv, tmp_path):
    """Verify we can read a pipe-delimited csv file without a header row."""
    total_chunks = 0
    for frame in CsvReader(header=None, sep="|").read(formats_pipe_csv):
        total_chunks += 1
        assert len(frame) == 3
        assert np.all(frame[0] == ["AA", "BB", "CC"])

    assert total_chunks == 1

    ## Provide header names and types in a few formats
    frame = next(
        CsvReader(
            column_names=["letters", "ints", "empty", "numeric"],
            header=None,
            sep="|",
        ).read(formats_pipe_csv)
    )
    assert len(frame) == 3
    assert np.all(frame["letters"] == ["AA", "BB", "CC"])
    column_types = frame.dtypes.to_dict()
    expected_column_types = {
        "letters": object,
        "ints": int,
        "empty": float,
        "numeric": float,
    }
    assert np.all(column_types == expected_column_types)

    frame = next(
        CsvReader(
            column_names=["letters", "ints", "empty", "numeric"],
            type_map={
                "letters": object,
                "ints": int,
                "empty": "Int64",
                "numeric": int,
            },
            header=None,
            sep="|",
        ).read(formats_pipe_csv)
    )
    assert len(frame) == 3
    assert np.all(frame["letters"] == ["AA", "BB", "CC"])
    column_types = frame.dtypes.to_dict()
    expected_column_types = {
        "letters": object,
        "ints": int,
        "empty": pd.Int64Dtype(),
        "numeric": int,
    }
    assert np.all(column_types == expected_column_types)

    parquet_schema_types = pa.schema(
        [
            pa.field("letters", pa.string()),
            pa.field("ints", pa.int64()),
            pa.field("empty", pa.int64()),
            pa.field("numeric", pa.int64()),
        ]
    )
    schema_file = tmp_path / "metadata.parquet"
    pq.write_metadata(parquet_schema_types, schema_file)

    frame = next(
        CsvReader(
            schema_file=schema_file, header=None, sep="|", parquet_kwargs={"dtype_backend": "numpy_nullable"}
        ).read(formats_pipe_csv)
    )

    assert len(frame) == 3
    assert np.all(frame["letters"] == ["AA", "BB", "CC"])
    column_types = frame.dtypes.to_dict()
    expected_column_types = {
        "letters": pd.StringDtype(),
        "ints": pd.Int64Dtype(),
        "empty": pd.Int64Dtype(),
        "numeric": pd.Int64Dtype(),
    }
    assert np.all(column_types == expected_column_types)


def test_csv_reader_provenance_info(tmp_path, basic_catalog_info):
    """Test that we get some provenance info and it is parseable into JSON."""
    reader = CsvReader(
        header=None,
        sep="|",
        column_names=["letters", "ints", "empty", "numeric"],
        type_map={
            "letters": object,
            "ints": int,
            "empty": "Int64",
            "numeric": int,
        },
        storage_options={"user_name": "user_pii", "user_key": "SECRETS!"},
    )
    provenance_info = reader.provenance_info()
    catalog_base_dir = tmp_path / "test_catalog"
    catalog_base_dir.mkdir(parents=True)
    io.write_provenance_info(catalog_base_dir, basic_catalog_info, provenance_info)

    with open(catalog_base_dir / "provenance_info.json", "r", encoding="utf-8") as file:
        data = file.read()
        assert "test_catalog" in data
        assert "REDACTED" in data
        assert "user_pii" not in data
        assert "SECRETS" not in data


def test_indexed_csv_reader(indexed_files_dir):
    # Chunksize covers all the inputs.
    total_chunks = 0
    for frame in IndexedCsvReader(chunksize=10_000).read(indexed_files_dir / "csv_list_single.txt"):
        total_chunks += 1
        assert len(frame) == 131

    assert total_chunks == 1

    # Chunksize requires splitting into just a few batches.
    total_chunks = 0
    for frame in IndexedCsvReader(chunksize=60).read(indexed_files_dir / "csv_list_single.txt"):
        total_chunks += 1
        assert len(frame) < 60

    assert total_chunks == 3

    # Requesting a very small chunksize. This will split up reads on the CSV.
    total_chunks = 0
    for frame in IndexedCsvReader(chunksize=5).read(indexed_files_dir / "csv_list_single.txt"):
        total_chunks += 1
        assert len(frame) <= 5

    assert total_chunks == 29


def test_parquet_reader(parquet_shards_shard_44_0):
    """Verify we can read the parquet file into a single data frame."""
    total_chunks = 0
    for frame in ParquetReader().read(parquet_shards_shard_44_0):
        total_chunks += 1
        assert len(frame) == 7

    assert total_chunks == 1


def test_parquet_reader_chunked(parquet_shards_shard_44_0):
    """Verify we can read the parquet file into a single data frame."""
    total_chunks = 0
    for frame in ParquetReader(chunksize=1).read(parquet_shards_shard_44_0):
        total_chunks += 1
        assert len(frame) == 1
    assert total_chunks == 7


def test_indexed_parquet_reader(indexed_files_dir):
    # Chunksize covers all the inputs.
    total_chunks = 0
    for frame in get_file_reader("indexed_parquet", chunksize=10_000).read(
        indexed_files_dir / "parquet_list_single.txt"
    ):
        total_chunks += 1
        assert len(frame) == 131

    assert total_chunks == 1

    # Chunksize requires splitting into just a few batches.
    total_chunks = 0
    for frame in IndexedParquetReader(chunksize=60).read(indexed_files_dir / "parquet_list_single.txt"):
        total_chunks += 1
        assert len(frame) < 60

    assert total_chunks == 3

    # Requesting a very small chunksize. This will split up reads on the CSV.
    total_chunks = 0
    for frame in IndexedParquetReader(chunksize=5).read(indexed_files_dir / "parquet_list_single.txt"):
        total_chunks += 1
        assert len(frame) <= 5

    assert total_chunks == 29


def test_parquet_reader_provenance_info(tmp_path, basic_catalog_info):
    """Test that we get some provenance info and it is parseable into JSON."""
    reader = ParquetReader(chunksize=1)
    provenance_info = reader.provenance_info()
    catalog_base_dir = tmp_path / "test_catalog"
    catalog_base_dir.mkdir(parents=True)
    io.write_provenance_info(catalog_base_dir, basic_catalog_info, provenance_info)


def test_parquet_reader_columns(parquet_shards_shard_44_0):
    """Verify we can read a subset of columns."""
    column_subset = ["id", "dec"]

    # test column_names class property
    for frame in ParquetReader(column_names=column_subset).read(parquet_shards_shard_44_0):
        assert set(frame.columns) == set(column_subset)

    # test read_columns kwarg
    for frame in ParquetReader().read(parquet_shards_shard_44_0, read_columns=column_subset):
        assert set(frame.columns) == set(column_subset)


def test_read_fits(formats_fits):
    """Success case - fits file that exists being read as fits"""
    total_chunks = 0
    for frame in FitsReader().read(formats_fits):
        total_chunks += 1
        assert len(frame) == 131

    assert total_chunks == 1


def test_read_fits_chunked(formats_fits):
    """Success case - fits file that exists being read as fits in chunks"""
    total_chunks = 0
    for frame in FitsReader(chunksize=1).read(formats_fits):
        total_chunks += 1
        assert len(frame) == 1

    assert total_chunks == 131


def test_read_fits_columns(formats_fits):
    """Success case - column filtering on reading fits file"""
    frame = next(FitsReader(column_names=["id", "ra", "dec"]).read(formats_fits))
    assert list(frame.columns) == ["id", "ra", "dec"]

    frame = next(FitsReader(column_names=["id", "ra", "dec"]).read(formats_fits, read_columns=["ra", "dec"]))
    assert list(frame.columns) == ["ra", "dec"]

    frame = next(FitsReader(skip_column_names=["ra_error", "dec_error"]).read(formats_fits))
    assert list(frame.columns) == ["id", "ra", "dec", "test_id"]

    frame = next(
        FitsReader(skip_column_names=["ra_error", "dec_error"]).read(formats_fits, read_columns=["ra", "dec"])
    )
    assert list(frame.columns) == ["ra", "dec"]


def test_fits_reader_provenance_info(tmp_path, basic_catalog_info):
    """Test that we get some provenance info and it is parseable into JSON."""
    reader = FitsReader()
    provenance_info = reader.provenance_info()
    catalog_base_dir = tmp_path / "test_catalog"
    catalog_base_dir.mkdir(parents=True)
    io.write_provenance_info(catalog_base_dir, basic_catalog_info, provenance_info)
