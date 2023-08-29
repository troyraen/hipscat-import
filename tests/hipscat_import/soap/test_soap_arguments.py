import pytest

from hipscat_import.soap.arguments import SoapArguments


def test_none():
    """No arguments provided. Should error for required args."""
    with pytest.raises(ValueError):
        SoapArguments()


def test_empty_required(tmp_path, small_sky_object_catalog, small_sky_source_catalog):
    """All non-runtime arguments are required."""

    ## List of required args:
    ##  - match expression that should be found when missing
    ##  - default value
    required_args = [
        ["object_catalog_dir", small_sky_object_catalog],
        ["object_id_column", "id"],
        ["source_catalog_dir", small_sky_source_catalog],
        ["source_object_id_column", "object_id"],
        ["output_catalog_name", "small_sky_association"],
        ["output_path", tmp_path],
    ]

    ## For each required argument, check that a ValueError is raised that matches the
    ## expected name of the missing param.
    for index, args in enumerate(required_args):
        test_args = [
            list_args[1] if list_index != index else None
            for list_index, list_args in enumerate(required_args)
        ]

        with pytest.raises(ValueError, match=args[0]):
            SoapArguments(
                object_catalog_dir=test_args[0],
                object_id_column=test_args[1],
                source_catalog_dir=test_args[2],
                source_object_id_column=test_args[3],
                output_catalog_name=test_args[4],
                output_path=test_args[5],
                ## always set these False
                progress_bar=False,
                overwrite=True,
            )


def test_compute_partition_size(tmp_path, small_sky_object_catalog, small_sky_source_catalog):
    """Test validation of compute_partition_size."""
    with pytest.raises(ValueError, match="compute_partition_size"):
        SoapArguments(
            object_catalog_dir=small_sky_object_catalog,
            object_id_column="id",
            source_catalog_dir=small_sky_source_catalog,
            source_object_id_column="object_id",
            output_catalog_name="small_sky_association",
            output_path=tmp_path,
            progress_bar=False,
            compute_partition_size=10,  ## not a valid option
            overwrite=True,
        )