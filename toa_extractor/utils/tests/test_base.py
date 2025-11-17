import os
import tempfile
from PIL import Image
import time

from toa_extractor.utils.base import (
    safe_get_key,
    root_name,
    output_name,
    encode_image_file,
    get_file_time,
    process_and_copy_image,
    search_substring_in_list,
)


class TestSafeGetKey:
    def test_single_string_key(self):
        d = {"a": 2, "b": 3}
        assert safe_get_key(d, "a", 1) == 2

    def test_single_list_key(self):
        d = {"a": 2, "b": 3}
        assert safe_get_key(d, ["a"], 1) == 2

    def test_missing_key_returns_default(self):
        d = {"a": 2, "b": 3}
        assert safe_get_key(d, ["asdf"], 1) == 1

    def test_nested_keys(self):
        d = {"a": 2, "b": 2, "c": {"d": 12, "e": 45, "f": {"g": 32}}}
        assert safe_get_key(d, ["c", "f", "g"], 1) == 32

    def test_nested_missing_key(self):
        d = {"a": 2, "b": 2, "c": {"d": 12, "e": 45, "f": {"g": 32}}}
        assert safe_get_key(d, ["c", "f", "asdfasfd"], 1) == 1

    def test_nested_missing_intermediate_key(self):
        d = {"a": 2, "b": 2, "c": {"d": 12, "e": 45, "f": {"g": 32}}}
        assert safe_get_key(d, ["c", "fasdfasf", "g"], 1) == 1


class TestRootName:
    def test_evt_gz_extension(self):
        assert root_name("file.evt.gz") == "file"

    def test_ds_extension(self):
        assert root_name("file.ds") == "file"

    def test_numbered_ds_extension(self):
        assert root_name("file.1.ds") == "file.1"

    def test_numbered_ds_gz_extension(self):
        assert root_name("file.1.ds.gz") == "file.1"

    def test_multiple_compression_extensions(self):
        assert root_name("file.1.ds.gz.Z") == "file.1"

    def test_zip_extension(self):
        assert root_name("file.evt.zip") == "file"

    def test_bz_extension(self):
        assert root_name("file.evt.bz") == "file"


class TestOutputName:
    def test_basic_output_name(self):
        assert output_name("file.evt.gz", "v3", "folded.h5") == "file_v3_folded.h5"

    def test_suffix_with_multiple_underscores(self):
        assert output_name("file.evt.gz", "v3", "__folded.h5") == "file_v3_folded.h5"

    def test_version_with_underscore(self):
        assert output_name("file.evt.gz", "_v3", "folded.h5") == "file_v3_folded.h5"

    def test_suffix_with_dot(self):
        assert output_name("file.evt.gz", "v3", ".h5") == "file_v3.h5"

    def test_none_version(self):
        result = output_name("file.evt.gz", None, "folded.h5")
        assert "file" in result and "folded.h5" in result


class TestEncodeImageFile:
    def test_encode_image_file(self):
        # Create a temporary image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img = Image.new("RGB", (100, 100), color="red")
            img.save(tmp.name, "JPEG")
            tmp_path = tmp.name

        try:
            result = encode_image_file(tmp_path)
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            os.unlink(tmp_path)


class TestGetFileTime:
    def test_get_file_time(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            file_time = get_file_time(tmp_path)
            assert isinstance(file_time, (int, float))
            time.sleep(1)
            now = time.time()
            assert file_time < now
            assert file_time > now - 60
        finally:
            os.unlink(tmp_path)


class TestProcessAndCopyImage:
    def test_same_source_and_target(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(tmp.name, "JPEG")
            tmp_path = tmp.name
        creation_time = get_file_time(tmp_path)
        try:
            config = {"max_width": 512, "quality": 85, "format": "JPEG"}
            result = process_and_copy_image(tmp_path, tmp_path, config)
            assert result == tmp_path
            assert get_file_time(result) == creation_time
        finally:
            os.unlink(tmp_path)

    def test_resize_large_image(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "source.jpg")
            target_path = os.path.join(tmp_dir, "target.jpg")

            # Create large image
            img = Image.new("RGB", (1000, 1000), color="green")
            img.save(source_path, "JPEG")

            config = {"max_width": 512, "quality": 85, "format": "JPEG"}
            result = process_and_copy_image(source_path, target_path, config)

            assert os.path.exists(result)
            processed_img = Image.open(result)
            assert processed_img.width <= 512
            processed_img.close()

    def test_target_newer_than_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "source.jpg")
            target_path = os.path.join(tmp_dir, "target.jpg")

            img = Image.new("RGB", (100, 100), color="red")
            img.save(source_path, "JPEG")
            img.save(target_path, "JPEG")
            time.sleep(1)
            # Touch target to make it newer
            os.utime(target_path, None)
            creation_time = get_file_time(target_path)

            config = {"max_width": 512, "quality": 85, "format": "JPEG"}
            result = process_and_copy_image(source_path, target_path, config)
            assert result == target_path
            assert get_file_time(result) == creation_time


class TestSearchSubstringInList:
    def test_single_match(self):
        assert search_substring_in_list("a", ["a", "b", "c"]) == ["a"]

    def test_multiple_matches(self):
        assert search_substring_in_list("a", ["a", "b", "c", "ab"]) == ["a", "ab"]

    def test_substring_in_middle(self):
        result = search_substring_in_list("a", ["a", "b", "c", "ab", "ba"])
        assert result == ["a", "ab", "ba"]

    def test_no_matches(self):
        assert search_substring_in_list("x", ["a", "b", "c"]) == []

    def test_empty_list(self):
        assert search_substring_in_list("a", []) == []
