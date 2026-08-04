import numpy as np

from pipeline.v7.phase1.dev_eval import _regions, _seam_boundaries, center_crop_to


def test_center_crop_480_to_464_is_deterministic():
    frames = np.zeros((2, 480, 832, 3), dtype=np.uint8)
    cropped, crop = center_crop_to(frames, 464, 832)
    assert cropped.shape == (2, 464, 832, 3)
    assert crop == {"top": 8, "bottom": 8, "left": 0, "right": 0}


def test_regions_use_half_open_intervals():
    assert _regions(9, (10, 20), (0, 5)) == ("non_support", False)
    assert _regions(10, (10, 20), (0, 5)) == ("support", False)
    assert _regions(19, (10, 20), (0, 5)) == ("support", False)
    assert _regions(20, (10, 20), (0, 5)) == ("non_support", False)
    assert _regions(4, (10, 20), (0, 5)) == ("non_support", True)
    assert _regions(5, (10, 20), (0, 5)) == ("non_support", False)


def test_seam_boundaries_are_owned_output_ends():
    provenance = {
        "actual_output_frames": 405,
        "windows": [
            {"owned_end": 73}, {"owned_end": 146}, {"owned_end": 405},
        ],
    }
    assert _seam_boundaries(provenance) == [73, 146]
