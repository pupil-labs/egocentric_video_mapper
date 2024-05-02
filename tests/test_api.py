import pupil_labs.action_cam_mapper as this_project


def test_package_metadata() -> None:
    assert hasattr(this_project, "__version__")
