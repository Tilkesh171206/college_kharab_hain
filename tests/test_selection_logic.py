from placement_detector import looks_like_selection


def test_selection_email():
    assert looks_like_selection(
        "Congratulations! Students have got selected through the campus placement drive.",
        "Congratulations :: Selections - ABC"
    )


def test_non_selection_registration_email():
    assert not looks_like_selection(
        "Please register on the portal for future hiring updates.",
        "INFORMATION: Registration on TCS Campus Commune Portal"
    )
