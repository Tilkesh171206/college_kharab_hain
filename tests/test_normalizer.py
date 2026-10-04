from models import SelectionRecord
from normalizer import normalize_record


def test_selected_count_from_students():
    record = SelectionRecord(
        company="Example",
        selected_students=[
            {"name": "A", "branch": "CSE"},
            {"name": "B", "branch": "ECE"},
        ],
    )
    normalized = normalize_record(record)
    assert normalized["selected_count"] == 2
