import json
from pathlib import Path


def test_knowledge_base_schema_and_counts():
    kb_path = Path(__file__).resolve().parent.parent / "knowledge_base.json"
    entries = json.loads(kb_path.read_text(encoding="utf-8"))
    assert 12 <= len(entries) <= 20
    ids = [e["id"] for e in entries]
    assert ids == [f"G{i:02d}" for i in range(1, 17)]
    assert len(set(ids)) == len(ids)

    for entry in entries:
        for key in ("id", "title", "text", "source"):
            assert isinstance(entry.get(key), str) and entry[key].strip()

    place_entries = [e for e in entries if e["id"] <= "G12"]
    assert len(place_entries) == 12
    for place in place_entries:
        assert isinstance(place.get("category"), str) and place["category"].strip()
        tags = place.get("tags")
        assert isinstance(tags, list) and len(tags) >= 3
        assert all(isinstance(tag, str) and tag.strip() for tag in tags)
