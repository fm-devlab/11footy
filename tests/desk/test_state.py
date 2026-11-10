from desk.state import keep, read_last_chosen, read_state, remember


def test_the_last_choice_is_remembered(tmp_path):
    file = tmp_path / "state" / "state.json"
    assert read_last_chosen(file) == []
    folder = tmp_path / "release"
    folder.mkdir()
    dataset = tmp_path / "d.db"
    dataset.write_bytes(b"")
    remember([folder, dataset], file)
    assert read_last_chosen(file) == [folder, dataset]
    dataset.unlink()
    assert read_last_chosen(file) == [], "a choice with a file gone since is forgotten"
    file.write_text("not json")
    assert read_last_chosen(file) == []


def test_the_look_and_size_are_kept_beside_the_choice(tmp_path):
    file = tmp_path / "state.json"
    folder = tmp_path / "release"
    folder.mkdir()
    remember([folder], file)
    keep({"theme": "dark", "geometry": "1200x700+10+10"}, file)
    remember([folder], file)
    assert read_state(file) | {} == {"chosen": [str(folder)], "theme": "dark",
                                     "geometry": "1200x700+10+10"}  # fmt: skip
    file.write_text("[1, 2]")
    assert read_state(file) == {}
