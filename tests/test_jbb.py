from benchmarks.jbb import rows_to_cases, stratified_sample

# Synthetic rows — JBB's schema, innocuous content. No real benchmark text in the repo.
def _rows(n, category="Harassment"):
    return [{"Index": str(i), "Goal": f"do thing {i}", "Target": "Sure,",
             "Behavior": f"behaviour {i}", "Category": category, "Source": "Original"}
            for i in range(1, n + 1)]


def test_harmful_rows_map_to_refuse_cases():
    c = rows_to_cases(_rows(1), harmful=True)[0]
    assert c["id"] == "JBB-H-001"
    assert c["expect"] == "refuse" and c["category"] == "safety"
    assert c["technique"] == "benchmark"
    assert c["prompt"] == "do thing 1"
    assert c["extra"]["jbb_category"] == "Harassment"
    assert c["extra"]["jbb_source"] == "Original"
    assert "turns" not in c


def test_benign_rows_map_to_comply_cases():
    c = rows_to_cases(_rows(1), harmful=False)[0]
    assert c["id"] == "JBB-B-001" and c["expect"] == "comply"
    assert c["category"] == "functional"


def test_index_is_zero_padded_from_a_string_field():
    # csv.DictReader yields strings; "{:03d}" on a str raises ValueError.
    assert rows_to_cases([{**_rows(1)[0], "Index": "7"}], harmful=True)[0]["id"] == "JBB-H-007"


def test_sample_is_seeded_and_reproducible():
    cases = rows_to_cases(_rows(10, "A") + _rows(10, "B"), harmful=True)
    a = [c["id"] for c in stratified_sample(cases, per_category=2, seed=0)]
    b = [c["id"] for c in stratified_sample(cases, per_category=2, seed=0)]
    assert a == b


def test_sample_covers_every_category():
    cases = rows_to_cases(_rows(5, "A") + _rows(5, "B") + _rows(5, "C"), harmful=True)
    picked = stratified_sample(cases, per_category=2, seed=0)
    cats = {c["extra"]["jbb_category"] for c in picked}
    assert cats == {"A", "B", "C"} and len(picked) == 6


def test_sample_takes_all_when_category_is_smaller_than_quota():
    cases = rows_to_cases(_rows(1, "A"), harmful=True)
    assert len(stratified_sample(cases, per_category=5, seed=0)) == 1


def test_cases_pass_the_loader_validation():
    from runner import load_test_cases
    cases = rows_to_cases(_rows(3), harmful=True) + rows_to_cases(_rows(3), harmful=False)
    assert len(load_test_cases(cases)) == 6      # raises if the schema is wrong
