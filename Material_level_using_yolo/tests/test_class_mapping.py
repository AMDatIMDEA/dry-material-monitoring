from __future__ import annotations

import pytest

from material_level_yolo.domain import SemanticRole
from material_level_yolo.errors import ModelCompatibilityError
from material_level_yolo.inference import normalize_model_names, resolve_semantic_classes


def test_role_resolution_uses_model_names_not_source_dataset_ids() -> None:
    mapping = resolve_semantic_classes(
        {9: "Powder", 2: "Empty", 4: "unrelated"},
        SemanticRole("Powder"),
        SemanticRole("Empty"),
    )
    assert mapping.material_id == 9
    assert mapping.empty_id == 2


def test_names_list_is_supported_and_original_spelling_is_retained() -> None:
    mapping = resolve_semantic_classes(
        [" EMPTY ", "powder"],
        SemanticRole("Powder"),
        SemanticRole("Empty"),
    )
    assert mapping.material_id == 1
    assert mapping.empty_id == 0
    assert mapping.empty_name == "EMPTY"


def test_explicit_ids_are_assertions_not_substitutes_for_names() -> None:
    mapping = resolve_semantic_classes(
        {0: "Empty", 1: "Powder"},
        SemanticRole("Powder", explicit_id=1),
        SemanticRole("Empty", explicit_id=0),
    )
    assert mapping.material_id == 1
    with pytest.raises(ModelCompatibilityError, match="maps to 'Empty'"):
        resolve_semantic_classes(
            {0: "Empty", 1: "Powder"},
            SemanticRole("Powder", explicit_id=0),
            SemanticRole("Empty", explicit_id=1),
        )


def test_wrong_class_mapping_is_rejected() -> None:
    with pytest.raises(ModelCompatibilityError, match="'Powder' is absent"):
        resolve_semantic_classes(
            {0: "polymer", 1: "Empty"},
            SemanticRole("Powder"),
            SemanticRole("Empty"),
        )


def test_ambiguous_class_mapping_is_rejected() -> None:
    with pytest.raises(ModelCompatibilityError, match="ambiguous"):
        resolve_semantic_classes(
            {0: "Powder", 1: " powder ", 2: "Empty"},
            SemanticRole("Powder"),
            SemanticRole("Empty"),
        )


@pytest.mark.parametrize("names", [{}, None, {0: ""}, {True: "Powder"}, {1.5: "Powder"}])
def test_unusable_model_names_are_rejected(names) -> None:
    with pytest.raises(ModelCompatibilityError):
        normalize_model_names(names)
