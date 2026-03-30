from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from architxt.tree import NodeLabel, NodeType, has_type

if TYPE_CHECKING:
    from collections.abc import Collection

    from langchain_core.language_models import BaseChatModel

    from architxt.schema import Schema
    from architxt.tree import Forest

__all__ = ['Renaming', 'apply_renaming', 'llm_group_labelling', 'llm_relation_labelling']

GROUP_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "You are a precise data architect. "
            "Return ONLY the SNAKE_CASE name for the database table name. "
            "Do not include any other text."
        ),
        # Few-shot example to anchor the behavior
        HumanMessage(
            "Sample Data: [{'id': 1, 'email': 'a@b.com'}, {'id': 2, 'email': 'c@d.com'}]\n"
            "Current Name: Tbl1\n"
            "Attributes: id, email\n"
            "Suggested Name:"
        ),
        AIMessage("user_accounts"),
        # The actual task
        HumanMessagePromptTemplate.from_template(
            "Sample Data: {samples}\nCurrent Name: {name}\nAttributes: {attributes}\nSuggested Name:"
        ),
    ]
)

RELATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "You are a precise data architect. "
            "Return ONLY the SNAKE_CASE name for the relationship between these tables. "
            "Do not include any other text."
        ),
        # Few-shot example
        HumanMessage(
            "Table A: users\nTable B: orders\nCurrent Relationship Name: link_1\nSuggested Relationship Name:"
        ),
        AIMessage("user_orders"),
        # The actual task
        HumanMessagePromptTemplate.from_template(
            "Table A: {left}\nTable B: {right}\nCurrent Relationship Name: {name}\nSuggested Relationship Name:"
        ),
    ]
)


@dataclasses.dataclass(frozen=True)
class Renaming:
    node_type: NodeType
    old_name: str
    new_name: str


def llm_group_labelling(
    schema: Schema,
    llm: BaseChatModel,
    *,
    forest: Forest | None = None,
    sample_size: int = 5,
) -> set[Renaming]:
    """
    Get a group renaming for a forest using an LLM.

    :param schema: The schema to relabel.
    :param llm: The LLM model to use.
    :param forest: The forest to relabel, needed to provide sample data.
    :param sample_size: Number of sample instances to provide to the LLM for each group.
    :return: A set of renaming for groups.
    """
    renames: set[Renaming] = set()
    datasets = schema.extract_datasets(forest) if forest else {}

    chain = GROUP_PROMPT | llm.bind(stop=["\n", " ", "."]) | StrOutputParser()

    for group in schema.groups:
        attributes = ", ".join(group.entities)
        samples = "No sample data"

        group_dataset = datasets.get(group.name)
        if group_dataset is not None and not group_dataset.empty:
            samples = group_dataset.head(sample_size).to_json(index=False, orient='records')

        response = chain.invoke({"name": group.name, "attributes": attributes, "samples": samples})
        new_name = response.replace("`", "").strip().replace(" ", "_").upper()

        if not new_name or group.name == new_name:
            continue

        renames.add(Renaming(NodeType.GROUP, group.name, new_name))

    return renames


def llm_relation_labelling(
    schema: Schema,
    llm: BaseChatModel,
    *,
    group_renames: Collection[Renaming] | None = None,
) -> set[Renaming]:
    """
    Get a renaming of relations for a forest using an LLM.

    :param schema: The schema to relabel.
    :param llm: The LLM model to use.
    :param group_renames: A collection of renaming for groups to provide context.
    :return: The renaming for relations.
    """
    group_renames_dict = (
        {r.old_name: r.new_name for r in group_renames if r.node_type == NodeType.GROUP} if group_renames else {}
    )

    chain = RELATION_PROMPT | llm.bind(stop=["\n", " ", "."]) | StrOutputParser()

    renames: set[Renaming] = set()
    for relation in schema.relations:
        left_name = group_renames_dict.get(relation.left, relation.left)
        right_name = group_renames_dict.get(relation.right, relation.right)

        response = chain.invoke({"left": left_name, "right": right_name, "name": relation.name})
        new_name = response.replace("`", "").strip().replace(" ", "_").upper()

        if not new_name or relation.name == new_name:
            continue

        renames.add(Renaming(NodeType.REL, relation.name, new_name))

    return renames


def apply_renaming(forest: Forest, renames: Collection[Renaming]) -> None:
    """
    Apply a collection of renaming to a forest.

    :param forest: The forest to modify in-place.
    :param renames: The collection of renaming to apply.
    """
    renames_dict = {(r.node_type, r.old_name): r.new_name for r in renames}

    for tree in forest:
        for subtree in tree.subtrees():
            if not has_type(subtree):
                continue

            key = (subtree.label.type, subtree.label.name)
            if key in renames_dict:
                subtree.label = NodeLabel(subtree.label.type, renames_dict[key])
