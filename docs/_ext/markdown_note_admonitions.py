from __future__ import annotations

import re
from typing import Optional

from docutils import nodes
from sphinx.application import Sphinx

# Match our standardized note markers and reasonable fallbacks.
NOTE_RE = re.compile(r"^\s*(?:ℹ️|ⓘ|\(i\)|i|!|‼️)?\s*Note\s*:?\s*", re.IGNORECASE)
IMPORTANT_RE = re.compile(r"^\s*(?:⚠️|!|‼️)?\s*Important\s*:?\s*", re.IGNORECASE)


def _gather_text(node: nodes.Node) -> str:
    return "".join(t.astext() for t in node.traverse(nodes.Text))


def _strip_left_chars_inplace(node: nodes.Element, chars_to_strip: int) -> int:
    """
    Remove exactly `chars_to_strip` characters from the left across Text nodes
    within `node`, preserving inline structure as much as possible.
    Returns the number of characters still to strip after processing this node.
    """
    if chars_to_strip <= 0:
        return 0

    # Work on a copy of the child list as we will mutate the original
    for child in list(node.children):
        if chars_to_strip <= 0:
            break

        if isinstance(child, nodes.Text):
            text = child.astext()
            if len(text) <= chars_to_strip:
                # Remove entire text node
                child.parent.remove(child)
                chars_to_strip -= len(text)
            else:
                # Trim prefix from this text node
                new_text = text[chars_to_strip:]
                child.parent.replace(child, nodes.Text(new_text))
                chars_to_strip = 0
        elif isinstance(child, nodes.Element):
            # Recurse into inline containers (e.g., strong/emphasis)
            chars_to_strip = _strip_left_chars_inplace(child, chars_to_strip)
            # If the element became empty, keep it as-is; docutils handles empties gracefully
        else:
            # Unknown node type; skip
            continue

    return chars_to_strip


def _convert_block_quote(bq: nodes.block_quote) -> Optional[nodes.Node]:
    if not bq.children or not isinstance(bq.children[0], nodes.paragraph):
        return None

    first_par = bq.children[0]
    text = _gather_text(first_par)

    kind = None
    pattern = None
    if NOTE_RE.match(text):
        kind, pattern = "note", NOTE_RE
    elif IMPORTANT_RE.match(text):
        kind, pattern = "important", IMPORTANT_RE
    else:
        return None

    # Determine how many characters to strip from the beginning of the first paragraph
    m = pattern.match(text)
    if not m:
        return None
    prefix_len = m.end()

    # Strip the prefix across inline nodes while preserving formatting
    _strip_left_chars_inplace(first_par, prefix_len)

    # Create the admonition node and move children
    new_node = nodes.note() if kind == "note" else nodes.important()
    for child in list(bq.children):
        child.parent = None
        new_node += child
    return new_node


def on_doctree_read(app: Sphinx, doctree: nodes.document) -> None:
    # Convert qualifying Markdown blockquotes to admonitions
    for bq in list(doctree.traverse(nodes.block_quote)):
        new_node = _convert_block_quote(bq)  # type: ignore[arg-type]
        if new_node is not None:
            bq.replace_self(new_node)


def setup(app: Sphinx):
    app.connect("doctree-read", on_doctree_read)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
