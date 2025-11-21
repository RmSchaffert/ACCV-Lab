from typing import List

from docutils import nodes
from sphinx.directives.code import LiteralInclude


class NoteLiteralInclude(LiteralInclude):
    option_spec = LiteralInclude.option_spec.copy()
    option_spec.update(
        {
            'note-tag': lambda s: s,  # e.g., '#@NOTE'
            'note-cont': lambda c: c,
            'highlight-notes': (
                lambda h: h.lower() in ["true", "yes", "y"] or (h.isnumeric() and float(h) != 0.0)
            ),  # flag to enable/disable
        }
    )

    def run(self):
        nodes_list = super().run()  # Let base class handle everything

        note_tag = self.options.get('note-tag', '#@NOTE;# @NOTE')
        note_cont = self.options.get('note-cont', '#')
        note_tags = note_tag.split(';')
        highlight_notes = self.options.get('highlight-notes', True)

        if not highlight_notes:
            return nodes_list

        for node in nodes_list:
            # literalinclude with a caption may wrap the code block inside a container.
            # Traverse to find any nested literal_block nodes as well.
            if isinstance(node, nodes.literal_block):
                literal_blocks = [node]
            else:
                try:
                    literal_blocks = list(node.traverse(nodes.literal_block))
                except Exception:
                    literal_blocks = []

            for lb in literal_blocks:
                text = lb.astext()
                lines = text.splitlines()
                hl_lines: List[int] = []
                in_block = False

                for i, line in enumerate(lines, start=1):
                    stripped = line.lstrip()
                    starts_with_tag = False
                    for nt in note_tags:
                        starts_with_tag = starts_with_tag or stripped.startswith(nt)
                    if starts_with_tag:
                        in_block = True
                        hl_lines.append(i)
                        continue
                    if in_block:
                        if stripped.startswith(note_cont):
                            hl_lines.append(i)
                        else:
                            in_block = False

                if hl_lines:
                    # Preserve existing highlight args (e.g., 'linenostart' from :lineno-match:)
                    existing = dict(lb.get('highlight_args', {}))
                    existing_hl = list(existing.get('hl_lines', []))
                    existing['hl_lines'] = sorted(set(existing_hl + hl_lines))
                    lb['highlight_args'] = existing

        return nodes_list


def setup(app):  # pragma: no cover - build-time only
    app.add_directive('note-literalinclude', NoteLiteralInclude)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
