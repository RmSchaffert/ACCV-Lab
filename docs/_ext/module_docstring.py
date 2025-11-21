from typing import List
import ast

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective
from sphinx.util import logging

LOGGER = logging.getLogger(__name__)


class ModuleDocstringInclude(SphinxDirective):
    required_arguments = 1  # path to file
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}

    def run(self):
        env = self.state.document.settings.env
        rel_filename, filename = env.relfn2path(self.arguments[0])
        env.note_dependency(rel_filename)

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as exc:
            LOGGER.warning('module-docstring: failed to read %s: %s', filename, exc)
            return []

        try:
            module = ast.parse(source)
            doc = ast.get_docstring(module) or ''
        except Exception as exc:
            LOGGER.warning('module-docstring: failed to parse %s: %s', filename, exc)
            doc = ''

        if not doc:
            return []

        container = nodes.container()
        lines = StringList(doc.splitlines(), source=filename)
        self.state.nested_parse(lines, 0, container)
        return [container]


def setup(app):  # pragma: no cover - build-time only
    app.add_directive('module-docstring', ModuleDocstringInclude)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
