
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util.docutils import SphinxDirective


class CollapseCodeDirective(SphinxDirective):
    has_content = True
    required_arguments = 1
    option_spec = {}

    def run(self):
        summary = self.arguments[0]
        content = "\n".join(self.content)

        html = f"""
        <details>
          <summary>{summary}</summary>
          <pre><code>{content}</code></pre>
        </details>
        """

        return [nodes.raw("", html, format="html")]


def setup(app):
    app.add_directive("collapse-code", CollapseCodeDirective)
    return {"version": "0.1", "parallel_read_safe": True}
