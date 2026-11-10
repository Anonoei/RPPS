<%!
    # pdoc3 template overrides for the RPPS docs. Used with `--template-dir pdoc_templates`;
    # only the values changed from pdoc's defaults are listed.
    show_source_code = True
    sort_identifiers = False       # keep the order the code defines things in
    hljs_style = 'atom-one-dark'   # dark syntax highlighting to match head.mako
    lunr_search = {'fuzziness': 1, 'index_docstrings': True}
    git_link_template = 'https://github.com/Anonoei/RPPS/blob/{commit}/{path}#L{start_line}-L{end_line}'
%>
