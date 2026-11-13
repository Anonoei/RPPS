<%doc>Dark theme. pdoc3 ships a light-only stylesheet; this is included after it and overrides its colors.</%doc>
<meta name="color-scheme" content="dark">
<style>
  :root {
    color-scheme: dark;
    --bg: #14161a;
    --bg-alt: #1c1f25;
    --bg-hover: #262a32;
    --border: #2f343d;
    --text: #d6dae1;
    --muted: #8b93a1;
    --link: #6cb6ff;
    --link-visited: #b392f0;
    --link-hover: #ffa657;
    --ident: #ff7b72;
    --highlight-color: #3a3520;
  }
  html, body { background: var(--bg); color: var(--text); }
  h1, h2, h3, h4, h5, h6 { color: #eef0f4; }
  a { color: var(--link); }
  a:visited { color: var(--link-visited); }
  a:hover { color: var(--link-hover); }
  .ident { color: var(--ident); }
  code { background: var(--bg-alt); color: var(--text); }
  h1 code { background: transparent; }
  pre { border-color: var(--border); }
  pre code.hljs, .hljs { background: var(--bg-alt); }
  hr { border-color: var(--border); }
  #sidebar { background: var(--bg-alt); border-color: var(--border); }
  #sidebar code { background: transparent; }
  #footer { border-color: var(--border); color: var(--muted); }
  #lunr-search, #lunr-search + * input {
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
  }
  .name { background: var(--bg-alt); }
  .name:hover { background: var(--bg-hover); }
  dt:target .name, h1:target, h2:target, h3:target, h4:target, h5:target, h6:target {
    background: var(--highlight-color);
  }
  .inherited { color: var(--muted); border-color: var(--border); }
  .source > summary, .git-link-div { color: var(--muted); }
  table, th, td { border-color: var(--border); }
  th { background: var(--bg-alt); }
  tbody tr:nth-child(even) { background: var(--bg-alt); }
  blockquote { border-color: var(--border); color: var(--muted); }
  .admonition { color: var(--text); }
  .admonition.note, .admonition.info, .admonition.important { background: #12303f; }
  .admonition.todo, .admonition.versionadded, .admonition.tip, .admonition.hint { background: #16301f; }
  .admonition.warning, .admonition.versionchanged, .admonition.deprecated { background: #3d3112; }
  .admonition.error, .admonition.danger, .admonition.caution { background: #3d1a1d; }
  #lunr-search + ul, .lunr-results { background: var(--bg-alt); }
</style>
