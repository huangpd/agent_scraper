"""共享 fixtures"""

import pytest


SAMPLE_HTML = """\
<html>
<head><title>Test</title></head>
<body>
<main>
  <div class="file-list">
    <div class="row">
      <a class="file-link" href="/repo/blob/main/config.json">config.json</a>
      <span class="size">1.2 KB</span>
    </div>
    <div class="row">
      <a class="file-link" href="/repo/blob/main/model.bin">model.bin</a>
      <span class="size">4.5 GB</span>
    </div>
    <div class="row">
      <a class="file-link" href="/repo/blob/main/README.md">README.md</a>
      <span class="size">3.0 KB</span>
    </div>
  </div>
  <button class="load-more">Load more files</button>
  <a class="next-page" href="/repo?page=2">Next</a>
</main>
</body>
</html>
"""

SAMPLE_HTML_WITH_FOLDERS = """\
<html><body><main>
  <div class="file-list">
    <a class="folder" href="/repo/tree/main/src">src</a>
    <a class="folder" href="/repo/tree/main/tests">tests</a>
    <a class="file-link" href="/repo/blob/main/setup.py">setup.py</a>
  </div>
</main></body></html>
"""
