# config/

JSON definitions RPPS loads by name, one folder per resource; the folder a file is in says what it is (so no file
writes a `"base"`), and a parent can hold small children as rows (`"variants"`). The format (names, `extends`, tags,
each resource's keys) is in [docs/guide/definitions.md](../docs/guide/definitions.md), also at
https://anonoei.github.io/RPPS/guide/definitions/. `python -m rpps ls` lists what is here.
