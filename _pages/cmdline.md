---
title: Using Stanford CoreNLPy from the command line
keywords: command-line
permalink: '/cmdline.html'
---

## Quick start

The minimal command to run Stanford CoreNLPy from the command line is:

```sh
python something
```

If this command is run from the distribution directory, it processes the included [sample file](files/input.txt) `input.txt`. We use a wildcard `"*"` after `-cp` to load all jar files in the current directory &ndash; it needs to be in quotes. This command writes the output to an XML [file](files/input.txt.xml.txt) named `input.txt.xml` in the same directory.
