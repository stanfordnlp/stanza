---
title: Frequently Asked Questions (FAQ)
keywords: stanfordnlp, frequently asked questions, faq
permalink: '/faq.html'
---

## Why am I getting an `OSError: [Errno 22] Invalid argument` error and therefore a `Vector file is not provided` exception while the model is being loaded?

If you are getting this error, it is very likely that you are running macOS and using Python with version <= 3.6.7 or <= 3.7.1. If this is the case, then you are affected by a [known Python bug](https://bugs.python.org/issue24658) on macOS, and upgrading your Python to >= 3.6.8 or >= 3.7.2 should solve this issue.

If you are not running macOS or already have the specified Python version and still seeing this issue, please report this to us via the [GitHub issue tracker](https://github.com/stanfordnlp/stanfordnlp/issues).
