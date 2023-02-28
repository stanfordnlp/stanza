---
layout: default
title: Stanford CoreNLP Client
keywords: Stanza, Stanford CoreNLP, Client, Server, Python
permalink: '/corenlp_client.html'
nav_order: 7
has_children: true
---

# Stanford CoreNLP Client

Stanza allows users to access our Java toolkit, Stanford CoreNLP, via its server interface, by writing native Python code. Stanza does this by first launching a Stanford CoreNLP server in a background process, and then sending annotation requests to this server process. The response from the CoreNLP server will then be parsed and rendered into a Document protobuf object. As a result of this server-client communication, users can obtain annotations by writing native Python program at the client side, and do not need to worry about anything on the Java server side.
{: .fs-5 .fw-300 }

You can find out more information about the full functionality of Stanford CoreNLP on [the CoreNLP website](https://stanfordnlp.github.io/CoreNLP/).
{: .fs-5 .fw-300 }
