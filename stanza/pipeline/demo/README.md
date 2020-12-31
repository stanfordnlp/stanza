## Interactive Demo for Stanza

### Requirements

stanza, flask

### Run the demo locally

1. Make sure you know how to disable your browser's CORS rule. For Chrome, [this extension](https://mybrowseraddon.com/access-control-allow-origin.html) works pretty well.
2. From this directory, start the Stanza demo server

```bash
export FLASK_APP=demo_server.py
flask run
```

3. In `stanza-brat.js`, uncomment the line at the top that declares `serverAddress` and point it to where your flask is serving the demo server (usually `http://localhost:5000`)

4. Open `stanza-brat.html` in your browser (with CORS disabled) and enjoy!

### Common issues

Make sure you have the models corresponding to the language you want to test out locally before submitting requests to the server! (Models can be obtained by `import stanza; stanza.download(<language_code>)`.
