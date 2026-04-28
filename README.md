# OlliVision

Minimales Python-Projekt mit CLI-Befehl:

```bash
python3 -m ollivision.app describe-scene
```

Aktueller Stand:
- simuliert eine Kameraaufnahme
- holt eine Dummy-Antwort vom Hermes-Client (simuliert)
- gibt die Antwort auf der Konsole aus

## Entwicklung

Empfohlen mit venv:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
pytest -q
```
