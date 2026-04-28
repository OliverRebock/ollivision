from ollivision.app import describe_scene, main


def test_describe_scene_returns_dummy_response():
    result = describe_scene()
    assert "Dummy Hermes Antwort" in result
    assert "simulierte Kameraaufnahme" in result


def test_cli_describe_scene_outputs_response(capsys):
    main(["describe-scene"])
    out = capsys.readouterr().out.strip()
    assert "Dummy Hermes Antwort" in out
