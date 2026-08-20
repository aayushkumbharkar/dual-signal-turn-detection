from demo import build_demo

def test_demo_initialization():
    app = build_demo()
    assert app is not None
