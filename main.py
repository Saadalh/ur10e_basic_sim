import sys


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.argv.remove("--test")
        from src.test_pi05 import run

        run()
    else:
        from src.sim import launch_simulation

        launch_simulation()
