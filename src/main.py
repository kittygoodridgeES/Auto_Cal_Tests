from gui import WizardApp
from gui import setup_logging

def run_gui(logger):
    app = WizardApp(logger)
    app.mainloop()

if __name__ == "__main__":
    run_gui(setup_logging())