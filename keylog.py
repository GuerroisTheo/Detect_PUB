"""All imports necessary to carry out this process"""
import pynput
from pynput import keyboard
from datetime import datetime   


class KeyLogger:
    def __init__(self):
        """The builder of the 'KeyLogger' class
        Class for creating keyloggers and controlling them.
        'stopMain' is a boolean which detect the 'alt+gr' and stop the main.py process
        """
        self.a_stopMain = True
        self.a_listener = keyboard.Listener(on_press=self.on_press)

    def on_press(self,p_key):
        """Fills the list with a time and the key pressed."""
        if(p_key == keyboard.Key.alt_gr):
            self.a_stopMain = False

    def start(self):
        """Starts the keylogger"""
        self.a_listener.start()

    def stop(self):
        """Stops the keylogger"""
        self.a_listener.stop()