"""All imports necessary to carry out this process"""
import pynput
from pynput import keyboard
from datetime import datetime   


class KeyLogger:
    def __init__(self):
        """The builder of the 'KeyLogger' class
        Class for creating keyloggers and controlling them.
        'stopMain' is a boolean which detect the 'ù' and stop the main.py process
        A keylogger stores the received input logs in a 'a_keys' list.
        The 'listener' attribute will allow us to listen to the keyboard.
        State and find_state are used to determine the state of our recording.
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