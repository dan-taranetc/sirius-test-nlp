from speech_recognition import Recognizer, AudioFile
from subprocess import Popen
import os


class SpeachDecoder:
    def __init__(self):
        self.recognizer = Recognizer()

    def voice_to_text(self, filename):
        args = ['ffmpeg', '-i', f'{filename}.ogg', f'{filename}.wav']
        process = Popen(args)
        process.wait()

        with AudioFile(f'{filename}.wav') as source:
            audio = self.recognizer.record(source)

        text = self.recognizer.recognize_google(audio, language='RU')

        os.remove(f'{filename}.ogg')
        os.remove(f'{filename}.wav')

        return text


decoder = SpeachDecoder()

