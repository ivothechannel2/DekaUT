import numpy as np
import librosa
import sounddevice as sd
from scipy.io import wavfile

class DekaUTEngine:
    def __init__(self, voicebank_path, nota_base_grabacion="C4"):
        """
        voicebank_path: Carpeta donde guardas tus archivos .wav (ej: 'a.wav', 'ka.wav')
        nota_base_grabacion: La nota musical exacta en la que el banco de voz fue grabado (por defecto Do4)
        """
        self.voicebank = voicebank_path
        self.nota_base = librosa.note_to_midi(nota_base_grabacion)
        self.sample_rate = 22050 # Librosa trabaja excelentemente bien a esta frecuencia por defecto

    def procesar_fonema(self, fonema, nota_destino_nombre, duracion_segundos):
        """
        Modifica el tono y la duración de un archivo de sonido sin alterar su timbre drásticamente.
        """
        try:
            # 1. Cargar el archivo .wav correspondiente usando librosa
            ruta_wav = f"{self.voicebank}/{fonema}.wav"
            audio, sr = librosa.load(ruta_wav, sr=self.sample_rate)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {fonema}.wav en el banco de voces.")
            return None

        # 2. Calcular la diferencia de semitonos (Pitch Shift)
        nota_destino_midi = librosa.note_to_midi(nota_destino_nombre)
        semitonos_diferencia = nota_destino_midi - self.nota_base

        # Aplicar el cambio de tono (Frecuencia) sin alterar la velocidad
        audio_modificado = librosa.effects.pitch_shift(
            audio, 
            sr=self.sample_rate, 
            n_steps=semitonos_diferencia
        )

        # 3. Calcular el estiramiento del tiempo (Time Stretch)
        duracion_original = librosa.get_duration(y=audio_modificado, sr=self.sample_rate)
        
        # Factor de velocidad: < 1.0 frena el audio (lo estira), > 1.0 lo acelera
        factor_velocidad = duracion_original / duracion_segundos

        # Aplicar el cambio de tiempo sin alterar el tono
        audio_final = librosa.effects.time_stretch(
            audio_modificado, 
            rate=factor_velocidad
        )

        return audio_final

    def reproducir(self, audio_data):
        """ Envía el arreglo de numpy directamente a los altavoces """
        if audio_data is not None:
            sd.play(audio_data, self.sample_rate)
            sd.wait() # Espera a que termine la reproducción
