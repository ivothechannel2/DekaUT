import tkinter as tk
from tkinter import simpledialog, messagebox
import numpy as np
import librosa
import sounddevice as sd
from scipy.io import wavfile
import sys
import os

# --- MOTOR DE AUDIO ---
class DekaUTEngine:
    def __init__(self, voicebank_path, nota_base_grabacion="C4"):
        self.voicebank = voicebank_path
        self.sample_rate = 22050 
        
        try:
            self.nota_base = librosa.note_to_midi(nota_base_grabacion)
            sd.query_devices()
            self.audio_disponible = True
        except Exception as e:
            print(f"Advertencia de audio: {e}")
            self.nota_base = 60
            self.audio_disponible = False

    def procesar_fonema(self, fonema, nota_destino_nombre, duracion_segundos):
        ruta_wav = None
        
        # 1. Buscador Inteligente
        if os.path.exists(self.voicebank):
            archivos = os.listdir(self.voicebank)
            for archivo in archivos:
                nombre_sin_ext = os.path.splitext(archivo)[0]
                if nombre_sin_ext == fonema and archivo.lower().endswith('.wav'):
                    ruta_wav = os.path.join(self.voicebank, archivo)
                    break
        
        if not ruta_wav:
            ruta_wav = f"{self.voicebank}/{fonema}.wav"

        # 2. Cargar el archivo de audio con Plan B Robusto (SciPy)
        audio = None
        try:
            print(f"Intentando cargar fonema '{fonema}' desde la ruta: {ruta_wav}")
            audio, sr = librosa.load(ruta_wav, sr=self.sample_rate)
        except Exception as e:
            print(f" [Aviso] Librosa falló, iniciando PLAN B (SciPy) para '{fonema}'")
            try:
                if not os.path.exists(ruta_wav):
                    print(f" [ERROR] El archivo no existe físicamente en: {ruta_wav}")
                    return None
                    
                sr_scipy, data_scipy = wavfile.read(ruta_wav)
                
                if data_scipy.dtype == np.int16:
                    data_float = data_scipy.astype(np.float32) / 32768.0
                elif data_scipy.dtype == np.int32:
                    data_float = data_scipy.astype(np.float32) / 2147483648.0
                else:
                    data_float = data_scipy.astype(np.float32)
                
                if len(data_float.shape) > 1:
                    data_float = np.mean(data_float, axis=1)
                
                if sr_scipy != self.sample_rate:
                    audio = librosa.resample(data_float, orig_sr=sr_scipy, target_sr=self.sample_rate)
                else:
                    audio = data_float
            except Exception as err_critico:
                print(f" [ERROR CRÍTICO] Ningún método pudo abrir '{fonema}': {err_critico}")
                return None

        if audio is None or len(audio) == 0:
            return None

        # 3. Procesamiento digital de sonido (DSP)
        try:
            nota_destino_midi = librosa.note_to_midi(nota_destino_nombre)
            semitonos_diferencia = nota_destino_midi - self.nota_base
            
            if semitonos_diferencia != 0:
                audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=semitonos_diferencia)

            duracion_original = librosa.get_duration(y=audio, sr=self.sample_rate)
            factor_velocidad = duracion_original / duracion_segundos
            audio_final = librosa.effects.time_stretch(audio, rate=factor_velocidad)
            return audio_final
        except Exception as e:
            print(f"Error procesando DSP para '{fonema}': {e}")
            return None

# --- INTERFAZ GRÁFICA INTERACTIVA ---
class DekaUTGui:
    def __init__(self, root, motor_audio):
        self.root = root
        self.motor = motor_audio
        self.root.title("DekaUT - Vocal Synth Software (Hiragana Edition)")
        self.root.geometry("900x500")
        self.root.configure(bg="#f4a261")

        self.nota_actual_id = None
        self.diccionario_notas = {}  
        
        self.alto_tecla = 20
        self.ancho_piano = 100
        self.lista_notas_musicales = ["B4", "A#4", "A4", "G#4", "G4", "F#4", "F4", "E4", "D#4", "D4", "C#4", "C4"]
        self.total_notas = len(self.lista_notas_musicales)

        self.crear_interfaz()
        
        if not self.motor.audio_disponible:
            messagebox.showwarning("Modo Silencioso", "DekaUT no detectó altavoces activos. El programa se abrirá pero no sonará.")

    def crear_interfaz(self):
        self.panel_superior = tk.Frame(self.root, bg="#f4a261")
        self.panel_superior.pack(fill=tk.X, padx=10, pady=5)
        
        btn_play = tk.Button(self.panel_superior, text="▶ REPRODUCIR SECUENCIA", command=self.reproducir_secuencia, bg="#2a9d8f", fg="white", font=("Arial", 10, "bold"))
        btn_play.pack(side=tk.LEFT, pady=2)

        self.main_frame = tk.Frame(self.root, bg="#f4a261")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas_piano = tk.Canvas(self.main_frame, width=self.ancho_piano, height=self.total_notas * self.alto_tecla, bg="white", highlightthickness=0)
        self.canvas_piano.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas_roll = tk.Canvas(self.main_frame, bg="#ffb703", highlightthickness=1, highlightbackground="black")
        self.canvas_roll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.dibujar_piano()
        self.dibujar_cuadrícula()

        self.canvas_roll.bind("<ButtonPress-1>", self.iniciar_nota)
        self.canvas_roll.bind("<B1-Motion>", self.arrastrar_nota)
        self.canvas_roll.bind("<ButtonRelease-1>", self.finalizar_nota)
        self.canvas_roll.bind("<Double-Button-1>", self.asignar_letra_nota)

        self.barra_inferior = tk.Frame(self.root, bg="#f4a261")
        self.barra_inferior.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)

        tk.Label(self.barra_inferior, text="Tempo", bg="#f4a261").grid(row=0, column=0, padx=5, sticky="w")
        self.entry_tempo = tk.Entry(self.barra_inferior, width=6, justify="center")
        self.entry_tempo.insert(0, "120")
        self.entry_tempo.grid(row=1, column=0, padx=5)

        tk.Label(self.barra_inferior, text="Tone Base", bg="#f4a261").grid(row=0, column=1, padx=5, sticky="w")
        self.entry_tone = tk.Entry(self.barra_inferior, width=6, justify="center")
        self.entry_tone.insert(0, "C4")
        self.entry_tone.grid(row=1, column=1, padx=5)

    def dibujar_piano(self):
        patron_negras = [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        for i, nombre_nota in enumerate(self.lista_notas_musicales):
            y1 = i * self.alto_tecla
            y2 = y1 + self.alto_tecla
            es_negra = patron_negras[i % 12]
            
            fill_color = "black" if es_negra else "white"
            text_color = "white" if es_negra else "black"
            
            self.canvas_piano.create_rectangle(0, y1, self.ancho_piano, y2, fill=fill_color, outline="gray")
            self.canvas_piano.create_text(self.ancho_piano - 25, y1 + 10, text=nombre_nota, fill=text_color, font=("Arial", 8))

    def dibujar_cuadrícula(self):
        for i in range(self.total_notas + 1):
            y = i * self.alto_tecla
            self.canvas_roll.create_line(0, y, 3000, y, fill="#e59866", dash=(4, 4))

    def iniciar_nota(self, event):
        fila = event.y // self.alto_tecla
        if fila >= self.total_notas: return
        
        self.start_y = fila * self.alto_tecla
        self.start_x = event.x
        self.nota_musical_detectada = self.lista_notas_musicales[fila]

        self.nota_actual_id = self.canvas_roll.create_rectangle(
            self.start_x, self.start_y, self.start_x + 20, self.start_y + self.alto_tecla,
            fill="#457b9d", outline="black"
        )

    def arrastrar_nota(self, event):
        if self.nota_actual_id:
            actual_x = max(event.x, self.start_x + 10)
            self.canvas_roll.coords(self.nota_actual_id, self.start_x, self.start_y, actual_x, self.start_y + self.alto_tecla)

    def finalizar_nota(self, event):
        if self.nota_actual_id:
            coords = self.canvas_roll.coords(self.nota_actual_id)
            texto_id = self.canvas_roll.create_text(
                coords[0] + 5, coords[1] + 10, 
                text="あ", fill="white", 
                font=("Arial", 10, "bold"), anchor="w"
            )
            self.diccionario_notas[self.nota_actual_id] = {
                "x_inicio": coords[0],
                "x_fin": coords[2],
                "nota_musical": self.nota_musical_detectada,
                "letra": "あ", 
                "texto_id": texto_id
            }
            self.nota_actual_id = None

    def asignar_letra_nota(self, event):
        elemento_tocado = self.canvas_roll.find_withtag(tk.CURRENT)
        if not elemento_tocado: return
        id_objeto = elemento_tocado[0]

        rect_id = None
        if id_objeto in self.diccionario_notas:
            rect_id = id_objeto
        else:
            for k, v in self.diccionario_notas.items():
                if v["texto_id"] == id_objeto:
                    rect_id = k
                    break

        if rect_id:
            letra_ingresada = simpledialog.askstring("DekaUT Lyrics", "Ingresa la sílaba en Hiragana (ej: か, し, あ):", parent=self.root)
            if letra_ingresada:
                self.diccionario_notas[rect_id]["letra"] = letra_ingresada
                self.canvas_roll.itemconfig(self.diccionario_notas[rect_id]["texto_id"], text=letra_ingresada)

    def reproducir_secuencia(self):
        if not self.diccionario_notas:
            messagebox.showinfo("DekaUT", "Primero dibuja algunas notas en la pantalla naranja.")
            return

        notas_ordenadas = sorted(self.diccionario_notas.values(), key=lambda x: x["x_inicio"])
        
        try:
            bpm = float(self.entry_tempo.get())
        except ValueError:
            bpm = 120
        
    def reproducir_secuencia(self):
        if not self.diccionario_notas:
            messagebox.showinfo("DekaUT", "Primero dibuja algunas notas en la pantalla naranja.")
            return

        notas_ordenadas = sorted(self.diccionario_notas.values(), key=lambda x: x["x_inicio"])
        
        try:
            bpm = float(self.entry_tempo.get())
        except ValueError:
            bpm = 120
        
        factor_escala_tiempo = 60.0 / bpm / 50.0 
        audio_secuencia_completa = []

        for nota in notas_ordenadas:
            ancho_pixels = nota["x_fin"] - nota["x_inicio"]
            duracion_segundos = max(ancho_pixels * factor_escala_tiempo, 0.1)
            
            audio_nota = self.motor.procesar_fonema(
                fonema=nota["letra"], 
                nota_destino_nombre=nota["nota_musical"], 
                duracion_segundos=duracion_segundos
            )
            
            if audio_nota is not None:
                audio_secuencia_completa.append(audio_nota)
                silencio_transicion = np.zeros(int(22050 * 0.02))
                audio_secuencia_completa.append(silencio_transicion)

        if audio_secuencia_completa:
            try:
                audio_final = np.concatenate(audio_secuencia_completa)
                print("¡Renderizado completado exitosamente!")
                sd.play(audio_final, self.motor.sample_rate)
            except Exception as e:
                messagebox.showerror("Error de salida", f"No se pudo reproducir el sonido:\n{e}")
        else:
            messagebox.showwarning("DekaUT Error", "No se pudo mapear ningún archivo. Revisa la consola.")

# --- INICIO DE LA APLICACIÓN ---
if __name__ == "__main__":
    try:
        # 1. Obtener la ruta absoluta exacta del directorio donde está guardado este script .py
        directorio_script = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Forzar a Python a trabajar en esa carpeta real (así ignora System32)
        os.chdir(directorio_script)
        
        # 3. Buscar las carpetas de manera relativa a tu script en Descargas
        ruta_posible_1 = "mi_voz"
        ruta_posible_2 = "DekaUT/mi_voz"
        
        ruta_final = ruta_posible_2 if os.path.exists(ruta_posible_2) else ruta_posible_1
        
        print(f"Dirección real fijada en: {directorio_script}")
        print(f"Ruta activa del Voicebank: {os.path.abspath(ruta_final)}")
        
        motor_dekaut = DekaUTEngine(voicebank_path=ruta_final, nota_base_grabacion="C4")
        root = tk.Tk()
        app = DekaUTGui(root, motor_audio=motor_dekaut)
        root.mainloop()
    except Exception as e:
        root_err = tk.Tk()
        root_err.withdraw()
        messagebox.showerror("Error", f"Error general de inicio:\n{e}")
        sys.exit(1)
