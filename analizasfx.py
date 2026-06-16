import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plot
import numpy as np

# Librerías necesarias para recorrer carpetas y crear el CSV
import os
import csv

# ==============================================================================
# 1. FUNCIÓN DE CONFIGURACIÓN DE ARGUMENTOS 
# ==============================================================================
def parse_arguments():
      
    """
    Esta función configura y lee los argumentos que el usuario introduce
    por consola al ejecutar el programa.

    Argumentos disponibles:
    -i / --input      : carpeta principal del dataset.
    -d / --display    : tipo de visualización: wave, spec o ambas.
    -f / --features   : tipo de características a extraer: temp, frec o all.
    -o / --output     : fichero CSV de salida.
    -p / --precision  : número de decimales para guardar las características.
    """
      
    # Creamos el parser, que será el encargado de interpretar los argumentos  
    parser = argparse.ArgumentParser(
        description="Herramienta de extracción de características y visualización de audio."
    )

    # Ruta de la carpeta principal del dataset.
    parser.add_argument("-i", "--input", required=True, 
                        help="Ruta de la carpeta principal del dataset")
    
    # Argumento opcional para elegir qué representación gráfica mostrar
    # IMPORTANTE: Para generar el CSV de WEKA no se recomienda usar -d, porque abriría gráficas
    parser.add_argument("-d", "--display", nargs="+", choices=["wave", "spec"], 
                        help="Opciones de visualización: 'wave' o 'spec'.")

    # Argumento opcional para elegir qué características extraer
    # Si no se indica, por defecto se extraen todas "all"
    parser.add_argument("-f", "--features", choices=["temp", "frec", "all"], 
                        default="all", 
                        help="Tipo de características a extraer por consola. Por defecto es 'all'.")

    # Fichero CSV donde se guardarán todas las características extraídas
    parser.add_argument("-o", "--output", default="dataset_weka.csv",
                        help="Nombre del fichero CSV de salida. Por defecto: dataset_weka.csv")

    # Número de decimales con los que se guardarán las características
    # Por defecto se guardan con 3 decimales
    parser.add_argument("-p", "--precision", type=int, default=3,
                        help="Número de decimales para las características. Por defecto: 3")

    # Parseamos los argumentos y los devolvemos
    args = parser.parse_args()
    return args


# ==============================================================================
# 2. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ==============================================================================
def main():

    """
    Función principal del programa.

    Pasos:
    1. Leer argumentos de consola.
    2. Recorrer la carpeta principal del dataset.
    3. Entrar en cada subcarpeta, que representa una clase.
    4. Cargar cada archivo de audio.
    5. Extraer características temporales y/o frecuenciales.
    6. Ordenar las características según el formato pedido.
    7. Añadir la etiqueta de clase al principio de cada fila.
    8. Guardar todo en un fichero CSV compatible con WEKA.
    """

    # Llamamos a la función para obtener los argumentos introducidos por el usuario
    args = parse_arguments()

    # Lista donde guardaremos todas las filas del CSV
    # Cada fila será: clase + características del audio
    filas_csv = []

    # Creamos la cabecera del CSV dependiendo de las características seleccionadas
    # Orden del CSV: clase, centroid, rms, zcr, mfccs, chromas
    cabecera = []

    # La primera columna será la clase
    cabecera.append("clase")

    # Si se extraen características frecuenciales, añadimos primero el centroide
    if args.features in ["frec", "all"]:
        cabecera.append("centroid")

    # Si se extraen características temporales, añadimos rms y zcr en ese orden
    if args.features in ["temp", "all"]:
        cabecera.extend(["rms", "zcr"])

    # Si se extraen características frecuenciales, añadimos MFCCs y Chromas
    if args.features in ["frec", "all"]:
        cabecera.extend([
            "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5",
            "mfcc6", "mfcc7", "mfcc8", "mfcc9", "mfcc10",
            "mfcc11", "mfcc12", "mfcc13"
        ])

        cabecera.extend([
            "chroma0", "chroma1", "chroma2", "chroma3",
            "chroma4", "chroma5", "chroma6", "chroma7",
            "chroma8", "chroma9", "chroma10", "chroma11"
        ])

    # Comprobamos que la ruta introducida sea una carpeta válida
    if not os.path.isdir(args.input):
        print(f"Error: La carpeta '{args.input}' no existe o no es válida.")
        return

    # Recorremos las subcarpetas del dataset.
    # Cada subcarpeta será una clase: Ambiente, Comentarista, Pasos, etc.
    for nombre_clase in os.listdir(args.input):

        ruta_clase = os.path.join(args.input, nombre_clase)

        # Si no es una carpeta, la ignoramos
        if not os.path.isdir(ruta_clase):
            continue

        print(f"\nProcesando clase: {nombre_clase}")

        # Recorremos todos los archivos dentro de la carpeta de esa clase
        for nombre_archivo in os.listdir(ruta_clase):

            # Procesamos solo archivos .wav
            if not nombre_archivo.lower().endswith(".wav"):
                continue

            ruta_audio = os.path.join(ruta_clase, nombre_archivo)

            # --- CARGA DEL ARCHIVO DE AUDIO ---
            try:
                # y contiene la señal de audio como array de muestras
                # sr contiene la frecuencia de muestreo
                y, sr = librosa.load(ruta_audio)

            except Exception as e:
                print(f"Error: No se pudo cargar el archivo '{ruta_audio}'.")
                continue 

            # --- MÓDULO DE VISUALIZACIÓN (--display) ---
            if args.display:
                if "wave" in args.display:
                    plot.figure(figsize=(10, 4))
                    librosa.display.waveshow(y, sr=sr, alpha=0.5, color='blue')
                    plot.title("Forma de Onda (Waveform)")
                    plot.xlabel("Tiempo (s)")
                    plot.ylabel("Amplitud")
                    plot.tight_layout()
                    plot.show() 
                    
                if "spec" in args.display:
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                    plot.figure(figsize=(10, 4))

                    # Representamos la amplitud de la señal en función del tiempo
                    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
                    plot.colorbar(format='%+2.0f dB')
                    plot.title("Espectrograma de Frecuencias")
                    plot.tight_layout()
                    plot.show()

            # --- MÓDULO DE EXTRACCIÓN DE CARACTERÍSTICAS (--features) ---

            # Lista donde se almacenarán las características finales 
            vector_caracteristicas = []

            # Variables auxiliares para guardar cada característica antes de ordenarlas
            centroid_media = None
            rms_media = None
            zcr_media = None
            mfccs_medios = []
            chroma_medios = []

            # Temporales
            if args.features in ["temp", "all"]:

                # Zero Crossing Rate: mide cuántas veces la señal cruza por cero
                zcr = librosa.feature.zero_crossing_rate(y)
                zcr_media = np.mean(zcr, axis=1)[0]
                
                # RMS: mide la energía media de la señal
                rms = librosa.feature.rms(y=y)
                rms_media = np.mean(rms, axis=1)[0]

            # Frecuenciales
            if args.features in ["frec", "all"]:

                # Centroide espectral: indica el centro de gravedad del espectro
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                centroid_media = np.mean(centroid, axis=1)[0]

                # MFCCs: coeficientes muy usados para representar el timbre del sonido
                # Se extraen 13 coeficientes
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfccs_medios = list(np.mean(mfccs, axis=1))

                # Chroma: representa la energía asociada a las 12 clases de tono
                # Sus componentes se nombran de chroma0 a chroma11
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_medios = list(np.mean(chroma, axis=1))

            # --------------------------------------------------------------------------
            # ORDEN FINAL CARACTERÍSTICAS:
            # centroid, rms, zcr, mfccs, chromas
            # La clase se añadirá después al principio de la fila.
            # --------------------------------------------------------------------------

            # Añadimos centroid si se han pedido características frecuenciales
            if args.features in ["frec", "all"]:
                vector_caracteristicas.append(centroid_media)

            # Añadimos rms y zcr si se han pedido características temporales
            if args.features in ["temp", "all"]:
                vector_caracteristicas.append(rms_media)
                vector_caracteristicas.append(zcr_media)

            # Añadimos mfccs y chromas si se han pedido características frecuenciales
            if args.features in ["frec", "all"]:
                vector_caracteristicas.extend(mfccs_medios)
                vector_caracteristicas.extend(chroma_medios)

            # ==============================================================================
            # 3. SALIDA PARA CSV
            # ==============================================================================

            if vector_caracteristicas:

                # Redondeamos las características al número de decimales indicado
                # Por defecto: 3 decimales
                vector_caracteristicas = [
                    round(float(valor), args.precision) for valor in vector_caracteristicas
                ]

                # Añadimos la clase al principio de la fila
                # La clase es el nombre de la carpeta donde está el audio
                fila = [nombre_clase] + vector_caracteristicas

                # Guardamos la fila en la lista general
                filas_csv.append(fila)

                print(f"  Audio procesado: {nombre_archivo}")

    # Si no se ha procesado ningún audio, avisamos
    if not filas_csv:
        print("\nNo se ha procesado ningún audio .wav.")
        return

    # Guardamos todas las filas en un fichero CSV
    with open(args.output, "w", newline="", encoding="utf-8") as archivo_csv:
        writer = csv.writer(archivo_csv)

        # Escribimos primero la cabecera
        writer.writerow(cabecera)

        # Escribimos después todas las filas de características
        writer.writerows(filas_csv)

    print("\nProceso finalizado correctamente.")
    print(f"CSV generado: {args.output}")
    print(f"Número total de audios procesados: {len(filas_csv)}")
    print(f"Decimales utilizados: {args.precision}")


if __name__ == "__main__":
    main()