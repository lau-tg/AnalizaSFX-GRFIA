# README - AnalizaSFX

## Descripción del proyecto

Este proyecto consiste en una herramienta en Python para procesar un conjunto de archivos de audio `.wav`, extraer características acústicas y generar un fichero CSV compatible con WEKA.

El objetivo principal es transformar audios organizados por categorías en una tabla de datos donde cada fila representa un audio y cada columna representa una característica extraída. La primera columna del CSV corresponde a la clase del audio, obtenida automáticamente a partir del nombre de la carpeta donde se encuentra.

El CSV generado puede cargarse posteriormente en WEKA para entrenar y evaluar clasificadores supervisados, como NaiveBayes, RandomForest, J48 o IBk.

---------------------------------------------------------------------------------

## Estructura esperada del dataset

El programa espera recibir como entrada una carpeta principal que contenga una subcarpeta por cada clase o categoría.

Ejemplo:

```text
dataset_audio/
├── Ambiente/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── Comentarista/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── Pasos/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└──  ...
     
```

Cada subcarpeta representa una clase. Por ejemplo, todos los audios dentro de `Ambiente/` se etiquetarán automáticamente como `Ambiente` en el CSV.

---------------------------------------------------------------------------------

## Librerías necesarias

El programa utiliza las siguientes librerías:

```python
argparse
librosa
librosa.display
matplotlib
numpy
os
csv
```

Para instalar las librerías externas principales, se puede usar:

```bash
pip install librosa matplotlib numpy
```

En algunos sistemas también puede ser necesario usar:

```bash
py -m pip install librosa matplotlib numpy
```

---------------------------------------------------------------------------------

## Ejecución del programa

El programa se ejecuta desde consola indicando la carpeta del dataset y el nombre del CSV de salida.

Ejemplo básico:

```bash
py analizasfx.py -i dataset_audio -o dataset_weka.csv -f all
```

Este comando procesa todos los archivos `.wav` dentro de `dataset_audio`, extrae todas las características y genera el fichero `dataset_weka.csv`.

---------------------------------------------------------------------------------

## Argumentos disponibles

### `-i` / `--input`

Ruta de la carpeta principal del dataset.

Ejemplo:

```bash
-i dataset_audio
```

Este argumento es obligatorio.

### `-o` / `--output`

Nombre del fichero CSV de salida.

Ejemplo:

```bash
-o dataset_weka.csv
```

Si no se indica, por defecto se genera un archivo llamado `dataset_weka.csv`.

### `-f` / `--features`

Permite seleccionar qué tipo de características extraer.

Opciones disponibles:

```text
temp  → características temporales
frec  → características frecuenciales
all   → todas las características
```

Por defecto se usa `all`.

Ejemplos:

```bash
py Aprendiendo6.py -i dataset_audio -o dataset_weka.csv -f temp
py Aprendiendo6.py -i dataset_audio -o dataset_weka.csv -f frec
py Aprendiendo6.py -i dataset_audio -o dataset_weka.csv -f all
```

### `-d` / `--display`

Permite visualizar gráficas de los audios.

Opciones disponibles:

```text
wave → forma de onda
spec → espectrograma
```

Ejemplo:

```bash
py Aprendiendo6.py -i dataset_audio -d wave
```

Importante: no se recomienda usar `-d` al generar el CSV completo, porque abriría gráficas para todos los audios del dataset.

### `-p` / `--precision`

Permite elegir el número de decimales con los que se guardan las características en el CSV.

Por defecto se usan 3 decimales.

Ejemplo con 3 decimales:

```bash
py Aprendiendo6.py -i dataset_audio -o dataset_weka.csv -f all
```

Ejemplo con 5 decimales:

```bash
py Aprendiendo6.py -i dataset_audio -o dataset_weka.csv -f all -p 5
```

---------------------------------------------------------------------------------

## Características extraídas

El programa puede extraer características temporales y frecuenciales.

### Características temporales

#### RMS

Mide la energía media de la señal de audio. Está relacionada con la intensidad o volumen del sonido.

#### ZCR - Zero Crossing Rate

Mide cuántas veces la señal cruza por cero. Puede ser útil para identificar sonidos ruidosos, percusivos o con cambios rápidos.


### Características frecuenciales

#### Centroide espectral

Indica el centro de gravedad del espectro de frecuencias. Valores más altos suelen estar asociados a sonidos más brillantes o agudos.

#### MFCCs

Se extraen 13 coeficientes MFCC. Son características muy usadas en análisis de audio porque representan información relacionada con el timbre.

#### Chroma

Se extraen 12 componentes chroma, nombradas de `chroma0` a `chroma11`. Estas representan la energía asociada a las 12 clases de tono.

---------------------------------------------------------------------------------

## Orden de las columnas del CSV

El CSV se genera siguiendo el siguiente orden:

```text
clase, centroid, rms, zcr, mfcc1, ..., mfcc13, chroma0, ..., chroma11
```

Con `-f all`, el CSV tendrá:

```text
1 columna de clase
1 centroide espectral
2 características temporales
13 MFCCs
12 Chromas
```

En total:

```text
29 columnas
```

La clase aparece en la primera columna y se obtiene automáticamente del nombre de la subcarpeta donde está cada audio.

---------------------------------------------------------------------------------

## Ejemplo de salida del CSV

Ejemplo simplificado:

```csv
clase,centroid,rms,zcr,mfcc1,mfcc2,...,mfcc13,chroma0,chroma1,...,chroma11
Ambiente,1635.006,0.037,0.052,-256.206,45.832,...,0.641,0.120,...,0.310
Comentarista,2244.680,0.263,0.093,-108.360,62.115,...,0.532,0.098,...,0.287
Pasos,1890.452,0.148,0.071,-180.532,40.203,...,0.488,0.220,...,0.190
```

Los valores numéricos se redondean al número de decimales indicado con `-p`. Si no se indica, se redondean a 3 decimales.

---------------------------------------------------------------------------------

## Uso del CSV en WEKA

Una vez generado el CSV, se puede abrir en WEKA siguiendo estos pasos:

1. Abrir WEKA.
2. Entrar en `Explorer`.
3. Ir a la pestaña `Preprocess`.
4. Pulsar `Open file`.
5. Seleccionar el archivo `dataset_weka.csv`.
6. Comprobar que se han cargado correctamente las columnas.
7. Seleccionar `clase` como atributo de clase.

Importante: en este proyecto, la columna `clase` está colocada en la primera posición. Por tanto, en WEKA puede ser necesario seleccionarla manualmente como atributo objetivo.

Después se puede ir a la pestaña `Classify` y probar clasificadores como:

```text
NaiveBayes
RandomForest
J48
IBk
```

La evaluación puede hacerse con `10-fold cross-validation`.

---------------------------------------------------------------------------------

## Funcionamiento general del código

El flujo del programa es el siguiente:

```text
1. Se leen los argumentos introducidos por consola.
2. Se comprueba que la carpeta del dataset existe.
3. Se recorren las subcarpetas del dataset.
4. Cada subcarpeta se considera una clase.
5. Se recorren los archivos .wav de cada clase.
6. Cada audio se carga con librosa.
7. Se extraen las características seleccionadas.
8. Las características se ordenan según el formato solicitado.
9. Se redondean los valores numéricos.
10. Se añade la clase al principio de la fila.
11. Se guardan todas las filas en un fichero CSV.
```

---------------------------------------------------------------------------------

## Ejemplo completo de ejecución

Para generar el CSV final con todas las características y 3 decimales:

```bash
py Aprendiendo6.py -i dataset_audio -o dataset_weka.csv -f all
```

Para generar el CSV con 4 decimales:

```bash
py Aprendiendo6.py -i dataset_audio -o dataset_weka.csv -f all -p 4
```

Para generar solo características temporales:

```bash
py Aprendiendo6.py -i dataset_audio -o dataset_temp.csv -f temp
```

Para generar solo características frecuenciales:

```bash
py Aprendiendo6.py -i dataset_audio -o dataset_frec.csv -f frec
```

---------------------------------------------------------------------------------

## Notas importantes

- El programa solo procesa archivos con extensión `.wav`.
- Los archivos que no sean `.wav` se ignoran.
- Si un audio da error al cargarse, el programa muestra un mensaje y continúa con el siguiente.
- No se recomienda usar `--display` al procesar todo el dataset, ya que abriría una gráfica por cada audio.
- La clase se toma automáticamente del nombre de la carpeta.
- El CSV generado está preparado para ser usado en WEKA.
- La columna `clase` debe seleccionarse como atributo objetivo en WEKA.

---------------------------------------------------------------------------------

## Finalidad del proyecto

La finalidad del proyecto es crear un flujo completo de clasificación de audio:

```text
Audios organizados por categorías
                ↓
Extracción automática de características con Python
                ↓
Generación de un CSV compatible con WEKA
                ↓
Entrenamiento y evaluación de clasificadores en WEKA
```

Este sistema permite comprobar si las características extraídas de los audios son suficientes para distinguir entre distintas clases sonoras, como sonidos de ambiente, comentarista o pasos.
