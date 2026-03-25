# Practica2_Unidad2_215510
Modelo de clasificación de imagenes para uso de cualquier modelo entrenado en formato Keras

# Clasificador de Imagenes con IA — Guia de Uso - Arturo Pérez González 215510

Sistema de clasificacion de imagenes. Incluye el modelo y el archivo de etiquetado que se realizo por el alumno el cual esta entrenado para clasificar iconos de las piezas de ajedrez, un servidor local en Python y una interfaz web para clasificar imagenes cargando el modelo entrenado.

---

## Archivos del sistema

| Archivo | Descripcion |
|---|---|
| `iconos_ajedrez_Perez_Arturo215510.ipynb` | Notebook de entrenamiento (se ejecuta en Google Colab) |
| `servidor.py` | Servidor local que carga el modelo y atiende la interfaz |
| `clasificador.html` | Interfaz web para clasificar imagenes |
| `labels.json` | Arcgivo de etiquetas para prueba |
| `modelo_iconos_ajedrez.keras` | Modelo generado por el alumno para la clasificacion |

---

## Requisitos

- Python 3.9 o superior
- Navegador web moderno (Chrome, Firefox, Edge)
- Conexion a internet (solo para la primera instalacion de dependencias)

---

## Paso 1 — Configurar el entorno local

Se recomienda usar un entorno virtual para no afectar otras instalaciones de Python en tu sistema.

---

### Windows (VSCode o terminal)

**Abrir la terminal correcta en VSCode:**
`Ver → Terminal` o atajo `Ctrl + `` ` ``

**Crear el entorno virtual:**
```bat
python -m venv venv
```

**Activar el entorno (usar cualquiera de los dos):**
```bat
venv\Scripts\activate
source venv/Scripts/activate
```

Sabras que esta activo porque la terminal mostrara `(venv)` al inicio de la linea.

**Instalar dependencias:**
```bat
pip install flask flask-cors tensorflow pillow numpy
```

> Si ves el error `python no se reconoce como comando`, instala Python desde [python.org](https://www.python.org/downloads/) y asegurate de marcar **"Add Python to PATH"** durante la instalacion.

---

### Linux (terminal)

**Abrir una terminal** en la carpeta del proyecto. Si usas VSCode: `Ver → Terminal` o `Ctrl + `` ` ``

**Crear el entorno virtual:**
```bash
python3 -m venv venv
```

**Activar el entorno:**
```bash
source venv/bin/activate
```

Sabras que esta activo porque la terminal mostrara `(venv)` al inicio de la linea.

**Instalar dependencias:**
```bash
pip install flask flask-cors tensorflow pillow numpy
```

> En algunas distribuciones puede ser necesario instalar primero: `sudo apt install python3-venv python3-pip`

---

## Paso 2 — Iniciar el servidor

Con el entorno virtual activo y desde la carpeta del proyecto:

**Windows:**
```bat
python servidor.py
```

**Linux:**
```bash
python3 servidor.py
```

Si el servidor inicia correctamente veras esto en la terminal:

```
=======================================================
  Servidor Clasificador de Imagenes
  http://127.0.0.1:5050
=======================================================
```

> Deja esta terminal abierta mientras uses la interfaz. Para detener el servidor presiona `Ctrl + C`.

---

## Paso 3 — Usar la interfaz web

1. Abre el archivo `clasificador.html` directamente en tu navegador (doble clic)
2. Presiona el boton **"conectar"** en la barra superior — el indicador debe volverse verde
3. Sigue los tres pasos en la barra lateral:

| Paso | Que hacer |
|---|---|
| **01 — Modelo** | Selecciona tu archivo `.keras` |
| **02 — Etiquetas** | Carga el `labels.json` generado por Colab, o escribe las clases manualmente separadas por linea |
| **03 — Imagen** | Sube la foto que quieres clasificar |

**NOTA: Si deseas utilizar el modelo adjunto de los iconos de ajedrez deberas descargar el archivo keras y el json (labels) para cargarlos en el html**

4. Presiona **"Clasificar"** — vera el nombre de la clase predicha y el porcentaje de confianza

> Puedes cambiar de modelo en cualquier momento volviendo al Paso 01 y cargando un archivo `.keras` diferente.

---

## Desactivar el entorno virtual

Cuando termines de usar el sistema, puedes desactivar el entorno con:

```bash
deactivate
```

Esto aplica igual en Windows y Linux. La proxima vez que quieras usar el sistema solo necesitas activarlo de nuevo (Paso 2) y ejecutar `servidor.py` (Paso 3).

---

## Solucion de problemas frecuentes

**"El servidor dice sin conexion"**
Verifica que `servidor.py` este corriendo en la terminal y que el puerto coincida con el que muestra la interfaz (por defecto 5050).

**"Error al cargar el modelo"**
Confirma que el archivo tenga extension `.keras` o `.h5` y que el servidor este activo. Modelos grandes (>100 MB) pueden tardar unos segundos en cargarse.

**"pip install falla en tensorflow"**
Asegurate de tener Python 3.9–3.11. TensorFlow actualmente no soporta Python 3.12 en todas las plataformas. Puedes verificar tu version con `python --version`.

**"No se activa el entorno en Windows"**
Si PowerShell bloquea la activacion, ejecuta primero:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Luego intenta activar el entorno nuevamente.

**"La clasificacion da resultados incorrectos"**
Verifica que el `labels.json` corresponde al mismo modelo `.keras` que cargaste. Cada modelo tiene su propio archivo de etiquetas.
