# Dengue Analytics — proyecto académico

## Descripción
Aplicación Flask que compara dos clasificadores binarios: regresión logística y red neuronal artificial MLP. La variable objetivo procede de `GROUPS: DENGUE FEVER OR COMPLICATED DENGUE` (1 = dengue fever; 2 = complicated dengue, según el dataset/artículo).

## Archivos
- `app.py`: servidor y endpoint `/predict`.
- `train_models.py`: limpieza, ingeniería de variables, entrenamiento y evaluación.
- `data/dengue_limpio.csv`: datos depurados y variables derivadas.
- `models/`: modelos serializados y métricas del conjunto de prueba.
- `templates/` y `static/`: interfaz web.
- `Procfile` y `render.yaml`: configuración de despliegue en Render.

## Preparación realizada
1. Se cargaron 303 filas y 25 columnas del Excel original.
2. Se detectaron y eliminaron filas duplicadas exactas (1 fila).
3. Se inspeccionaron columnas constantes; no se encontraron columnas constantes en el archivo recibido.
4. Se seleccionó la variable objetivo y cinco indicadores clínico-laboratoriales resaltados por el artículo suministrado: PCV, recuento de plaquetas, WBC más alto, ALT y BP al ingreso.
5. Se crearon variables binarias: PCV >= 42, plaquetas <= 75000, WBC >= 7000 y ALT >= 70. Se derivó además `BP_CODE_2` conservando el código original 2 como indicador, sin asignarle una etiqueta clínica definitiva porque no se adjuntó el archivo de diccionario/codificación.
6. El entrenamiento utiliza imputación de valores faltantes, escalado estándar y partición estratificada 80/20. En este subconjunto seleccionado no hay faltantes después de la selección, pero el pipeline conserva la imputación para robustez.

## Importante sobre la codificación
El artículo define hipotensión como predictor, pero el Excel adjunto solo contiene `BP AT ADMISSION` con códigos 1 y 2. No se adjuntó el archivo Readme de codificación mencionado en el artículo. Por eso la app solicita el código numérico y crea `BP_CODE_2`; antes de una interpretación clínica debe verificarse si el código 2 representa hipotensión en el diccionario original.

## Ejecutar localmente
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Abrir http://127.0.0.1:5000

Para volver a entrenar:
```bash
python train_models.py
```

## Publicación en internet (Render)
1. Descomprime el ZIP y sube la carpeta a un repositorio GitHub.
2. En https://render.com crea una cuenta y elige **New + > Blueprint**.
3. Conecta el repositorio. Render detectará `render.yaml` y desplegará el servicio.
4. Al finalizar, Render entregará una URL pública `https://...onrender.com`. Copia esa URL para el buzón de entrega.
5. El plan gratuito puede tardar en iniciar después de periodos sin uso.

## Advertencia
Proyecto académico experimental. No es un dispositivo médico, no diagnostica dengue y no debe usarse para decidir tratamientos. El dataset tiene 303 registros de un estudio específico; los resultados requieren validación externa y revisión profesional antes de cualquier aplicación clínica.
