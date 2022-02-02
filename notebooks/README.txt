Si ya se ha ejecutado el frontend saltar directamente al paso 10
1. descargar una imagen de ubuntu 21.10 
2. descargar e instalar virtualbox
3. seleccionar e instalar la imagen de ubuntu
(instalación normal e instalar programas de terceros para hardware de gráficos y de wifi y formatos multimedia adicionales)
* Nota hay que asignar suficiente espacio, unos 50GB
4. Instalar las actualizaciones del sistema antes de empezar
5. Instalar git con sudo apt install git
6. instalar wheel con sudo apt install python-wheel-common
7. Una vez instalada la máquina virtual clonar el repositorio 
https://github.com/JavierHernandezSanchez/TFM.git
8. instalar venv con apt install python3.9-venv
9. sudo apt install python3-pip
10. crear un entorno virtual con python3 -m venv notebookenv
11. activar el entorno virtual con source notebookenv/bin/activate