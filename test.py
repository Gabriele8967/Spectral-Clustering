import gruppoX

import argparse
import sys
import os
import time
import numpy as np
import struct
from pathlib import Path

def empty_aligned(shape, dtype, alignment):
    # Alloca un array NumPy il cui primo elemento e' allineato ad 'alignment' byte
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    buf = np.empty(nbytes + alignment, dtype=np.uint8)
    offset = (-buf.ctypes.data) % alignment
    return buf[offset:offset + nbytes].view(dtype).reshape(shape)

def load_ds2(filename, dtype, alignment):
    with open(filename, 'rb') as f:
        # Leggi header (n, d)
        header = f.read(8)  # 4 + 4 byte
        n, d = struct.unpack('ii', header)  # 'ii' = 2 int32
        # Alloca array allineato
        aligned = empty_aligned((n, d), dtype=dtype, alignment=alignment)
        # Leggi dati direttamente nell'array allineato
        f.readinto(aligned.data)
    return aligned

def save_ds2(data, ds2name):
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, d = data.shape
    with open(ds2name, 'wb') as f:
        f.write(struct.pack('ii', n, d))
        data.tofile(f)

def csv_to_ds2(csvname, dtype, delimiter=','):
    path = Path(csvname)
    ds2name = path.with_suffix(f".{np.dtype(dtype).itemsize * 8}.ds2")
    data = np.loadtxt(csvname, delimiter=delimiter).astype(dtype)
    save_ds2(data, ds2name)
    return ds2name

def load_file(file_name, dtype, alignment, delimiter=','):
    path = Path(file_name)
    ext = path.suffix.lower()
    if ext not in ['.csv', '.ds2']:
        raise ValueError("Formato file non riconosciuto")
    if ext == ".csv":
        # Il file .ds2 dipende dalla precisione (float32 o float64)
        ds2name = path.with_suffix(f".{np.dtype(dtype).itemsize * 8}.ds2")
        if not ds2name.is_file():
            csv_to_ds2(file_name, dtype, delimiter)
        file_name = ds2name
    return load_ds2(file_name, dtype, alignment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test del progetto Spectral Clustering')

    # Definizione dei parametri
    parser.add_argument('DS', help='nome file dataset (csv o ds2)')
    parser.add_argument('c', type=int, help='numero di cluster')
    parser.add_argument('T', type=int, help='numero di iterazioni per il calcolo degli autovettori')
    parser.add_argument('I', type=int, help='numero massimo di iterazioni del k-means')
    parser.add_argument('a', type=str, choices=['32', '64', '64omp'], help='float+sse, double+avx, double+avx+openmp')
    parser.add_argument('-s', '--silent', action='store_true', help='modalità silenziosa')

    # Parsing degli argomenti
    args = parser.parse_args()
    if(not args.silent): print(args)

    bits = int(args.a.replace('omp',''))

    # Validazione file DS
    if not os.path.exists(args.DS):
        print(f"Errore: Il file {args.DS} non esiste")
        sys.exit(1)

    ds_path = Path(args.DS)
    if ds_path.suffix.lower() not in ['.csv', '.ds2']:
        print(f"Errore: Il file {args.DS} deve avere estensione .csv o .ds2")
        sys.exit(1)

    # Validazione parametri numerici
    if args.c < 2:
        print("Errore: c deve essere un intero maggiore o uguale a 2")
        sys.exit(1)

    if args.T <= 0:
        print("Errore: T deve essere un intero positivo")
        sys.exit(1)

    if args.I <= 0:
        print("Errore: I deve essere un intero positivo")
        sys.exit(1)

    DS = load_file(args.DS, dtype=f'float{bits}', alignment=int(bits/2))

    spectral = gruppoX.SpectralClustering(n_clusters=args.c, eig_iters=args.T,
                                          km_iters=args.I, version=args.a,
                                          silent=args.silent)

    # Le verifiche che dipendono dal dataset (N > K, c <= N) sono nel modulo C
    try:
        # =========================
        labels = spectral.fit_predict(DS)
        # =========================
    except ValueError as e:
        print(f"Errore: {e}")
        sys.exit(1)

    if not args.silent:
        print("LABELS:")
        print(labels)
        print("CLUSTER SIZES:")
        print(np.bincount(labels, minlength=args.c))

    # EIG: matrice Y (C/assembly, oggetto della competizione)
    # KM:  k-means (scikit-learn)
    if not args.silent:
        print(f"EIG time: {spectral.eig_time_:.5f} seconds")
        print(f"KM  time: {spectral.km_time_:.5f} seconds")
    else:
        print(spectral.eig_time_)
        print(spectral.km_time_)

    size = f"size-{DS.shape[0]}x{DS.shape[1]}_c-{args.c}"
    save_ds2(spectral.embedding_, f"Y_{args.a}_{size}.ds2")
    save_ds2(labels, f"labels_{args.a}_{size}.ds2")
