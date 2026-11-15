#ifndef SPECTRAL_COMMON
#define SPECTRAL_COMMON

#define	type	float
#define	align	16

#define	MATRIX		type*
#define	VECTOR		type*

// Numero di vicini del grafo K-NN: costante del problema
#define	K	8

typedef struct{
	// ---------------- Ingresso ----------------
	MATRIX DS;					// dataset (N x D)
	// ---------------- Uscita ----------------
	MATRIX Y;					// i c autovettori dominanti (N x c, row-major);
								// al termine le righe sono normalizzate (matrice Y)
	// ---------------- Parametri e dimensioni ----------------
	int N;						// numero di righe del dataset
	int D;						// numero di colonne/feature del dataset
	int c;						// numero di cluster (= numero di autovettori)
	int eig_iters;				// numero di iterazioni T per il calcolo degli autovettori
	int silent;					// modalita' silenziosa

	// ---------------- Strutture Dati ------------

	//NOTA: ARRAY DA ALLOCARE QUANDO CREO LA STRUCT
	//
							    // modello i mie K-NN in questo modo:
	int* idx;					// a parità di indice, idx[i] ha l'indice del PTO i
	VECTOR dst;					// mentre dst[i] ha la distanza da me al PTO i

	VECTOR sigma;				//dimensione N
	MATRIX W;					//ALLOCARE E INIZIALIZZARE CON TUTTI GLI ELEMENTI A 0
	VECTOR g;					//dimensione N
	MATRIX M;					//ALLOCARE E INIZIALIZZARE CON TUTTI GLI ELEMENTI A 0
	MATRIX U;				    //DIMENSIONE N * c. INIZIALIZZARE COME DEFINITO DALL'EQ. 8

} params;

#endif
