#include <stdlib.h>
#include <stdio.h>
#include <xmmintrin.h>
#include <math.h>
#include <omp.h>

#include "common.h"

static void initialize_KNN(params* input) {

	for (int i = 0; i < input->N * K; i++) {
			input->idx[i] = -1;
			input->dst[i] = INFINITY;
	}
};

static type calculate_dst(const type* vi, const type* vj, const int D) {

	type dst = 0;
	for (int t = 0; t < D; t++) {
		type partial_dst = vi[t] - vj[t];
		dst += partial_dst * partial_dst;
	}

	return dst;
}

static void insert_new_NN(params* input, const int i, const int j, const type dst_j) {

	//Partendo dall'inizio, scansioni il vettore dst[] dei K-NN vicini del PTO i
	//Ogni volta che trovo un PTO con distanza minore di quella del PTO che voglio inserire,
	//incremento position
	unsigned position = 0;
	for (int t = i * K; t < ((i * K) + K); t++) {
		if (input->dst[t] < dst_j) {
			position++;
		}
	}

	//A questo punto, position contiene la posizione esatta in cui inserire il nuovo vicino
	//Quindi, partendo dal penultimo, li sposto tutti verso destra, in modo tale che
	//libero lo slot dove inserire il nuovo vicino
	for(int k = K - 1; k > position; k--){
		input->idx[k] = input->idx[k - 1];
		input->dst[k] = input->dst[k - 1];
	}

	//Inserisco il nuovo vicino
	input->idx[position] = j;
	input->dst[position] = dst_j;
}

static void w(const int i, const int j, params* input) {
	type sigma_i = input->sigma[i];
	type sigma_j = input->sigma[j];

	type sisj = sigma_i * sigma_j;
	type dstq_vivj = input->dst[i * K + j];				//La distanza è già al quadrato, è stata salvata così

	type wij = sisj / (sisj + dstq_vivj);

	input->W[i * input->N + j] = wij;
	input->W[j * input->N + i] = wij;
}

static void g(const int i, params* input) {

	type grade = 0;
	for (int j = 0; j < input->N; j++) {
		grade += input->W[i * input->N + j];
	}

	input->g[i] = grade;
}

static void m(const int i, const int j, params* input) {


	//NOTA BENE
	//NELLE ALTRE VERSIONI, CALCOLARE LA MATRICE M USANDO LA SECONDA
	//DEFINIZIONE, OVVERO IL PRODOTTO TRA G^-1/2 * W * G^-1/2
	type wij = input->W[i * input->N + j];
	type gigj = input->g[i] * input->g[j];
	type mij = wij / sqrt(gigj);

	input->M[i * input->N + j] = mij;
	input->M[j * input->N + i] = mij;

}

// Funzione di esempio scritta in assembly (spectral32.nasm)

extern void prova(params* input);

/*
*	knngraph
*	========
*
*	Per ogni punto del dataset calcola i K vicini e le relative
*	distanze al quadrato, memorizzandoli in input->nn_id e input->nn_dist.
*	Le liste sono ordinate per distanza; a parita' vince l'indice minore.
*/
void knngraph(params* input){

	initialize_KNN(input);									//CHIEDERE - Inizializzazione totale e non ad ogni iterazione i come nello pseudo-codice.
	for (int i = 0; i < input->N; i++) {
		const type* vi = &input->DS[i * input->D];			//Prendo il PTO i di dimensione D
		for (int j = 0; j < input->D; j++) {
			if (j != i) {
				const type* vj = &input->DS[i * input->D + j];
				const type dst_j = calculate_dst(vi, vj, input->D);
				const type dst_MAX = input->dst[i * K + (K - 1)];			//Assumo che la distanza massima sia quella del PTO più lontano nel vettore della distanza
				if (dst_j < dst_MAX) {								        //Questo if le prime K volte sarà sempre vero, quindi forse si potrebbe valutare di fare altro
					insert_new_NN(input, i, j, dst_j);
				}
			}
		}
	}
}

/*
*	affinity
*	========
*
*	A partire dalle liste dei K vicini costruisce il grafo simmetrico,
*	le scale locali, la matrice di affinita' W, i gradi e la matrice
*	normalizzata M, memorizzata in forma sparsa (input->row_ptr,
*	input->col_idx, input->val, input->nnz).
*/
void affinity(params* input){

	int i;
	for (i = 0; i < input->N; i++) {
		input->sigma[i] = sqrt(input->dst[(i * K) + K]);
	}

	//Costruire l'insieme degli archi come scritto nello pseudo-codice non è necessario
	//in quanto nel vettore dst, sono presenti tutti gli archi tra i e i vicini, per ogni i

	for (i = 0; i < input->N; i++) {
		for (int j = 0; j < K; j++) {
			w(i, j, input);
		}
	}

	for (i = 0; i < input->N; i++) {
		g(i, input);
	}

	for (i = 0; i < input->N; i++) {
		for (int j = 0; j < K; j++) {
			m(i, j, input);
		}
	}
}

/*
*	eigenvectors
*	============
*
*	Calcola i c autovettori dominanti di M mediante iterazione
*	ortogonale (input->eig_iters iterazioni) e li memorizza in input->U
*	(N x c, row-major, gia' allocata dal chiamante).
*/
void eigenvectors(params* input){
	// TODO
}

/*
*	normalize
*	=========
*
*	Normalizza ogni riga di input->U in modo che abbia norma unitaria,
*	ottenendo (in place) la matrice Y. Una riga nulla resta nulla.
*/
void normalize(params* input){
	// TODO
}

/*
*	spectral
*	========
*
*	Calcola la matrice Y, ossia la matrice U dei c autovettori dominanti
*	della matrice di affinita' normalizzata del grafo dei K vicini,
*	con le righe normalizzate a norma unitaria.
*	Il clustering delle righe di Y (k-means) e' eseguito in Python.
*
*	input->U deve essere allocata dal chiamante (N x c elementi);
*	al termine contiene Y.
*	Le strutture intermedie allocate qui vanno liberate con spectral_free.
*/
void spectral(params* input){
	knngraph(input);
	affinity(input);
	eigenvectors(input);
	normalize(input);
	// Esempio di chiamata a una funzione assembly: da rimuovere
	prova(input);
}

/*
*	spectral_free
*	=============
*
*	Libera eventuali strutture intermedie
*/
void spectral_free(params* input){
	//deallocare la memoria delle varie matrici/vettori utilizzate
}

int main(int argc, char *argv[]) {
	return 0;
}

