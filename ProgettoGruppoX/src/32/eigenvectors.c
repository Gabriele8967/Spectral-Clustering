#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include <string.h>


void eigenvectors(params* input){

    type* U = calloc(input->N * input->c, sizeof(type));

    for (int i = 0; i < input->N; i++){
        for (int j = 0; j < input->c; j++){
            if( i % input->c == j ){
                U[i * input-> c + j] = 1;
            }
            else{
                U[i * input-> c + j] = 0;
            }
        }
    }

    orthonormalize(U);
    type* Z = calloc(input->N * input->c, sizeof(type));
    type sum = 0.0;

    for (int t = 0; t < input->eig_iters; t++){
        for(int i = 0; i < input->N; i++){
            for(int j = 0; j < input->c; j++){
                sum = 0.0;
                for(int k = 0; k < input->N; k++){
                    sum += input->M[i *input->N + k]*U[k * input->c + j];
                }
            Z[i * input->c + j] = (U[i * input->c + j] + sum) / 2;
            }
        }

        orthonormalize(Z); // chiamo la funzione orthonormalize supponendo che dentro la funzione vengano aggiornati i valori di Z
                            

        type* temp = U;   // Salvo il vecchio indirizzo U
        U = Z;            // Il nuovo U punta all'ortonormalizzazione di Z (la t-esima)
        Z = temp;        // Z prende il valore del vecchio puntatore U, in questo modo può essere sovrascritto tranquillamente, se non 
                        // avessimo fatto così, U e Z avrebbero puntato allo stesso array in memoria e avrei ad ogni iterazione, modificato i valori sia di Z che di U
                        // ma Z usa i valori di U che non devono cambiare durante il ciclo, se al ciclo successivo usa un valore sovrascritto, il risultato è falsato.

    
    }
    
    memcpy(input->Y, U, input->N * input->c * sizeof(type));
    free(U);
    free(Z);  // mi è servito per la computazione interna alla funzione, lo dealloco in quanto non mi serve più.

}

