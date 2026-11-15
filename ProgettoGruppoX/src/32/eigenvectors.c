#include <stdio.h>
#include <stdlib.h>
#include <common.h>


type* eigenvectors(const type* M, params* p){

    type* U = calloc(p->N * p->c, sizeof(type));

    for (int i = 0; i < p->N; i++){
        for (int j = 0; j < p->c; j++){
            if( i % p->c == j ){
                U[i * p-> c + j] = 1;
            }
            else{
                U[i * p-> c + j] = 0;
            }
        }
    }

    orthonormalize(U);
    type* Z = calloc(p->N * p->c, sizeof(type));
    type sum = 0.0;

    for (int t = 0; t < p->eig_iters; t++){
        for(int i = 0; i < p->N; i++){
            for(int j = 0; j < p->c; j++){
                sum = 0.0;
                for(int k = 0; k < p->N; k++){
                    sum += M[i *p->N + k]*U[k * p->c + j];
                }
            Z[i * p->c + j] = (U[i * p->c + j] + sum) / 2;
            }
        }

        orthonormalize(Z); // chiamo la funzione orthonormalize supponendo che dentro la funzione vengano aggiornati i valori di Z
                            

        type* temp = U;   // Salvo il vecchio indirizzo U
        U = Z;            // Il nuovo U punta all'ortonormalizzazione di Z (la t-esima)
        Z = temp;        // Z prende il valore del vecchio puntatore U, in questo modo può essere sovrascritto tranquillamente, se non 
                        // avessimo fatto così, U e Z avrebbero puntato allo stesso array in memoria e avrei ad ogni iterazione, modificato i valori sia di Z che di U
                        // ma Z usa i valori di U che non devono cambiare durante il ciclo, se al ciclo successivo usa un valore sovrascritto, il risultato è falsato.

    
    }
    free(Z); // mi è servito per la computazione interna alla funzione, lo dealloco in quanto non mi serve più.
    return U;
}

