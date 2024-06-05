# Koopman-VAE
The goal of this repository is to implement a VAE that integrates a Koopman layer within it in order
to receive better entanglements. 

## Roadmap 
The roadmap of the project is as follows: 
- [X] Copy the base implementation of C-DSVAE
- [X] Remove the mutual information losses from it (as they are irrelevant for our purpose)
- [ ] Connect the LSTM directly to the decoder.
- [ ] Add Koopman layer and further losses in the "middle".



