# Relay BP 
Based on the paper: Improved belief propagation is sufficient for real-time decoding of quantum memory https://arxiv.org/pdf/2506.01779


Relay BP starts with an implementation of belief propagation, but with a leak factor for every marginal.


Some points to keep in mind and investigate:
"Figure 2 compares both XZ- and XYZ-decoding to standard benchmarks. For the BB codes, we compare to BP+OSD+CS-10 (via the LDPC package [47] using 10,000 BP iterations, with a combination-sweep (CS) of10);"

The number 10,000 is unsusual, even for the classical case. In fact in https://arxiv.org/pdf/1904.02703 the number of iterations used was 32.

Add figure describing decoding as a function of number of iterations here:


Another thing that was odd is the small sample set of codes used to test relay BP on.
Basically the surface code, gross code and two gross code (maybe from here : T. J. Yoder, E. Schoute, P. Rall, E. Pritchett, J. M. Gambetta, A. W. Cross, M. Carroll, and M. E. Beverland, Tour de gross: A modular quantum computer based on bivariate bicycle codes https://arxiv.org/pdf/2506.03094) 
