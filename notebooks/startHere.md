# Quantum Error Correcting Codes, mainly decoding them

This is the entry point to this repository. Hopefully, it would allow the reader who knows some classical ECC and some linear algebra to understand how to carry their knowledge from classical ECC to quantum ECC.

I found that most books teach Quantum computing, maybe some Quantum mechanics, and at the very end give some qECC examples.

I'm hoping to do the opposite, i.e., first get the reader to master qECC, then learn some quantum computing where qECC are used.


## Some pointers
To start reading, go to:

linearCodesOverF2.ipynb - that should calibrate you to notation and some basic ideas from classical error correcting codes.

Then the most natural place to continue is: orderedStatisticsDecoding.ipynb

Next review an implementation of a belief propagation decoder here: minSumExample.ipynb and you can check under the hood in minSum.py

beliefPropagationAndOsd.ipynb



polinomialCodes.py contains codes that were introduced in a paper by Panteleev and Kalachev, but also IBM
relaybp.py is my own implementation of relay BP from a paper by IBM.