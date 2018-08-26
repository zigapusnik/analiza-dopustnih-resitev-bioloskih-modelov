# Analiza prostora dopustnih rešitev v visokodimenzionalnih dinamičnih modelih bioloških sistemov

Datoteka results_rep_genetic vsebuje vse dopustne rešitve modela biološkega represilatorja, ki smo jih našli z genetskim algoritmom.

Datoteka results_flipflop_genetic vsebuje vse dopustne rešitve modela biološke pomnilne celice D s predpomnenjem, ki smo jih našli z genetskim algoritmom.

Datoteka modelSolver.py preišče in analizira prostor kinetičnih parametrov modela biološkega represilatorja ter pomnilne celice D s predpomnenjem, vse vmesne rezultate pa shrani v datoteki results_rep_genetic in results_flipflop_genetic. Če prostora ne želimo preiskovati, moramo v datoteki modelSolver.py zakomentirati vrstice 810 - 820.

Datoteko modelSolver.py poženemo z ukazom:
  - python modelSolver.py

