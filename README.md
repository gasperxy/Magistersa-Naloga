# Magistersa-Naloga

To je repozitorij z vsemi datotekami za magistersko nalogo z naslovom **1-2-3 Domneva**.

Za uporabljanje kodo predpostavljam, da je na računalniku nameščena distribucija pythona Anaconda. Koda bi delovala tudi brez celotnega paketa Anaconda, vedar bi bili potrebni malenkost drugačni ukazi.


### Kloniramo repozitorij

`git clone https://github.com/gasperxy/Magistersa-Naloga.git`

### Ustvarimo conda okolje

Premaknimo v mapo z kodo: `cd Koda`

Ustvarimo okolje iz yml datoteke: `conda env create -f mag_env.yml`

### Dodamo conda okolje v jupyter notebook.

Za uporabo Jupyter notebook potrebujemo naložen **Jupyter** in **ipykernel**.

Aktiviramo narejeno okolje: `source activate Koda`

Dodamo okolje v jupyter: `python -m ipykernel install --user --name Koda`

Nato zaženemo zvezek:

`jupyter notebook`
