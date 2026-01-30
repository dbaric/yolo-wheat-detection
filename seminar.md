# Seminar: Detekcija klasova pšenice (Global Wheat Detection)

Opis rada u bilježnici `main.ipynb`: što se radilo, kako i zašto. Rezultati i brojke slijede izvršavanje bilježnice.

---

## 1. Cilj i podaci

Zadatak je detekcija klasova pšenice (wheat heads) na snimkama polja, u sklopu teme natjecanja [Global Wheat Detection na Kaggleu](https://www.kaggle.com/competitions/global-wheat-detection). Podaci dolaze s tog natjecanja: slike u JPG formatu i oznake u YOLO formatu (jedna `.txt` datoteka po slici). U oznakama se koristi jedna klasa (wheat head); svaki red u labelu sadrži `class_id`, te normalizirane koordinate centra i širine/visine okvira (`cx`, `cy`, `w`, `h` u rasponu 0-1).

U bilježnici se redom obrađuju: metrike evaluacije detekcije (IoU, preciznost, odziv, F1), učenje osjetljivo na cijenu (cost-sensitive learning), stablo odluke i inženjering značajki kao baseline, te YOLO za stvarnu detekciju (unaprijed trenirani model i fino ugađanje). Na kraju se uspoređuju početno i završno treniranje modela.

---

## 2. Priprema i čišćenje podataka

Provjerava se usklađenost slika i oznaka: broj slika u `train/` i broj label datoteka u `labels/`, te podudarnost imena (svaka slika ima odgovarajuću `.txt` oznaku). Rezultat je 3422 slike, 3422 labela, i potpuna podudarnost imena (`match = True`). Odluka je da su podaci već usklađeni i da nije potrebno uklanjati ili ispravljati primjerke.

---

## 3. Metrike evaluacije detekcije

Uvedene su mjere za usporedbu predviđenih okvira s ground truth oznakama:

- **IoU (Intersection over Union)**: omjer presjeka i unije dva pravokutna okvira; računa se iz normaliziranih koordinata (centar, širina, visina). Koristi se za odlučivanje je li predviđeni okvir „pogodio” stvarni objekt.
- **Uparivanje okvira**: za zadani prag IoU (npr. 0.5) predviđeni se okviri uparuju s ground truth okvirima (najbolji IoU iznad praga); preostali predviđeni broje se kao lažno pozitivni (FP), preostali ground truth kao lažno negativni (FN).
- **Preciznost** = TP / (TP + FP), **odziv** = TP / (TP + FN), **F1** kao harmonijska sredina preciznosti i odziva.

Za ilustraciju metrika napravljene su sintetičke predikcije: za prvih 10 slika ground truth okviri su malo pomaknuti (šum), a dodani su i nasumični okviri. Na tom primjeru dobiveno je: preciznost 0,0362, odziv 0,0500, F1 0,0420; TP = 18, FP = 479, FN = 342. Iz toga se vidi da bez pravog modela metrike ostaju niske i da je važno imati dobro uparivanje okvira (IoU prag) za interpretaciju preciznosti i odziva.

---

## 4. Učenje osjetljivo na cijenu (Cost-Sensitive Learning)

Radi se o tome kako izbor **praga pouzdanosti** (confidence threshold) utječe na preciznost, odziv i na „cijenu” koju želimo minimizirati. Pretpostavlja se da je propust detekcije (FN) skuplji od lažne detekcije (FP); u bilježnici je cijena definirana kao `fn * 10 + fp * 1`. Za niz pragova (npr. 0,1, 0,2, …, 0,9) računaju se preciznost, odziv i ukupna cijena na istom malom skupu predikcija. Optimalan prag (po minimalnoj cijeni) u tom eksperimentu ispada 0,1. Crtaju se krivulje: preciznost, odziv i cijena u ovisnosti o pragu. Izbor praga izravno utječe na trade-off između preciznosti i odziva i na poslovnu metriku (cijenu).

---

## 5. Stablo odluke (baseline klasifikacija)

Kao jednostavan baseline uvodí se **klasifikacija na razini slike**: sadrži li slika pšenicu ili ne. Iz svake slike izvlače se značajke:

- **Boja**: srednja vrijednost i standardna devijacija po kanalima R, G, B.
- **Rubovi**: siva slika, Canny rubovi, gustoća rubova (udio piksela koji su rub).
- **Tekstura**: GLCM (gray-level co-occurrence matrix) na sivoj slici, kontrast i homogenost.

S tim značajkama uči se **stablo odluke**; podaci se dijele na skup za učenje i test (npr. 70 % / 30 %). Na malom testnom uzorku (6 primjeraka) classification report pokazuje preciznost i odziv 1,00 za klasu „sadrži pšenicu”. Značajke na tom uzorku dovoljno su diskriminativne za binarnu klasifikaciju, ali baseline ne daje pozicije okvira i ne mjeri se na isti način kao detekcijski model (IoU, mAP).

---

## 6. Inženjering značajki i odabir (RFE)

Proširuje se skup značajki (dodavanjem prosjeka boja u središnjem dijelu slike) i primjenjuje se **RFE (Recursive Feature Elimination)** s Random Forest estimatorom: odabire se fiksan broj najvažnijih značajki. Rezultat je lista imena odabranih značajki. Time se smanjuju šum i redundantnost i priprema se manji, interpretabilni skup značajki za klasifikaciju/stablo odluke; ne mijenja se YOLO pipeline, koji koristi vlastite značajke.

---

## 7. YOLO s unaprijed treniranim modelom (bez fine-tuninga)

Učitavaju se unaprijed trenirani YOLO modeli (npr. YOLO11n i YOLO11m) trenirani na COCO skupu. Na slikama iz wheat skupa oni detektiraju **samo COCO klase** (npr. osobe, ptice), a ne klasove pšenice. U bilježnici se mjeri prosječno vrijeme inferencije i ukupan broj detekcija po modelu. Unaprijed trenirani model „iz kutije” nije primjenjiv za detekciju pšenice; potrebno je fino ugađanje na označenim podacima za wheat heads.

---

## 8. YOLO fino ugađanje

Podaci se dijele na train / val / test (npr. 70 % / 15 % / 15 %) u YOLO formatu (slike u `images/`, oznake u `labels/`). Koristi se YOLO11 (npr. `yolo11n.pt`) kao početni model.

- **Početno (POC) treniranje**: manji broj epoha (5), manja rezolucija slike (npr. 320), veći batch (npr. 64 ili 128), uključen mixed precision (AMP). Koristi se za brzu provjeru pipelinea i da li model uopće uči.
- **Završno treniranje**: više epoha (50), veća rezolucija (640), manji batch (16), pune augmentacije. Očekuje se bolja točnost i stabilnost.

Nakon treniranja (u bilježnici je korišten početni model iz POC faze) evaluacija na validacijskom skupu daje: **mAP@50 ≈ 0,757**, **mAP@50-95 ≈ 0,343**, **preciznost ≈ 0,833**, **odziv ≈ 0,687**. Fino ugađani YOLO dobro detektira wheat heads na ovom skupu. Grafovi tijekom treniranja (preciznost, odziv, mAP@50, mAP@50-95 po epohama) prikazani su u `docs/fine_tuning.png`.

---

## 9. Usporedba: unaprijed trenirani vs. fino ugađani model

Na istom validacijskom skupu (513 slika, 21356 instanci) uspoređuju se:

- **Unaprijed trenirani YOLO (COCO)**: na wheat podacima praktički ne detektira pšenicu: mAP@50 ≈ 0,0017, mAP@50-95 ≈ 0,0003, preciznost ≈ 0,0034, odziv ≈ 0.
- **Fino ugađani model (početni POC)**: mAP@50 ≈ 0,7567, mAP@50-95 ≈ 0,3427, preciznost ≈ 0,8334, odziv ≈ 0,6866.

Relativno poboljšanje za fino ugađani model je vrlo veliko (npr. mAP@50 za nekoliko desetaka tisuća postotaka), COCO ne sadrži klasu „wheat head”, pa je to i očekivano.

Zaključak: za detekciju pšenice nužno je fino ugađanje (ili treniranje od nule) na označenim wheat podacima.

---

## 10. Primjer korištenja (inferencija)

Učitava se najbolji trenirani model (početni iz `train/` ili završni iz `train2/` ako postoji) i pokreće se inferencija na nekoliko testnih slika. Za svaku sliku prikazuje se broj detektiranih wheat heads, vrijeme obrade te primjer okvira s koordinatama i pouzdanošću. Primjer iz bilježnice: na jednoj slici 15 detekcija (npr. pouzdanosti oko 0,71-0,50), na drugoj 37 detekcija (npr. 0,83-0,76), na trećoj 22 detekcije. Model u praksi daje okvire i pouzdanosti spremne za daljnju uporabu (npr. filtriranje po pragu ili vizualizacija).

---

## Sažetak

Redom je obavljeno: provjera podataka (3422 slike s oznakama); uvođenje metrika detekcije (IoU, preciznost, odziv, F1) i cost-sensitive odluke o pragu; baseline s stablima odluke i RFE odabirom značajki; pokazano da unaprijed trenirani YOLO na COCO ne detektira pšenicu; fino ugađanje YOLO11 na wheat skupu s početnim (POC) i opcionalno završnim treniranjem; usporedba metrika (mAP, preciznost, odziv) između unaprijed treniranog i fino ugađanog modela; te primjer inferencije na testnim slikama. Svi brojčani rezultati navedeni u ovom tekstu odgovaraju izlazima iz bilježnice `main.ipynb`.
